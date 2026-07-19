"""Python client SDK for the deployed Document Analyst (Part 3).

TODO: Implement `DocumentAnalystClient` and `AnalystClientError` per Task 3.1:
  - __init__(endpoint_name, host=None, token=None, timeout=120.0, max_retries=3):
    read DATABRICKS_HOST/DATABRICKS_TOKEN from env when not provided.
  - ask(question) -> str
  - ask_streaming(question) -> Iterator[str]   (yield chunks as they arrive)
  - health_check() -> bool                      (True only when endpoint READY)
  - exponential backoff on 429/503, TimeoutError with elapsed time, and wrap HTTP
    errors in AnalystClientError(status_code, message, request_id).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import httpx

_RETRYABLE_STATUS_CODES = {429, 503}


class AnalystClientError(Exception):
    def __init__(self, message: str, status_code=None, request_id=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        if self.request_id is not None:
            parts.append(f"request_id={self.request_id}")
        return " | ".join(parts)


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        resolved_host = host or os.environ.get("DATABRICKS_HOST")
        resolved_token = token or os.environ.get("DATABRICKS_TOKEN")

        if not resolved_host:
            raise OSError(
                "DATABRICKS_HOST not provided and not set in the environment."
            )
        if not resolved_token:
            raise OSError(
                "DATABRICKS_TOKEN not provided and not set in the environment."
            )

        self.endpoint_name = endpoint_name
        self.host = resolved_host.rstrip("/")
        self.token = resolved_token
        self.timeout = timeout
        self.max_retries = max_retries

        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    # ─── internal helpers ────────────────────────────────────────────────

    def _invocations_url(self) -> str:
        return f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"

    def _endpoint_status_url(self) -> str:
        return f"{self.host}/api/2.0/serving-endpoints/{self.endpoint_name}"

    def _request_id_from(self, response: httpx.Response) -> str | None:
        return response.headers.get("x-request-id") or response.headers.get(
            "x-databricks-request-id"
        )

    def _post_with_retry(self, url: str, payload: dict, *, stream: bool = False):
        """POST with exponential backoff on 429/503."""
        attempt = 0
        start = time.monotonic()

        while True:
            try:
                if stream:
                    client = httpx.Client(timeout=self.timeout)
                    request = client.build_request(
                        "POST", url, headers=self._headers, json=payload
                    )
                    response = client.send(request, stream=True)
                else:
                    response = httpx.post(
                        url, headers=self._headers, json=payload, timeout=self.timeout
                    )
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - start
                raise TimeoutError(
                    f"Request to {self.endpoint_name} timed out after "
                    f"{elapsed:.2f}s (timeout={self.timeout}s)"
                ) from exc

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                if stream:
                    response.close()
                wait = 2**attempt
                time.sleep(wait)
                attempt += 1
                continue

            if response.status_code >= 400:
                if stream:
                    response.read()  # <-- MUST read the body before accessing .json()/.text
                request_id = self._request_id_from(response)
                try:
                    message = response.json().get("message", response.text)
                except (json.JSONDecodeError, ValueError):
                    message = response.text
                if stream:
                    response.close()
                raise AnalystClientError(
                    message=message,
                    status_code=response.status_code,
                    request_id=request_id,
                )

            return response

    @staticmethod
    def _extract_answer(body) -> str:
        """Pull the assistant's answer text out of the endpoint's response.

        Path A (mlflow.langchain.log_model, used here) wraps raw LangGraph state
        in a one-element batch list: [{"messages": [...], ...}]. Handle that
        shape first, then fall back to a bare dict (OpenAI-style or unwrapped
        state) for robustness.
        """
        if isinstance(body, list) and body:
            body = body[0]

        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if content:
                return content

        messages = body.get("messages") or []
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                return last.get("content", "")
        return ""

    # ─── public API ──────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True only if the endpoint is in the READY state."""
        try:
            response = httpx.get(
                self._endpoint_status_url(), headers=self._headers, timeout=self.timeout
            )
        except httpx.TimeoutException:
            return False

        if response.status_code != 200:
            return False

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            return False

        state = body.get("state", {})
        return state.get("ready") == "READY"

    def ask(self, question: str) -> str:
        payload = {"messages": [{"role": "user", "content": question}]}
        response = self._post_with_retry(self._invocations_url(), payload)

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise AnalystClientError(
                f"Could not parse response as JSON: {response.text[:500]}"
            ) from exc

        return self._extract_answer(body)

    def ask_streaming(self, question: str) -> Iterator[str]:
        """Yield text chunks as they arrive.

        A models-from-code LangChain endpoint (Path A, used here) does not
        implement true token-by-token streaming — Databricks rejects the
        request outright with 400 "This endpoint does not support streaming."
        rather than returning a non-incremental SSE response. Per the Task 3.1
        caveat, we treat this as a valid outcome: catch that specific failure
        and fall back to a single non-streaming ask() call, yielding the whole
        answer once.
        """
        payload = {
            "messages": [{"role": "user", "content": question}],
            "stream": True,
        }

        try:
            response = self._post_with_retry(self._invocations_url(), payload, stream=True)
        except AnalystClientError as exc:
            if exc.status_code == 400:
                # Endpoint doesn't support streaming at all (Path A model).
                # Fall back to a single full-answer call instead of raising.
                answer = self.ask(question)
                if answer:
                    yield answer
                return
            raise

        content_type = response.headers.get("content-type", "")
        yielded_anything = False

        try:
            if "text/event-stream" in content_type:
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except (json.JSONDecodeError, ValueError):
                        continue

                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    chunk = delta.get("content")
                    if chunk:
                        yielded_anything = True
                        yield chunk
            else:
                body = response.read()
                try:
                    parsed = json.loads(body)
                    answer = self._extract_answer(parsed)
                    if answer:
                        yielded_anything = True
                        yield answer
                except (json.JSONDecodeError, ValueError):
                    pass
        finally:
            response.close()

        if not yielded_anything:
            answer = self.ask(question)
            if answer:
                yield answer