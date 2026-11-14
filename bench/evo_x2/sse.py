"""Bounded incremental SSE parsing and metadata-only OpenAI stream capture."""

import codecs
import hashlib

from .contract import finite
from .fixtures import strict_json


class SSEDecoder:
    """Decode arbitrary UTF-8/network boundaries, including CRLF and comments.

    An event is dispatched only after its blank line. Unterminated events at EOF
    are errors, not successful completions. No network or file access occurs.
    """

    def __init__(self, max_event_bytes=1024 * 1024, max_total_bytes=64 * 1024 * 1024):
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.buffer = ""
        self.data = []
        self.event_bytes = 0
        self.total_bytes = 0
        self.max_event_bytes = max_event_bytes
        self.max_total_bytes = max_total_bytes
        self.started = False

    def _line(self, line):
        if not line:
            result = "\n".join(self.data) if self.data else None
            self.data, self.event_bytes = [], 0
            return result
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self.event_bytes += len(value.encode("utf-8")) + 1
            if self.event_bytes > self.max_event_bytes:
                raise ValueError("SSE event exceeds byte limit")
            self.data.append(value)
        return None

    def feed(self, raw, *, final=False):
        if not isinstance(raw, bytes):
            raise ValueError("SSE input must be bytes")
        self.total_bytes += len(raw)
        if self.total_bytes > self.max_total_bytes:
            raise ValueError("SSE stream exceeds byte limit")
        text = self.decoder.decode(raw, final=final)
        if not self.started and text:
            self.started = True
            text = text.removeprefix("\ufeff")
        self.buffer += text
        events = []
        while True:
            positions = [position for position in (self.buffer.find("\n"), self.buffer.find("\r")) if position >= 0]
            if not positions:
                break
            end = min(positions)
            if self.buffer[end] == "\r" and end == len(self.buffer) - 1 and not final:
                break
            width = 2 if self.buffer[end:end + 2] == "\r\n" else 1
            event = self._line(self.buffer[:end])
            self.buffer = self.buffer[end + width:]
            if event is not None:
                events.append(event)
        if len(self.buffer.encode("utf-8")) > self.max_event_bytes:
            raise ValueError("SSE line exceeds byte limit")
        if final and (self.data or self.buffer.strip()):
            raise ValueError("unterminated SSE event at EOF")
        return events


class StreamCapture:
    """Capture selected timestamps and exact usage, without retaining answer text.

    API completion is intentionally distinct from engine normal completion:
    the upstream API may represent an engine cancellation as finish_reason=stop.
    """

    def __init__(self, send_s, status_code=200, content_type="text/event-stream"):
        if not finite(send_s):
            raise ValueError("invalid send timestamp")
        self.send_s, self.status_code = send_s, status_code
        self.errors = []
        if type(status_code) is not int or status_code != 200:
            self.errors.append("http_status_not_200")
        if not isinstance(content_type, str) or content_type.split(";", 1)[0].strip().lower() != "text/event-stream":
            self.errors.append("content_type_not_event_stream")
        self.decoder = SSEDecoder()
        self.response_id = self.model = self.finish_reason = self.usage = None
        self.api_timings = None
        self.done_s = self.first_byte_s = self.first_content_s = self.first_any_s = None
        self.last_read_s = send_s
        self.event_times = []
        self.content_bytes = self.reasoning_bytes = 0
        self.content_hash, self.reasoning_hash = hashlib.sha256(), hashlib.sha256()
        self.closed = False

    def _usage(self, usage):
        if self.usage is not None or not isinstance(usage, dict):
            raise ValueError("duplicate_or_invalid_usage")
        result = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if type(value) is not int or value < 0:
                raise ValueError("usage_counts_must_be_nonnegative_integers")
            result[key] = value
        if result["total_tokens"] != result["prompt_tokens"] + result["completion_tokens"]:
            raise ValueError("usage_total_mismatch")
        details = usage.get("prompt_tokens_details")
        result["cached_tokens"] = None
        if details is not None:
            if not isinstance(details, dict):
                raise ValueError("invalid_cached_usage")
            cached = details.get("cached_tokens")
            if type(cached) is not int or not 0 <= cached <= result["prompt_tokens"]:
                raise ValueError("invalid_cached_usage")
            result["cached_tokens"] = cached
        self.usage = result

    def _event(self, text, time_s):
        if self.done_s is not None:
            raise ValueError("data_after_done")
        if text == "[DONE]":
            self.done_s = time_s
            return
        value = strict_json(text)
        if not isinstance(value, dict):
            raise ValueError("SSE_JSON_must_be_an_object")
        if "error" in value:
            self.errors.append("api_error_event")
            return
        identity = value.get("id")
        if identity is not None:
            if not isinstance(identity, str) or not identity or (self.response_id is not None and identity != self.response_id):
                raise ValueError("response_id_changed_or_invalid")
            self.response_id = identity
        model = value.get("model")
        if model is not None:
            if not isinstance(model, str) or not model or (self.model is not None and model != self.model):
                raise ValueError("response_model_changed_or_invalid")
            self.model = model
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("single_choice_stream_required")
        if not choices and "usage" not in value:
            raise ValueError("empty_chunk_without_usage")
        for choice in choices:
            if not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] != 0:
                raise ValueError("single_choice_index_zero_required")
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise ValueError("delta_must_be_an_object")
            if "tool_calls" in delta or "function_call" in delta:
                raise ValueError("tool_execution_not_supported_by_capture")
            content, reasoning = delta.get("content", ""), delta.get("reasoning_content", "")
            if not isinstance(content, str) or not isinstance(reasoning, str):
                raise ValueError("delta_text_must_be_a_string")
            if self.finish_reason is not None and (content or reasoning):
                raise ValueError("content_after_finish")
            if content:
                encoded = content.encode("utf-8")
                self.content_hash.update(encoded)
                self.content_bytes += len(encoded)
                if self.first_content_s is None:
                    self.first_content_s = time_s
            if reasoning:
                encoded = reasoning.encode("utf-8")
                self.reasoning_hash.update(encoded)
                self.reasoning_bytes += len(encoded)
            if (content or reasoning) and self.first_any_s is None:
                self.first_any_s = time_s
            finish = choice.get("finish_reason")
            if finish is not None:
                if self.finish_reason is not None or finish not in ("stop", "length"):
                    raise ValueError("duplicate_or_unsupported_finish_reason")
                self.finish_reason = finish
        if "usage" in value and value["usage"] is not None:
            self._usage(value["usage"])
        if "timings" in value:
            timings = value["timings"]
            if not isinstance(timings, dict) or self.api_timings is not None:
                raise ValueError("duplicate_or_invalid_api_timings")
            # These API-side aggregates are never engine token timestamps.
            selected = {key: timings[key] for key in ("cache_n", "prompt_n", "predicted_n", "prompt_ms", "predicted_ms") if key in timings}
            if any(not finite(item) or item < 0 for item in selected.values() if item is not None):
                raise ValueError("invalid_api_timing_value")
            self.api_timings = selected

    def feed(self, raw, time_s):
        if self.closed:
            raise ValueError("capture already closed")
        if not finite(time_s) or time_s < self.last_read_s:
            raise ValueError("non_monotonic_transport_timestamp")
        self.last_read_s = time_s
        if raw and self.first_byte_s is None:
            self.first_byte_s = time_s
        for event in self.decoder.feed(raw):
            self.event_times.append(time_s)
            self._event(event, time_s)
        if len(self.event_times) > 32768:
            raise ValueError("SSE event count exceeds limit")

    def finish(self, *, transport_error=None):
        if not self.closed:
            self.closed = True
            try:
                for event in self.decoder.feed(b"", final=True):
                    self.event_times.append(self.last_read_s)
                    self._event(event, self.last_read_s)
            except (ValueError, UnicodeError):
                self.errors.append("incomplete_or_invalid_sse_at_end")
        if transport_error:
            self.errors.append(transport_error)
        for key, value in (("done", self.done_s), ("finish_reason", self.finish_reason),
                           ("usage", self.usage), ("response_id", self.response_id)):
            if value is None:
                self.errors.append(f"missing_{key}")
        errors = sorted(set(self.errors))
        return {"status_code": self.status_code, "response_id": self.response_id,
                "response_model": self.model, "send_s": self.send_s,
                "first_response_byte_s": self.first_byte_s, "complete_s": self.done_s,
                "ttft_api_s": None if self.first_content_s is None else self.first_content_s - self.send_s,
                "ttft_any_s": None if self.first_any_s is None else self.first_any_s - self.send_s,
                "sse_event_times_s": self.event_times, "engine_token_intervals": None,
                "content_utf8_bytes": self.content_bytes, "reasoning_utf8_bytes": self.reasoning_bytes,
                "content_sha256": self.content_hash.hexdigest(), "reasoning_sha256": self.reasoning_hash.hexdigest(),
                "usage": self.usage, "api_timings": self.api_timings, "finish_reason": self.finish_reason,
                "api_stream_complete": not errors, "errors": errors,
                "normal_completion": False, "production_qualified": False}
