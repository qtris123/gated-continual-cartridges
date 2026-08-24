"""vLLM OpenAI server whose responses survive non-finite logprobs.

vLLM 0.11 returns chat completions as `JSONResponse(generator.model_dump())`,
which Starlette encodes with the stdlib json module. That encoder rejects NaN
and Inf, so a single non-finite logprob turns the whole request into a 500.
FinQA phase-1 synthesis hit this 39k times: the client saw InternalServerError,
and because a failed request discards its entire batch, 121 of 256 batches
(3872 rows) were lost.

Non-finite values are replaced with -1000.0, the sentinel `OpenAIClient`
already writes for absent top-logprob entries, so they carry no probability
mass into `FlatTopLogprobs.reconstruct`. Scrubbing runs only after the strict
encoder has already raised, leaving the common path untouched.

Drop-in replacement for `python -m vllm.entrypoints.openai.api_server`.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi.responses import JSONResponse

import vllm.entrypoints.openai.api_server as api_server
from vllm.logger import init_logger

logger = init_logger("vllm_nan_safe_server")

ABSENT_LOGPROB = -1000.0

_scrubbed_responses = 0
_scrubbed_values = 0


def _scrub(obj: Any) -> Any:
    global _scrubbed_values
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        _scrubbed_values += 1
        return ABSENT_LOGPROB
    if isinstance(obj, dict):
        return {key: _scrub(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(value) for value in obj]
    return obj


class NanSafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        global _scrubbed_responses
        try:
            return super().render(content)
        except ValueError:
            before = _scrubbed_values
            cleaned = _scrub(content)
            _scrubbed_responses += 1
            logger.warning(
                "NAN-SCRUB: replaced %d non-finite float(s) with %.1f "
                "(%d response(s) scrubbed so far)",
                _scrubbed_values - before,
                ABSENT_LOGPROB,
                _scrubbed_responses,
            )
            return super().render(cleaned)


def main() -> None:
    # create_chat_completion resolves JSONResponse from module globals at call
    # time, so rebinding it here covers every request handler in the server.
    api_server.JSONResponse = NanSafeJSONResponse

    api_server.cli_env_setup()
    parser = api_server.FlexibleArgumentParser(
        description="vLLM OpenAI-compatible server with NaN-safe JSON encoding."
    )
    parser = api_server.make_arg_parser(parser)
    args = parser.parse_args()
    api_server.validate_parsed_serve_args(args)
    api_server.uvloop.run(api_server.run_server(args))


if __name__ == "__main__":
    main()
