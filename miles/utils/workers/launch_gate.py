import httpx

from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.retry_utils import retry_until_deadline

GATE_PORT_NAME = "gate"
GATE_TIMEOUT_META_KEY = "launch_gate_timeout_seconds"
LAUNCH_GATE_TIMEOUT_SECONDS = 1800.0

_INITIAL_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 5.0
_ATTEMPT_TIMEOUT_SECONDS = 30.0


async def activate_launch_gate(gate_url: str, timeout: float = LAUNCH_GATE_TIMEOUT_SECONDS) -> None:
    async def _activate(remaining_seconds: float) -> None:
        response = await GeneralHttpClientProvider.client().post(
            f"{gate_url}/gate/activate",
            json={},
            timeout=min(_ATTEMPT_TIMEOUT_SECONDS, remaining_seconds),
        )
        response.raise_for_status()

    await retry_until_deadline(
        _activate,
        total_seconds=timeout,
        retry_on=(httpx.HTTPError, OSError),
        initial_delay=_INITIAL_DELAY_SECONDS,
        max_delay=_MAX_DELAY_SECONDS,
        log_fields=dict(gate_url=gate_url),
    )
