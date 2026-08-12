from dataclasses import dataclass

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient


@dataclass(frozen=True)
class UpdatableEngines:
    rollout_engines: list[SGLangApiClient]
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
    snapshot_cell_id_to_hashes: dict[str, str]
