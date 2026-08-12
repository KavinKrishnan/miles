import functools
from argparse import Namespace

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as single_turn_generate
from miles.utils.megatron_config import resolve_megatron_config


@functools.cache
def _compute_model_ids(megatron_config: str | None) -> tuple[str, ...]:
    return tuple(resolve_megatron_config(Namespace(megatron_config=megatron_config)).model_ids)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    model_ids = _compute_model_ids(input.args.megatron_config)
    input.sample.trainer_model_id = model_ids[(input.sample.group_index or 0) % len(model_ids)]
    return await single_turn_generate(input)
