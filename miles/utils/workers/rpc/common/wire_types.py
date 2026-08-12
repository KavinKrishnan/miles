from __future__ import annotations

import base64
import pickle
from argparse import Namespace
from typing import Annotated, Any

from pydantic import BeforeValidator, PlainSerializer


def _as_namespace(value: Any) -> Namespace:
    if isinstance(value, Namespace):
        return value
    assert isinstance(value, dict), f"a namespace argument arrives as a mapping, got {type(value).__name__}"
    return Namespace(**value)


def _as_mapping(value: Namespace) -> dict[str, Any]:
    return vars(value)


WireNamespace = Annotated[
    Any,
    BeforeValidator(_as_namespace),
    PlainSerializer(_as_mapping, return_type=dict[str, Any]),
]

PICKLED_HATCH_MARKER = "MILES_PICKLED_HATCH"

PICKLED_TAG = "__miles_pickled__"


def _from_pickled(value: Any) -> Any:
    if not isinstance(value, dict) or (encoded := value.get(PICKLED_TAG)) is None:
        return value
    return pickle.loads(base64.b64decode(encoded))


def _to_pickled(value: Any) -> dict[str, str]:
    return {PICKLED_TAG: base64.b64encode(pickle.dumps(value)).decode()}


# TODO(MILES_PICKLED_HATCH): temporary, whitelisted escape hatch for the argparse Namespace a
# trainer is built from, which no wire type reproduces losslessly. Reclaim it once the
# arguments subsystem is split into wire-typed pieces; every other parameter must stay
# strictly wire-typed.
Pickled = Annotated[
    Any,
    BeforeValidator(_from_pickled),
    PlainSerializer(_to_pickled, return_type=dict[str, str]),
]
