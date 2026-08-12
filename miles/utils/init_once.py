from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class InitOnce:
    def __init__(self, *, component: str) -> None:
        self._component = component
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def enter(self) -> None:
        assert not self._initialized, (
            f"{self._component} has already been initialized in this process, and initializing it a second time "
            f"would re-initialize a live system behind the back of whoever is already driving it; a restarted "
            f"orchestration script has to resume an initialized component instead of initializing it again"
        )
        self._initialized = True
        logger.info(f"{self._component} is now initialized")

    def assert_initialized(self) -> None:
        assert self._initialized, f"{self._component} is not initialized yet"
