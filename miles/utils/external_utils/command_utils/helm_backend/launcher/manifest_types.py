from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import STATE_FILE_FLAG
from miles.utils.pydantic_utils import FrozenOpenBaseModel, FrozenStrictBaseModel

logger = logging.getLogger(__name__)

RESTART_AT_ANNOTATION = "miles.radixark.io/restart-at"


class EnvEntry(FrozenOpenBaseModel):
    name: str
    value: str = ""


class Container(FrozenOpenBaseModel):
    name: str
    command: list[str] = []
    env: list[EnvEntry] = []


class PodSpec(FrozenOpenBaseModel):
    containers: list[Container] = []


class PodTemplateMetadata(FrozenOpenBaseModel):
    annotations: dict[str, str] = {}


class PodTemplate(FrozenOpenBaseModel):
    metadata: PodTemplateMetadata | None = None
    spec: PodSpec | None = None


class ObjectSpec(FrozenOpenBaseModel):
    replicas: int | None = None
    template: PodTemplate | None = None


class ObjectMetadata(FrozenOpenBaseModel):
    name: str


class ManifestObject(FrozenOpenBaseModel):
    kind: str
    metadata: ObjectMetadata
    spec: ObjectSpec | None = None

    @property
    def key(self) -> str:
        return f"{self.kind}/{self.metadata.name}"

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def replicas(self) -> int | None:
        return self.spec.replicas if self.spec is not None else None

    @property
    def restart_at(self) -> str | None:
        if self.spec is None or self.spec.template is None or self.spec.template.metadata is None:
            return None
        return self.spec.template.metadata.annotations.get(RESTART_AT_ANNOTATION)

    @property
    def body(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)

    def containers_named(self, container: str) -> list[Container]:
        if self.kind != "StatefulSet" or self.spec is None or self.spec.template is None:
            return []
        pod = self.spec.template.spec
        return [described for described in (pod.containers if pod is not None else []) if described.name == container]


class Manifest(FrozenStrictBaseModel):
    objects: list[ManifestObject]

    @classmethod
    def parse(cls, rendered: str) -> Manifest:
        return cls(objects=[document for document in yaml.safe_load_all(rendered) if document])

    @property
    def by_key(self) -> dict[str, ManifestObject]:
        return {described.key: described for described in self.objects}

    def restart_at(self, *, preferred_object_name: str) -> str | None:
        """The stamp a previous hot restart wrote, which an ordinary relaunch has to render unchanged."""
        stamps = {described.key: stamp for described in self.objects if (stamp := described.restart_at) is not None}
        if len(set(stamps.values())) > 1:
            logger.warning(
                f"The installed manifest carries {sorted(set(stamps.values()))} as its restart stamps, but one hot "
                f"restart stamps one value onto every object it replaces, so an upgrade of this run was interrupted "
                f"or one of its objects was patched by hand; carrying the stamp of {preferred_object_name} forward, "
                f"which rolls whichever object holds another one"
            )
        for described in self.objects:
            if described.name == preferred_object_name and (preferred := described.restart_at) is not None:
                return preferred
        return next(iter(stamps.values()), None)

    def state_file(self, *, container: str) -> Path | None:
        for described in self.objects:
            for found in described.containers_named(container):
                if STATE_FILE_FLAG in found.command:
                    return Path(found.command[found.command.index(STATE_FILE_FLAG) + 1])
        return None
