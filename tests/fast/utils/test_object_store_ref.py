from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.object_store import RAY_OBJECT_REF_TAG, RayObjectStore, StoreObjectRef

FREED_OBJECT_TIMEOUT_SECONDS = 60.0


class TestStoreObjectRef:
    def test_a_mooncake_reference_survives_a_json_round_trip(self):
        """The trainer ships it to the driver over rpc, which encodes the model as json."""
        ref = StoreObjectRef(payload={"key": "miles-object-store/7", "size": 12})

        restored = StoreObjectRef.model_validate_json(ref.model_dump_json())

        assert restored == ref
        assert restored.payload == {"key": "miles-object-store/7", "size": 12}

    def test_a_reference_cannot_be_repointed_after_it_is_handed_over(self):
        """A consumer holding it must read the object the producer put, not one somebody swapped in."""
        ref = StoreObjectRef(payload="k")

        with pytest.raises(ValidationError):
            ref.payload = "other"


class TestARayReferenceOnTheWire:
    def test_it_survives_a_json_round_trip(self, ray_local_mode):
        """comm-backend=rpc still allows the ray object store, so its reference has to cross the wire."""
        ref = RayObjectStore().put({"tokens": [1, 2, 3]})

        restored = StoreObjectRef.model_validate_json(ref.model_dump_json())

        assert RayObjectStore().get(restored).value == {"tokens": [1, 2, 3]}

    def test_the_encoded_form_is_a_tagged_string(self, ray_local_mode):
        """An ObjectRef is not json, so it travels as the cloudpickle bytes ray documents as the last resort."""
        ref = RayObjectStore().put("payload")

        encoded = ref.model_dump(mode="json")["payload"]

        assert isinstance(encoded[RAY_OBJECT_REF_TAG], str)

    def test_a_reference_that_never_left_this_process_is_unchanged(self, ray_local_mode):
        """Ray communication passes the model as is, and re-encoding it would cost a copy per call."""
        ref = RayObjectStore().put("payload")

        assert RayObjectStore().get(ref).value == "payload"

    def test_removing_a_reference_that_arrived_over_the_wire_releases_it_by_hand(self, monkeypatch, ray_local_mode):
        """Cloudpickling a reference pins the object, so reference counting can no longer free it."""
        import ray

        freed: list[list] = []
        monkeypatch.setattr(ray._private.internal_api, "free", lambda refs: freed.append(list(refs)))
        restored = StoreObjectRef.model_validate_json(RayObjectStore().put("payload").model_dump_json())

        RayObjectStore().remove(restored)

        assert freed == [[restored.payload]]

    def test_removing_a_reference_that_never_left_this_process_frees_nothing(self, monkeypatch, ray_local_mode):
        """Under comm-backend=ray the reference is still counted, and a free destroys rather than decrements."""
        import ray

        freed: list[list] = []
        monkeypatch.setattr(ray._private.internal_api, "free", lambda refs: freed.append(list(refs)))
        ref = RayObjectStore().put("payload")

        RayObjectStore().remove(ref)

        assert freed == []
        assert RayObjectStore().get(ref).value == "payload"

    def test_the_same_reference_is_freed_only_once(self, monkeypatch, ray_local_mode):
        """A rollout is removed shard by shard, and freeing an id twice would reach whatever reused it."""
        import ray

        freed: list[list] = []
        monkeypatch.setattr(ray._private.internal_api, "free", lambda refs: freed.append(list(refs)))
        restored = StoreObjectRef.model_validate_json(RayObjectStore().put("payload").model_dump_json())

        RayObjectStore().remove(restored)
        RayObjectStore().remove(restored)

        assert len(freed) == 1


class TestFreeingARayObjectForReal:
    def test_the_object_is_gone_after_a_wire_borne_reference_is_removed(self, ray_local_mode):
        """The point of the explicit free is that the pinned object really stops occupying the store."""
        import ray

        restored = StoreObjectRef.model_validate_json(RayObjectStore().put({"tokens": [1, 2, 3]}).model_dump_json())

        RayObjectStore().remove(restored)

        with pytest.raises(ray.exceptions.ObjectLostError):
            ray.get(restored.payload, timeout=FREED_OBJECT_TIMEOUT_SECONDS)

    def test_an_object_nobody_removed_is_still_readable(self, ray_local_mode):
        """A free reaching further than the reference it was handed would take the run's data with it."""
        kept = StoreObjectRef.model_validate_json(RayObjectStore().put({"tokens": [4]}).model_dump_json())
        removed = StoreObjectRef.model_validate_json(RayObjectStore().put({"tokens": [5]}).model_dump_json())

        RayObjectStore().remove(removed)

        assert RayObjectStore().get(kept).value == {"tokens": [4]}
