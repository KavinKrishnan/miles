from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio

import pytest

from miles.ray.train.multi_policy import (
    MultiPolicyCheckpointState,
    MultiPolicySaveCoordinator,
    assert_restored_rollout_ids,
    load_multi_policy_state,
    multi_policy_state_path,
    save_multi_policy_state,
)


def _make_coordinator(*model_ids: str) -> MultiPolicySaveCoordinator:
    return MultiPolicySaveCoordinator(model_ids=list(model_ids), primary_model_id=model_ids[0])


async def _noop(force_sync: bool) -> None:
    return None


class TestMultiPolicyCheckpointState:
    def test_the_global_state_is_indexed_by_the_primary_rollout_id(self, tmp_path):
        """DataSource and RolloutExecutor are global, so one index must identify the whole run."""
        state = MultiPolicyCheckpointState(primary_model_id="a", rollout_ids={"a": 7, "b": 5})

        save_multi_policy_state(tmp_path, state)

        assert multi_policy_state_path(tmp_path, 7).exists()
        assert load_multi_policy_state(tmp_path, 7) == state

    def test_a_missing_state_file_reads_as_none(self, tmp_path):
        """A run that never saved must start fresh instead of exploding on load."""
        assert load_multi_policy_state(tmp_path, 3) is None

    def test_matching_rollout_ids_pass_the_load_assert(self):
        """The happy path must not warn or fail when every policy restored where the primary says."""
        state = MultiPolicyCheckpointState(primary_model_id="a", rollout_ids={"a": 7, "b": 5})

        assert_restored_rollout_ids(state, {"a": 7, "b": 5})

    def test_a_policy_restored_at_the_wrong_rollout_id_fails_loudly(self):
        """Restoring inconsistent positions trains each policy against the wrong global state."""
        state = MultiPolicyCheckpointState(primary_model_id="a", rollout_ids={"a": 7, "b": 5})

        with pytest.raises(AssertionError, match="multi policy checkpoint mismatch"):
            assert_restored_rollout_ids(state, {"a": 7, "b": 6})

    def test_a_missing_policy_fails_loudly(self):
        """Dropping a policy on resume would leave its checkpoint silently unused."""
        state = MultiPolicyCheckpointState(primary_model_id="a", rollout_ids={"a": 7, "b": 5})

        with pytest.raises(AssertionError, match="multi policy checkpoint mismatch"):
            assert_restored_rollout_ids(state, {"a": 7})

    def test_a_policy_the_checkpoint_never_recorded_is_not_compared(self):
        """A policy that had already finished has no position in the record, so any restore is legal."""
        state = MultiPolicyCheckpointState(primary_model_id="a", rollout_ids={"a": 7}, finished_model_ids=["b"])

        assert_restored_rollout_ids(state, {"a": 7, "b": 12})


class TestMultiPolicySaveCoordinator:
    async def test_the_primary_waits_for_the_others_to_reach_their_round_boundary(self):
        """The naive scheme: a global checkpoint is only consistent once every policy parked."""
        coordinator = _make_coordinator("a", "b")

        begin = asyncio.create_task(coordinator.begin_save(4))
        await asyncio.sleep(0.01)
        assert not begin.done()

        parked = asyncio.create_task(coordinator.maybe_park("b", 2, _noop))
        await begin

        assert coordinator.rollout_ids == {"a": 4, "b": 2}
        assert not parked.done()

        await coordinator.end_save()
        assert await parked is True

    async def test_a_policy_saves_its_own_model_before_parking(self):
        """Each policy checkpoint must be written at the boundary the primary recorded."""
        coordinator = _make_coordinator("a", "b")
        saved: list[str] = []

        async def _save(force_sync: bool) -> None:
            saved.append("b")

        begin = asyncio.create_task(coordinator.begin_save(4))
        parked = asyncio.create_task(coordinator.maybe_park("b", 2, _save))
        await begin

        assert saved == ["b"]

        await coordinator.end_save()
        await parked

    async def test_the_primary_decides_whether_the_others_flush_their_checkpoint(self):
        """A last checkpoint that is still buffered when the process exits is a lost checkpoint."""
        coordinator = _make_coordinator("a", "b")
        seen: list[bool] = []

        async def _save(force_sync: bool) -> None:
            seen.append(force_sync)

        begin = asyncio.create_task(coordinator.begin_save(4, force_sync=True))
        parked = asyncio.create_task(coordinator.maybe_park("b", 2, _save))
        await begin

        assert seen == [True]

        await coordinator.end_save()
        await parked

    async def test_a_policy_at_a_boundary_without_a_pending_save_runs_on(self):
        """Save points are the primary's call; nobody else may stop the run."""
        coordinator = _make_coordinator("a", "b")

        assert await coordinator.maybe_park("b", 2, _noop) is False
        assert coordinator.rollout_ids == {}

    async def test_a_finished_policy_does_not_block_the_next_checkpoint(self):
        """A policy that exhausted its rounds never parks again; waiting for it would hang the run."""
        coordinator = _make_coordinator("a", "b")

        await coordinator.leave("b")
        await asyncio.wait_for(coordinator.begin_save(9), timeout=1)

        assert coordinator.rollout_ids == {"a": 9}
        assert coordinator.finished_model_ids == ["b"]
        await coordinator.end_save()

    async def test_a_later_checkpoint_never_records_a_policy_at_its_previous_position(self):
        """A stale rollout_id in the sidecar reads as a plausible but wrong resume point."""
        coordinator = _make_coordinator("a", "b")

        begin = asyncio.create_task(coordinator.begin_save(4))
        parked = asyncio.create_task(coordinator.maybe_park("b", 2, _noop))
        await begin
        await coordinator.end_save()
        await parked

        await coordinator.leave("b")
        await asyncio.wait_for(coordinator.begin_save(9), timeout=1)

        assert coordinator.rollout_ids == {"a": 9}
        await coordinator.end_save()

    async def test_a_policy_that_never_parks_fails_the_save_instead_of_hanging(self):
        """Without a deadline the primary waits forever holding a queue nobody drains."""
        coordinator = _make_coordinator("a", "b")

        with pytest.raises(TimeoutError, match="still running"):
            await coordinator.begin_save(4, timeout=0.01)

    async def test_a_failed_save_releases_the_policies_it_stopped(self):
        """A save left flagged in flight parks every other policy forever."""
        coordinator = _make_coordinator("a", "b")

        async def _park_later() -> bool:
            await asyncio.sleep(0.01)
            return await coordinator.maybe_park("b", 2, _noop)

        parked = asyncio.create_task(_park_later())

        with pytest.raises(RuntimeError):
            async with coordinator.saving(4, force_sync=False):
                raise RuntimeError("save failed")

        assert await asyncio.wait_for(parked, timeout=1) is True

    async def test_back_to_back_saves_each_record_a_fresh_position_for_every_policy(self):
        """A record carrying a policy's position from the previous checkpoint reads as a plausible resume point."""
        coordinator = _make_coordinator("a", "b")
        recorded: list[dict[str, int]] = []

        async def _park_rounds() -> None:
            for rollout_id in range(3):
                assert await coordinator.maybe_park("b", rollout_id, _noop) is True

        parking = asyncio.create_task(_park_rounds())
        for rollout_id in range(3):
            await asyncio.wait_for(coordinator.begin_save(rollout_id), timeout=1)
            recorded.append(coordinator.rollout_ids)
            await coordinator.end_save()

        await asyncio.wait_for(parking, timeout=1)
        assert recorded == [{"a": 0, "b": 0}, {"a": 1, "b": 1}, {"a": 2, "b": 2}]

    async def test_a_save_waits_for_the_previous_one_to_release_its_parked_policies(self):
        """Starting while a policy is still parked would record its old rollout id as if it had just parked."""
        coordinator = _make_coordinator("a", "b")

        parked = asyncio.create_task(coordinator.maybe_park("b", 2, _noop))
        await coordinator.begin_save(4)
        await coordinator.end_save()

        begin = asyncio.create_task(coordinator.begin_save(5))
        await asyncio.sleep(0)
        assert not begin.done()

        assert await asyncio.wait_for(parked, timeout=1) is True
        parked_again = asyncio.create_task(coordinator.maybe_park("b", 3, _noop))
        await asyncio.wait_for(begin, timeout=1)

        assert coordinator.rollout_ids == {"a": 5, "b": 3}
        await coordinator.end_save()
        await parked_again

    async def test_a_finishing_policy_saves_its_last_round_without_the_primary(self):
        """The primary stops driving saves once it exits, so the tail of a slower policy has to save itself."""
        coordinator = _make_coordinator("a", "b")

        await coordinator.leave("a")
        async with coordinator.final_saving("b", 9):
            pass

        assert coordinator.rollout_ids == {"b": 9}

    async def test_a_final_save_and_a_global_save_never_overlap(self):
        """Two writers of the same record would leave one policy's position out of the checkpoint set."""
        coordinator = _make_coordinator("a", "b")
        order: list[str] = []

        async def _final() -> None:
            async with coordinator.final_saving("b", 9):
                order.append("final-start")
                await asyncio.sleep(0.02)
                order.append("final-end")

        final = asyncio.create_task(_final())
        await asyncio.sleep(0)
        await coordinator.leave("b")
        await asyncio.wait_for(coordinator.begin_save(4), timeout=1)
        order.append("global")
        await coordinator.end_save()
        await final

        assert order == ["final-start", "final-end", "global"]

    async def test_a_single_policy_run_never_waits(self):
        """The stop-and-wait bubble must not exist when there is nobody to wait for."""
        coordinator = _make_coordinator("only")

        await asyncio.wait_for(coordinator.begin_save(1), timeout=1)
        await coordinator.end_save()

        assert coordinator.rollout_ids == {"only": 1}
