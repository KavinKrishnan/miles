from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args as _make_args

import miles.ray.rollout.eval_fleet as eval_fleet_mod
from miles.ray.rollout.eval_fleet import EvalFleet, EvalFleetInfo, EvalFleetPin, EvalFleetSession
from miles.rollout.checkpoint_eval import EvalSkip
from miles.utils.workers.worker_spec import HostAndPort


def make_args(**overrides):
    defaults = dict(
        eval_num_gpus=1,
        eval_num_gpus_per_engine=1,
        use_fault_tolerance=False,
        sglang_model_routers={"default": ("10.0.0.1", 30000), "eval": ("10.0.0.2", 31000)},
    )
    defaults.update(overrides)
    return _make_args(**defaults)


class FakeRemoteMethod:
    def __init__(self, engine, name):
        self.engine = engine
        self.name = name

    def remote(self, *args, **kwargs):
        self.engine.log.append((self.name, args, kwargs))
        result = self.engine.responses[self.name](*args, **kwargs)
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(result)
        return fut


class FakeEngine:
    def __init__(self, log):
        self.log = log
        self.weight_version = None

        def load(model_path, weight_version=None):
            self.weight_version = weight_version
            return None

        self.responses = {
            "update_weights_from_disk": load,
            "get_weight_version": lambda: self.weight_version,
        }

    def __getattr__(self, name):
        if name in ("update_weights_from_disk", "get_weight_version"):
            return FakeRemoteMethod(self, name)
        raise AttributeError(name)


class FakeServerEngineWrapper:
    def __init__(self, actor):
        self._actor = actor
        self.is_allocated = True
        self.stopped = False

    @property
    def actor_handle(self):
        return self._actor

    def mark_stopped(self):
        self.stopped = True
        self.is_allocated = False


class FakeEvalServer:
    async def probe_and_mark_dead(self):
        self.probe_calls += 1

    def __init__(self, engines):
        self._engines = engines
        self.wrappers = [FakeServerEngineWrapper(e) for e in engines]
        self.recover_calls = 0
        self.probe_calls = 0
        self.router_ip = "10.0.0.2"
        self.router_port = 31000

    @property
    def server_groups(self):
        return [SimpleNamespace(all_engines=self.wrappers)]

    @property
    def engines(self):
        return [SimpleNamespace(actor_handle=e) for e in self._engines]

    async def recover(self):
        self.recover_calls += 1

    async def wait_all_engines_alive(self):
        pass


@pytest.fixture
def router_always_ready(monkeypatch):
    async def noop_router_ready(self, timeout=180.0):
        return None

    monkeypatch.setattr(eval_fleet_mod.EvalFleet, "_wait_router_ready", noop_router_ready)


def make_fleet(args, engines):
    return EvalFleet(args, srv=FakeEvalServer(engines))


class TestEvalFleetInfo:
    def test_describes_the_fleet_its_router_serves(self):
        """The description the executor retargets its eval args to comes from the server, not its own args."""
        fleet = make_fleet(make_args(eval_num_gpus=4, eval_num_gpus_per_engine=2), [])

        assert fleet.info == EvalFleetInfo(
            router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=4, num_gpus_per_engine=2
        )


class TestEvalFleetPinning:
    async def test_pins_every_engine_before_reporting_success(self, router_always_ready):
        """Every engine is reloaded from the snapshot before the pin reports no skip."""
        log = []
        fleet = make_fleet(make_args(), [FakeEngine(log), FakeEngine(log)])

        pin = await fleet.pin("/snap/step_5", "5")

        load_events = [e for e in log if e[0] == "update_weights_from_disk"]
        assert len(load_events) == 2
        assert all(e[2]["weight_version"] == "5" for e in load_events)
        assert pin == EvalFleetPin(skip_reason=None)

    async def test_requires_all_engines_to_match_and_retries(self, router_always_ready):
        """The router load-balances across engines, so one stale engine = mixed
        versions: the pin must fail even when the other engine matches, retry once,
        then degrade to an attributable skip."""
        log = []
        good, stale = FakeEngine(log), FakeEngine(log)
        stale.responses["get_weight_version"] = lambda: "999"
        fleet = make_fleet(make_args(), [good, stale])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin.skip_reason == "pin_violation"
        assert len([e for e in log if e[0] == "update_weights_from_disk"]) == 4  # 2 engines x 2 attempts

    async def test_recovers_before_pinning(self, router_always_ready):
        """A revived engine must be up before the load: pin runs the health sequence first."""
        fleet = make_fleet(make_args(), [FakeEngine([])])

        await fleet.pin("/snap/step_5", "5")

        assert (fleet._srv.probe_calls, fleet._srv.recover_calls) == (1, 1)

    async def test_leaves_probing_to_the_health_monitor(self, router_always_ready):
        """With --use-fault-tolerance a RolloutHealthMonitor already probes these engines."""
        fleet = make_fleet(make_args(use_fault_tolerance=True), [FakeEngine([])])

        await fleet.pin("/snap/step_5", "5")

        assert fleet._srv.probe_calls == 0
        assert fleet._srv.recover_calls == 1

    async def test_skips_when_the_fleet_stays_unhealthy(self, router_always_ready):
        """An unhealthy fleet reports an attributable skip instead of raising."""
        fleet = make_fleet(make_args(), [FakeEngine([])])

        async def never_alive():
            raise TimeoutError("engines never came up")

        fleet._srv.wait_all_engines_alive = never_alive

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin.skip_reason == "unhealthy"


class FakeInferenceController:
    def __init__(self, pins: list[EvalFleetPin]):
        self.calls: list[dict] = []
        self._pins = pins

    async def pin_eval_fleet(self, *, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        self.calls.append(dict(checkpoint_dir=checkpoint_dir, weight_version=weight_version))
        return self._pins[len(self.calls) - 1]


@pytest.fixture
def fleet_states(monkeypatch):
    built = []
    monkeypatch.setattr(eval_fleet_mod, "GenerateState", lambda args: built.append(args) or f"fake-state-{len(built)}")
    return built


def make_session(controller, *, info=None):
    return EvalFleetSession(
        make_args(),
        info=info or EvalFleetInfo(router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=2, num_gpus_per_engine=1),
        inference_controller=controller,
    )


class TestEvalFleetSession:
    def test_builds_its_state_against_the_fleet_router(self, fleet_states):
        """The executor generates against the eval router and the fleet's gpu sizing, not the rollout ones."""
        make_session(FakeInferenceController([]))

        (state_args,) = fleet_states
        assert (state_args.sglang_router_ip, state_args.sglang_router_port) == ("10.0.0.2", 31000)
        assert (state_args.rollout_num_gpus, state_args.rollout_num_gpus_per_engine) == (2, 1)

    async def test_pins_over_rpc_and_returns_the_cached_state(self, fleet_states):
        """Pinning is the controller's call; the state is built once and handed back per point."""
        controller = FakeInferenceController([EvalFleetPin(skip_reason=None), EvalFleetPin(skip_reason=None)])
        session = make_session(controller)

        first = await session.pin("/snap/step_5", "5")
        second = await session.pin("/snap/step_6", "6")

        assert controller.calls == [
            dict(checkpoint_dir="/snap/step_5", weight_version="5"),
            dict(checkpoint_dir="/snap/step_6", weight_version="6"),
        ]
        assert first == second == "fake-state-1"
        assert len(fleet_states) == 1

    async def test_a_remote_skip_stays_an_attributable_skip(self, fleet_states):
        """The reason the controller skipped for must survive the wire as EvalSkip."""
        session = make_session(FakeInferenceController([EvalFleetPin(skip_reason="pin_violation")]))

        with pytest.raises(EvalSkip) as exc:
            await session.pin("/snap/step_5", "5")

        assert exc.value.reason == "pin_violation"
