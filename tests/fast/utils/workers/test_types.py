import pytest

from miles.utils.workers.types import DeployComponent, DeploySelector, HotRestartComponent, parse_hot_restart


class TestDeploySelectorParse:
    def test_the_whole_run_names_no_component_of_it(self) -> None:
        """`all` is the default every existing run parses, and it selects over components rather than naming one."""
        assert DeploySelector.parse("all") == DeploySelector(component=DeployComponent.ALL)

    def test_a_component_without_an_instance_covers_every_instance_of_it(self) -> None:
        """A run with one trainer names it by component alone, exactly as before instances existed."""
        assert DeploySelector.parse("trainer") == DeploySelector(component=DeployComponent.TRAINER, instance=None)

    def test_a_trainer_instance_is_the_model_id_it_trains(self) -> None:
        """A multi policy run installs one trainer deployment per policy, told apart by this instance."""
        assert DeploySelector.parse("trainer:policy_a") == DeploySelector(
            component=DeployComponent.TRAINER, instance="policy_a"
        )

    def test_an_inference_instance_is_named_after_its_deployment(self) -> None:
        """Several inference deployments serve one run, and each is deployed on its own."""
        assert DeploySelector.parse("inference:instance_b") == DeploySelector(
            component=DeployComponent.INFERENCE, instance="instance_b"
        )

    @pytest.mark.parametrize("value", ["primary:x", "all:x"])
    def test_rejects_an_instance_of_a_component_that_comes_in_exactly_one(self, value: str) -> None:
        """A run deploys one orchestration script, so a second instance of it would be an undefined deployment."""
        with pytest.raises(AssertionError, match="names an instance"):
            DeploySelector.parse(value)

    def test_rejects_a_separator_that_names_no_instance(self) -> None:
        """A trailing separator reads as an empty instance name, which would silently deploy every instance."""
        with pytest.raises(AssertionError, match="without naming an instance"):
            DeploySelector.parse("trainer:")

    def test_rejects_a_component_that_is_not_one_of_the_four(self) -> None:
        """The four components partition the run, so a fifth name would deploy an undefined subset."""
        with pytest.raises(AssertionError, match=r"\['all', 'primary', 'trainer', 'inference'\]"):
            DeploySelector.parse("router")


class TestDeploySelectorValue:
    @pytest.mark.parametrize("value", ["all", "trainer", "trainer:policy_a", "inference:instance_b"])
    def test_the_value_parses_back_into_the_same_selector(self, value: str) -> None:
        """The value is passed to the pods as --deploy-component, so it has to survive the round trip."""
        assert DeploySelector.parse(DeploySelector.parse(value).value).value == value


class TestDeploySelectorSelects:
    def test_the_whole_run_selects_every_component(self) -> None:
        """An unsplit run deploys each component itself, so every one of them is selected."""
        selector = DeploySelector.parse("all")

        assert selector.selects(DeployComponent.TRAINER)
        assert selector.selects(DeployComponent.INFERENCE)
        assert selector.selects(DeployComponent.PRIMARY)

    def test_a_component_selects_no_other_component(self) -> None:
        """Deploying what another release already deploys would install the same workers twice."""
        assert not DeploySelector.parse("trainer").selects(DeployComponent.INFERENCE)

    def test_a_named_instance_selects_only_itself(self) -> None:
        """Each policy's trainer is its own deployment, so one must not carry another policy's ranks."""
        selector = DeploySelector.parse("trainer:policy_a")

        assert selector.selects(DeployComponent.TRAINER, instance="policy_a")
        assert not selector.selects(DeployComponent.TRAINER, instance="policy_b")

    def test_a_selector_without_an_instance_selects_every_instance(self) -> None:
        """`trainer` deploys the trainers of every policy, so no instance of it is left out."""
        selector = DeploySelector.parse("trainer")

        assert selector.selects(DeployComponent.TRAINER, instance="policy_a")
        assert selector.selects(DeployComponent.TRAINER, instance="policy_b")

    def test_an_unnamed_instance_is_selected_by_a_named_selector(self) -> None:
        """Callers that ask about a component as a whole must not be answered no by an instance deployment."""
        assert DeploySelector.parse("trainer:policy_a").selects(DeployComponent.TRAINER)

    def test_refuses_to_be_asked_whether_it_selects_all(self) -> None:
        """`all` is a selector over components, so asking whether it is deployed is a question about nothing."""
        with pytest.raises(AssertionError, match="never a component itself"):
            DeploySelector.parse("all").selects(DeployComponent.ALL)


class TestDeploySelectorDeploysOrchestrationScript:
    @pytest.mark.parametrize("value", ["all", "primary"])
    def test_the_deployments_that_carry_the_script(self, value: str) -> None:
        """Exactly one deployment of a run runs the training loop, and everything else is called by it."""
        assert DeploySelector.parse(value).deploys_orchestration_script()

    @pytest.mark.parametrize("value", ["trainer", "trainer:policy_a", "inference", "inference:instance_b"])
    def test_a_worker_deployment_runs_no_script(self, value: str) -> None:
        """A second copy of the training loop would drive the same run twice."""
        assert not DeploySelector.parse(value).deploys_orchestration_script()


class TestDeploySelectorIsSplit:
    def test_the_whole_run_is_not_split(self) -> None:
        """One release holds every worker, so nothing has to be addressed across deployments."""
        assert not DeploySelector.parse("all").is_split()

    @pytest.mark.parametrize("value", ["primary", "trainer", "trainer:policy_a", "inference:instance_b"])
    def test_any_named_component_is_one_deployment_of_several(self, value: str) -> None:
        """The rest of the run lives in other releases, which is what makes static addresses necessary."""
        assert DeploySelector.parse(value).is_split()


class TestDeployComponentTakesInstance:
    @pytest.mark.parametrize("component", [DeployComponent.TRAINER, DeployComponent.INFERENCE])
    def test_the_components_a_run_deploys_several_of(self, component: DeployComponent) -> None:
        """A run trains several policies and serves several inference deployments, one instance each."""
        assert component.takes_instance()

    @pytest.mark.parametrize("component", [DeployComponent.ALL, DeployComponent.PRIMARY])
    def test_the_components_a_run_deploys_exactly_one_of(self, component: DeployComponent) -> None:
        """Naming an instance of them would be a second copy of the run itself."""
        assert not component.takes_instance()


class TestParseHotRestart:
    def test_an_empty_value_asks_for_no_restart(self):
        """Every ordinary launch goes through this parser, so the empty default has to mean nothing happens."""
        assert parse_hot_restart("") == frozenset()

    def test_both_supported_components_parse(self):
        """This is the value the feature is documented with."""
        assert parse_hot_restart("orchestration,rollout_executor") == frozenset(HotRestartComponent)

    def test_whitespace_and_empty_entries_are_tolerated(self):
        """Users copy the value out of docs and shells, so a stray space must not be an error."""
        assert parse_hot_restart(" orchestration , rollout_executor ,") == frozenset(HotRestartComponent)

    @pytest.mark.parametrize("value", ["orchestration", "rollout_executor"])
    def test_one_component_on_its_own_is_refused(self, value: str):
        """A new script cannot drive the executor its predecessor initialized, and the reverse kills a live run."""
        with pytest.raises(AssertionError, match="replaced together or not at all"):
            parse_hot_restart(value)

    def test_an_unsupported_component_is_refused_by_name(self):
        """Silently ignoring `trainer` would promise a restart that never happens."""
        with pytest.raises(AssertionError, match="'trainer'"):
            parse_hot_restart("orchestration,trainer")
