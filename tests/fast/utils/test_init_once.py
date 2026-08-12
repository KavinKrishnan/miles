import pytest

from miles.utils.init_once import InitOnce


class TestInitOnce:
    def test_a_fresh_guard_reports_it_is_not_initialized(self):
        """Nothing has run yet, so a restarted script must be told to initialize rather than resume."""
        assert InitOnce(component="Widget").is_initialized is False

    def test_entering_once_marks_the_component_initialized(self):
        """The one legitimate init has to leave a state a later script can query."""
        guard = InitOnce(component="Widget")

        guard.enter()

        assert guard.is_initialized is True

    def test_entering_twice_raises_instead_of_reinitializing_a_live_system(self):
        """A mistaken second init would silently rebuild a component someone else is already driving."""
        guard = InitOnce(component="Widget")
        guard.enter()

        with pytest.raises(AssertionError, match="already been initialized"):
            guard.enter()

    def test_the_message_names_the_component_that_was_initialized_twice(self):
        """A run has many components, so the failure has to say which one was initialized twice."""
        guard = InitOnce(component="TrainerController(actor)")
        guard.enter()

        with pytest.raises(AssertionError, match=r"TrainerController\(actor\)"):
            guard.enter()

    def test_asserting_initialized_fails_before_the_first_init(self):
        """load_state on an uninitialized component would touch attributes init never built."""
        with pytest.raises(AssertionError, match="not initialized yet"):
            InitOnce(component="Widget").assert_initialized()

    def test_asserting_initialized_passes_after_the_first_init(self):
        """Resuming an initialized component is exactly the supported path."""
        guard = InitOnce(component="Widget")
        guard.enter()

        guard.assert_initialized()

    def test_two_guards_of_one_process_are_independent(self):
        """One process hosts several components, and initializing one says nothing about the others."""
        first = InitOnce(component="A")
        second = InitOnce(component="B")

        first.enter()

        assert second.is_initialized is False

    def test_two_guards_of_the_same_component_name_are_independent(self):
        """A healed cell is a new process with a new guard, so its init must not be refused by its predecessor's."""
        replaced = InitOnce(component="TrainRayActor(role=actor, rank=0)")
        replaced.enter()

        replacement = InitOnce(component="TrainRayActor(role=actor, rank=0)")
        replacement.enter()

        assert replacement.is_initialized is True
