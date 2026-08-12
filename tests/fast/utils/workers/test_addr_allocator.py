from __future__ import annotations

from tests.fast.ray.rollout.conftest import fake_engine

from miles.utils.workers.addr_allocator import PortAllocator


class TestPortAllocator:
    def test_a_fresh_allocator_has_no_cursors(self):
        """A brand new allocator starts with no per-node cursors."""
        c = PortAllocator()
        assert c._next_port_of_ip == {}

    def test_alloc_advances_the_cursor_of_its_node(self, patch_ray_get):
        """Two allocations on the same node must hand out non-overlapping ports."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        first = cursors.alloc(engine, node_ip="10.0.0.1")
        second = cursors.alloc(engine, node_ip="10.0.0.1")
        assert second > first
        assert cursors._next_port_of_ip["10.0.0.1"] == second + 1

    def test_alloc_starts_from_the_base_port_on_an_unseen_node(self, patch_ray_get):
        """A node with no cursor yet starts at the base port, away from ray's range."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        assert cursors.alloc(engine, node_ip="10.0.0.1") == 20000

    def test_alloc_consecutive_reserves_a_whole_block(self, patch_ray_get):
        """A consecutive=N allocation must move this node's cursor past the entire block."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        first = cursors.alloc(engine, node_ip="10.0.0.1", consecutive=5)
        assert cursors._next_port_of_ip["10.0.0.1"] == first + 5

    def test_alloc_tracks_nodes_independently(self, patch_ray_get):
        """Each node ip owns its own cursor."""
        cursors = PortAllocator()
        engine_a = fake_engine(host="10.0.0.1", port_seed=0)
        engine_b = fake_engine(host="10.0.0.2", port_seed=0)
        cursors.alloc(engine_a, node_ip="10.0.0.1")
        cursors.alloc(engine_b, node_ip="10.0.0.2")
        assert set(cursors._next_port_of_ip.keys()) == {"10.0.0.1", "10.0.0.2"}

    def test_a_cursor_past_the_last_port_restarts_at_the_base_port(self, patch_ray_get):
        """Ports are never reclaimed, so a long fault-tolerance run walks the cursor off the end."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        cursors._next_port_of_ip["10.0.0.1"] = 65535

        assert cursors.alloc(engine, node_ip="10.0.0.1", consecutive=4) == 20000

    def test_a_cursor_that_still_fits_is_left_alone(self, patch_ray_get):
        """Resetting early would hand out ports that are still in use by live cells."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        cursors._next_port_of_ip["10.0.0.1"] = 65530

        assert cursors.alloc(engine, node_ip="10.0.0.1", consecutive=4) == 65530
