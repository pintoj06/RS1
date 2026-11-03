"""SearchNode methods that read the costmap and camera: obstacle lookups,
reachability, and the camera footprint used for coverage.

Costmaps here use the values Nav2 actually publishes on
global_costmap/costmap: an OccupancyGrid scaled 0-100, with 100 for lethal,
99 for inscribed and -1 for unknown. The raw 0-254 costs only appear on
costmap_raw.
"""

import math

import pytest

from conftest import FakeParams, make_costmap, set_cell

LETHAL = 100
INSCRIBED = 99


@pytest.fixture
def fake_search(search_node):
    """A stand-in SearchNode with only the state these methods read."""
    SearchNode = search_node.SearchNode

    class FakeSearch:
        _is_blocked = SearchNode._is_blocked
        _reachable_from = SearchNode._reachable_from
        _is_reachable = SearchNode._is_reachable
        _camera_fov = SearchNode._camera_fov
        _camera_altitude = SearchNode._camera_altitude
        _footprint_radius = SearchNode._footprint_radius

        def __init__(self):
            self.costmap = None
            self.cam_fx = self.cam_fy = None
            self.cam_width = self.cam_height = None
            self.get_parameter = FakeParams(
                obstacle_cost_threshold=INSCRIBED,
                fallback_camera_hfov=1.0472,
                fallback_image_width=720,
                fallback_image_height=480,
                flight_altitude=10.0,
            )

    return FakeSearch()


class TestIsBlocked:
    def test_no_costmap_means_clear(self, fake_search):
        assert fake_search._is_blocked(1.0, 1.0) is False

    def test_reads_the_right_cell(self, fake_search):
        grid = make_costmap(10, 10, resolution=0.5, origin=(-2.0, -2.0))
        set_cell(grid, col=6, row=2, value=LETHAL)
        fake_search.costmap = grid
        # Cell (col 6, row 2) spans x [1.0, 1.5), y [-1.0, -0.5).
        assert fake_search._is_blocked(1.2, -0.8)
        assert not fake_search._is_blocked(0.8, -0.8)
        assert not fake_search._is_blocked(1.2, -0.3)

    def test_threshold(self, fake_search):
        grid = make_costmap(3, 1)
        set_cell(grid, 0, 0, INSCRIBED - 1)
        set_cell(grid, 1, 0, INSCRIBED)
        fake_search.costmap = grid
        assert not fake_search._is_blocked(0.5, 0.5)
        assert fake_search._is_blocked(1.5, 0.5)

    def test_default_threshold_fits_published_scale(self, search_node, fake_search):
        # Regression: the default was once 200, on Nav2's raw 0-254 scale.
        # The published grid tops out at 100, so nothing was ever blocked and
        # the sweep was planned straight through the mountain.
        import inspect
        source = inspect.getsource(search_node.SearchNode.__init__)
        default = int(source.split("'obstacle_cost_threshold',")[1].split(')')[0])
        fake_search.get_parameter.values['obstacle_cost_threshold'] = default
        grid = make_costmap(4, 1)
        set_cell(grid, 0, 0, LETHAL)
        set_cell(grid, 1, 0, INSCRIBED)
        set_cell(grid, 2, 0, 20)          # faint inflation well clear of rock
        fake_search.costmap = grid
        assert fake_search._is_blocked(0.5, 0.5)
        assert fake_search._is_blocked(1.5, 0.5)
        assert not fake_search._is_blocked(2.5, 0.5)
        assert not fake_search._is_blocked(3.5, 0.5)

    def test_unknown_and_off_map_are_clear(self, fake_search):
        fake_search.costmap = make_costmap(4, 4, fill=-1)
        assert not fake_search._is_blocked(1.5, 1.5)
        fake_search.costmap = make_costmap(4, 4, fill=LETHAL)
        assert not fake_search._is_blocked(-3.0, 1.0)
        assert not fake_search._is_blocked(1.0, 50.0)


class TestReachability:
    def walled_pocket(self):
        """10x10 map with a closed ring of wall around cells 6..8."""
        grid = make_costmap(10, 10)
        for i in range(5, 10):
            set_cell(grid, i, 5, LETHAL)
            set_cell(grid, 5, i, LETHAL)
        for i in range(5, 10):
            set_cell(grid, i, 9, LETHAL)
            set_cell(grid, 9, i, LETHAL)
        return grid

    def test_none_without_costmap(self, fake_search):
        assert fake_search._reachable_from((1, 1)) is None
        assert fake_search._is_reachable(1, 1, None) is True

    def test_open_map_is_all_reachable(self, fake_search):
        fake_search.costmap = make_costmap(6, 6, fill=-1)
        reached = fake_search._reachable_from((0.5, 0.5))
        assert reached.all()

    def test_sealed_pocket_is_unreachable(self, fake_search):
        fake_search.costmap = self.walled_pocket()
        reached = fake_search._reachable_from((1.5, 1.5))
        assert fake_search._is_reachable(3.5, 3.5, reached)
        assert not fake_search._is_reachable(7.5, 7.5, reached)   # inside the ring
        assert not fake_search._is_reachable(5.5, 7.5, reached)   # the wall itself

    def test_gap_in_wall_lets_flood_in(self, fake_search):
        grid = self.walled_pocket()
        set_cell(grid, 5, 7, 0)       # open a door in the left wall
        fake_search.costmap = grid
        reached = fake_search._reachable_from((1.5, 1.5))
        assert fake_search._is_reachable(7.5, 7.5, reached)

    def test_start_in_wall_or_off_map(self, fake_search):
        fake_search.costmap = self.walled_pocket()
        assert fake_search._reachable_from((5.5, 5.5)) is None
        assert fake_search._reachable_from((-4.0, 2.0)) is None

    def test_off_map_point_is_left_to_nav2(self, fake_search):
        fake_search.costmap = self.walled_pocket()
        reached = fake_search._reachable_from((1.5, 1.5))
        assert fake_search._is_reachable(40.0, 40.0, reached)


class TestCameraFootprint:
    def test_fallback_fov(self, fake_search):
        hfov, vfov = fake_search._camera_fov()
        assert hfov == pytest.approx(1.0472)
        # Same focal length both ways, so tan scales with image size.
        assert math.tan(vfov / 2) == pytest.approx(math.tan(hfov / 2) * 480 / 720)

    def test_fov_from_camera_info(self, fake_search):
        fake_search.cam_fx = fake_search.cam_fy = 320.0
        fake_search.cam_width, fake_search.cam_height = 640, 480
        hfov, vfov = fake_search._camera_fov()
        assert hfov == pytest.approx(math.pi / 2)              # atan(1) * 2
        assert vfov == pytest.approx(2 * math.atan(0.75))

    def test_footprint_uses_shorter_side(self, fake_search):
        fake_search.cam_fx = fake_search.cam_fy = 320.0
        fake_search.cam_width, fake_search.cam_height = 640, 480
        # 10 m up: half-widths are 10 * 1.0 and 10 * 0.75.
        assert fake_search._footprint_radius() == pytest.approx(7.5)

    def test_footprint_scales_with_altitude(self, fake_search):
        low = fake_search._footprint_radius()
        fake_search.get_parameter.values['flight_altitude'] = 20.0
        assert fake_search._footprint_radius() == pytest.approx(2 * low)
