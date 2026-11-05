"""SearchNode methods that read the costmap and camera: obstacle lookups,
reachability, and the camera footprint used for coverage.

Costmaps here use the values Nav2 actually publishes on
global_costmap/costmap: an OccupancyGrid scaled 0-100, with 100 for lethal,
99 for inscribed and -1 for unknown. The raw 0-254 costs only appear on
costmap_raw.
"""

import math
from types import SimpleNamespace

import pytest

from conftest import (FakeLogger, FakeParams, fill_rect, make_costmap,
                      set_cell)

LETHAL = 100
INSCRIBED = 99


@pytest.fixture
def fake_search(search_node):
    """A stand-in SearchNode with only the state these methods read."""
    SearchNode = search_node.SearchNode

    class FakeSearch:
        _is_blocked = SearchNode._is_blocked
        _costmap_grid = SearchNode._costmap_grid
        _flood = SearchNode._flood
        _reachable_from = SearchNode._reachable_from
        _confirmed_from = SearchNode._confirmed_from
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
                observation_gap_fill=0.0,
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
        # Read from the whole module rather than from __init__: the
        # declarations are grouped into _declare_* methods, and a test that
        # knows WHICH method holds a parameter breaks every time they move.
        import inspect, re
        source = inspect.getsource(search_node)
        m = re.search(r"declare_parameter\(\s*'obstacle_cost_threshold'\s*,\s*(-?\d+)",
                      source)
        assert m, 'obstacle_cost_threshold is no longer declared'
        default = int(m.group(1))
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

    def test_start_in_wall_falls_back_to_the_nearest_free_cell(self, fake_search):
        """A drone hovering inside an obstacle's inflation ring reads as being
        in a wall, which happens in ordinary flight. This used to return None,
        and a caller given None stopped checking reachability at all - so the
        flood starts from the nearest free cell instead.

        What it must NOT do is seed both sides of the wall at once: the cell
        the drone is standing in is on the ring around a sealed pocket, and
        the answer has to be the OUTSIDE of it, not outside and inside merged.
        """
        fake_search.costmap = self.walled_pocket()
        reached = fake_search._reachable_from((5.5, 5.5))
        assert reached is not None
        assert fake_search._is_reachable(1.5, 1.5, reached)        # outside
        assert not fake_search._is_reachable(7.5, 7.5, reached)    # the pocket

    def test_off_map_start_has_no_answer(self, fake_search):
        fake_search.costmap = self.walled_pocket()
        assert fake_search._reachable_from((-4.0, 2.0)) is None

    def test_no_free_cell_anywhere_near(self, fake_search):
        fake_search.costmap = make_costmap(6, 6, fill=LETHAL)
        assert fake_search._reachable_from((2.5, 2.5)) is None

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


class TestConfirmedReachability:
    """The two-flood scheme, and the mountain bug it was written to fix.

    The scout flies at 10 m over rock formations 15-49 m across and 24-27 m
    tall, so its 2D lidar slices through solid stone and only ever paints the
    NEAR face. Everything behind that face - the rock's own interior included
    - stays unobserved for as long as the drone approaches from one side.
    """

    UNKNOWN = -1

    def crescent(self, observed_outside=True):
        """A wall across the middle with open ends: what one face of a rock
        looks like after a single pass. Beyond it everything is unobserved.

        Row 0 is where the drone starts. Row 4 is the wall, cells 2..7 only,
        so the flood can still run round both ends at columns 0-1 and 8-9.
        Rows 5+ have never been seen.
        """
        grid = make_costmap(10, 10)
        for row in range(5, 10):
            for col in range(10):
                set_cell(grid, col, row, self.UNKNOWN)
        for col in range(2, 8):
            set_cell(grid, col, 4, LETHAL)
        if not observed_outside:
            for col in range(10):
                set_cell(grid, col, 0, self.UNKNOWN)
        return grid

    def test_optimistic_flood_walks_straight_into_the_rock(self, fake_search):
        # This is the bug, pinned down. Unobserved ground counts as free, and
        # the mapped face is a crescent rather than a ring, so the flood runs
        # round its ends and fills the space behind it. Both of the old tests
        # therefore passed for a point in the middle of solid rock and a
        # waypoint was planted there.
        fake_search.costmap = self.crescent()
        behind = (5.5, 6.5)                       # behind the wall, unobserved
        assert not fake_search._is_blocked(*behind)       # unknown reads clear
        reached = fake_search._reachable_from((0.5, 0.5))
        assert fake_search._is_reachable(behind[0], behind[1], reached)

    def test_confirmed_flood_refuses_ground_it_has_not_seen(self, fake_search):
        # Same map, same point, honest answer: nothing confirms the drone can
        # get there, so no leg is planned there.
        fake_search.costmap = self.crescent()
        confirmed = fake_search._confirmed_from((0.5, 0.5))
        assert not fake_search._is_reachable(5.5, 6.5, confirmed)
        # Observed open ground on this side is still confirmed, so the sweep
        # is not throttled back to nothing.
        assert fake_search._is_reachable(0.5, 0.5, confirmed)
        assert fake_search._is_reachable(9.5, 3.5, confirmed)

    def test_confirmed_grows_as_the_lidar_sees_more(self, fake_search):
        """The verdict has to be revisable - that is the point of the rewrite.

        The old code remembered a no-go disc, so ground written off on the
        approach stayed written off even after the drone flew round and could
        see it was open. Here the same ground flips to confirmed as soon as it
        has been observed.
        """
        grid = self.crescent()
        fake_search.costmap = grid
        assert not fake_search._is_reachable(
            5.5, 6.5, fake_search._confirmed_from((0.5, 0.5)))
        # The drone flies round the end of the wall and sweeps its far side.
        for col in range(10):
            for row in range(5, 10):
                set_cell(grid, col, row, 0)
        fake_search.costmap = grid
        assert fake_search._is_reachable(
            5.5, 6.5, fake_search._confirmed_from((0.5, 0.5)))

    def test_rock_itself_is_never_confirmed(self, fake_search):
        fake_search.costmap = self.crescent()
        confirmed = fake_search._confirmed_from((0.5, 0.5))
        assert not fake_search._is_reachable(4.5, 4.5, confirmed)   # the wall

    def test_nothing_observed_yet_gives_empty_not_none(self, fake_search):
        """On the pad the lidar has seen nothing useful. That must read as
        "nothing confirmed yet", which is retried, not as "no costmap"."""
        fake_search.costmap = make_costmap(8, 8, fill=self.UNKNOWN)
        confirmed = fake_search._confirmed_from((0.5, 0.5))
        assert confirmed is not None
        assert not confirmed.any()
        # ...whereas the optimistic flood reaches everywhere, so the area is
        # reported as unverified rather than as rock.
        assert fake_search._reachable_from((0.5, 0.5)).all()

    def test_drone_on_an_unobserved_pad_still_gets_a_seed(self, fake_search):
        """Window seeding, not single-cell. The drone sits on the ute, so the
        cell underneath it can read blocked or unobserved; seeding from that
        one cell alone would confirm nothing at all."""
        grid = make_costmap(10, 10, resolution=1.0)
        set_cell(grid, 0, 0, LETHAL)          # the pad under the drone
        fake_search.costmap = grid
        confirmed = fake_search._confirmed_from((0.5, 0.5))
        assert confirmed.any()
        assert fake_search._is_reachable(5.5, 5.5, confirmed)

    def test_no_costmap_means_no_answer(self, fake_search):
        assert fake_search._confirmed_from((1.0, 1.0)) is None

    def test_off_map_start(self, fake_search):
        fake_search.costmap = make_costmap(4, 4)
        assert fake_search._confirmed_from((-10.0, 0.0)) is None

    def test_sealed_pocket_fails_both_floods(self, fake_search):
        """A pocket walled all the way round is ruled out even optimistically,
        which is what makes it safe to drop from the coverage total."""
        grid = make_costmap(10, 10)
        for i in range(5, 10):
            set_cell(grid, i, 5, LETHAL)
            set_cell(grid, 5, i, LETHAL)
        for i in range(5, 10):
            set_cell(grid, i, 9, LETHAL)
            set_cell(grid, 9, i, LETHAL)
        fake_search.costmap = grid
        assert not fake_search._is_reachable(
            7.5, 7.5, fake_search._reachable_from((1.5, 1.5)))
        assert not fake_search._is_reachable(
            7.5, 7.5, fake_search._confirmed_from((1.5, 1.5)))

    def test_matches_the_old_hand_written_flood(self, fake_search):
        """scipy labelling must give the same 4-connected answer the Python
        breadth-first search gave - it only replaced it for speed."""
        import numpy as np
        rng = np.random.default_rng(1234)
        grid = make_costmap(40, 40)
        for col in range(40):
            for row in range(40):
                if rng.random() < 0.3:
                    set_cell(grid, col, row, LETHAL)
        set_cell(grid, 0, 0, 0)
        fake_search.costmap = grid

        free = np.asarray(grid.data, dtype=np.int16).reshape(40, 40) < INSCRIBED
        want = np.zeros_like(free)
        want[0, 0] = True
        stack = [(0, 0)]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < 40 and 0 <= c2 < 40 and free[r2, c2] and not want[r2, c2]:
                    want[r2, c2] = True
                    stack.append((r2, c2))
        assert np.array_equal(fake_search._reachable_from((0.5, 0.5)), want)


@pytest.fixture
def fake_review(search_node):
    """A stand-in SearchNode carrying the coverage-grid state that the map
    review reads, plus a record of whether it asked for a replan."""
    import numpy as np
    SearchNode = search_node.SearchNode

    class FakeReview:
        _is_blocked = SearchNode._is_blocked
        _costmap_grid = SearchNode._costmap_grid
        _flood = SearchNode._flood
        _reachable_from = SearchNode._reachable_from
        _confirmed_from = SearchNode._confirmed_from
        _is_reachable = SearchNode._is_reachable
        _project = SearchNode._project
        _unverified_cells = SearchNode._unverified_cells
        _review_map = SearchNode._review_map

        def __init__(self):
            self.costmap = None
            self.confirmed = self.possible = None
            self._shadow_cache = self._shadow_cache_for = None
            # A 10x10 m area as 10x10 one-metre coverage cells, origin (0, 0).
            self.coverage_resolution = 1.0
            self.cov_rows = self.cov_cols = 10
            self.cov_origin_x = self.cov_origin_y = 0.0
            self.inside_mask = np.ones((10, 10), dtype=bool)
            self.cells_in_shape = 100
            self.plan_reachable_cells = None
            self.map_replans = 0
            self.area_set = True
            self.planned = True
            self.replanning = False
            self.holding = self.estopped = False
            self.busy = False
            self.goal_handle = None
            self.timed_out = False
            self._replan_wanted = False
            self._replan_is_map_driven = False
            self.replanned = False
            self.at = (0.5, 0.5)
            self.get_parameter = FakeParams(
                obstacle_cost_threshold=INSCRIBED,
                avoid_obstacles=True,
                drop_unreachable=True,
                map_review_period=2.0,
                map_review_fraction=0.02,
                max_map_replans=20,
                observation_gap_fill=0.0,
            )

        def _set_status(self, state, detail=''):
            self.statuses.append((state, detail))
            self.status_state, self.status_detail = state, detail

        def paused(self):
            return self.holding or self.estopped

        def get_logger(self):
            return FakeLogger()

        def _lookup_robot_xy(self):
            return self.at

        def _after_abandoned_goal(self):
            self.replanned = True

    return FakeReview()


class TestProjectAndUnverified:
    def test_project_resamples_onto_the_coverage_grid(self, fake_review):
        import numpy as np
        # Costmap at 0.5 m, coverage at 1.0 m: four costmap cells per
        # coverage cell, so the projection must pick, not average.
        fake_review.costmap = make_costmap(20, 20, resolution=0.5)
        grid = np.zeros((20, 20), dtype=bool)
        grid[:, :10] = True                 # x < 5 m is reachable
        out = fake_review._project(grid)
        assert out[:, :5].all()
        assert not out[:, 5:].any()

    def test_project_defaults_outside_the_costmap(self, fake_review):
        import numpy as np
        # Costmap covers only the bottom-left 4x4 m of the 10x10 m area.
        fake_review.costmap = make_costmap(4, 4, resolution=1.0)
        grid = np.zeros((4, 4), dtype=bool)
        out = fake_review._project(grid, default=True)
        assert not out[:4, :4].any()        # inside the map: what the grid says
        assert out[9, 9]                    # outside it: left to Nav2

    def test_project_without_a_costmap(self, fake_review):
        assert fake_review._project(None).all()

    def test_unverified_counts_seen_but_unconfirmed_ground(self, fake_review):
        import numpy as np
        fake_review.costmap = make_costmap(10, 10, resolution=1.0)
        fake_review.possible = np.ones((10, 10), dtype=bool)
        fake_review.confirmed = np.zeros((10, 10), dtype=bool)
        fake_review.confirmed[:5, :] = True
        assert fake_review._unverified_cells() == 50

    def test_nothing_unverified_once_everything_is_confirmed(self, fake_review):
        import numpy as np
        fake_review.costmap = make_costmap(10, 10, resolution=1.0)
        fake_review.possible = np.ones((10, 10), dtype=bool)
        fake_review.confirmed = np.ones((10, 10), dtype=bool)
        assert fake_review._unverified_cells() == 0

    def test_sealed_ground_is_not_unverified(self, fake_review):
        """Walled in with no way through is a verdict, not an open question -
        so it must not keep the sweep waiting for a better view."""
        import numpy as np
        fake_review.costmap = make_costmap(10, 10, resolution=1.0)
        fake_review.possible = np.zeros((10, 10), dtype=bool)
        fake_review.confirmed = np.zeros((10, 10), dtype=bool)
        assert fake_review._unverified_cells() == 0

    def test_unverified_is_zero_before_any_flood(self, fake_review):
        assert fake_review._unverified_cells() == 0


class TestReviewMapTrigger:
    def all_clear(self):
        return make_costmap(10, 10, resolution=1.0)

    def test_first_pass_only_records_the_baseline(self, fake_review):
        fake_review.costmap = self.all_clear()
        fake_review._review_map()
        assert fake_review.plan_reachable_cells == 100
        assert not fake_review.replanned

    def test_no_replan_when_nothing_changed(self, fake_review):
        fake_review.costmap = self.all_clear()
        fake_review._review_map()           # baseline
        fake_review._review_map()
        assert not fake_review.replanned
        assert fake_review.map_replans == 0

    def test_replans_when_ground_opens_up(self, fake_review):
        """The case the no-go list could never handle: ground that was written
        off becomes reachable once the drone can see it."""
        grid = make_costmap(10, 10, resolution=1.0, fill=-1)
        for col in range(10):               # only the first row observed
            set_cell(grid, col, 0, 0)
        fake_review.costmap = grid
        fake_review._review_map()
        assert fake_review.plan_reachable_cells == 10

        for col in range(10):               # the drone has now seen it all
            for row in range(10):
                set_cell(grid, col, row, 0)
        fake_review.costmap = grid
        fake_review._review_map()
        assert fake_review.replanned
        assert fake_review.map_replans == 1
        assert fake_review._replan_is_map_driven

    def test_replans_when_ground_turns_out_blocked(self, fake_review):
        grid = self.all_clear()
        fake_review.costmap = grid
        fake_review._review_map()
        for col in range(10):
            for row in range(5, 10):
                set_cell(grid, col, row, LETHAL)
        fake_review.costmap = grid
        fake_review._review_map()
        assert fake_review.replanned

    def test_small_change_is_ignored(self, fake_review):
        """A threshold, not any change at all - otherwise every flicker of the
        costmap would restart the sweep."""
        grid = self.all_clear()
        fake_review.costmap = grid
        fake_review._review_map()
        set_cell(grid, 0, 0, LETHAL)        # 1 cell of 100, under the 2%
        fake_review.costmap = grid
        fake_review._review_map()
        assert not fake_review.replanned

    def test_cancels_a_goal_in_flight_instead_of_replanning_under_it(self, fake_review):
        class Handle:
            def __init__(self):
                self.cancelled = False

            def cancel_goal_async(self):
                self.cancelled = True

        grid = make_costmap(10, 10, resolution=1.0, fill=-1)
        for col in range(10):
            set_cell(grid, col, 0, 0)
        fake_review.costmap = grid
        fake_review._review_map()
        fake_review.goal_handle = Handle()
        for col in range(10):
            for row in range(10):
                set_cell(grid, col, row, 0)
        fake_review.costmap = grid
        fake_review._review_map()
        assert fake_review.goal_handle.cancelled
        assert fake_review.timed_out          # goal_finished picks it up
        assert fake_review._replan_wanted
        assert not fake_review.replanned      # not replanned under the goal

    def test_cap_stops_endless_map_replans(self, fake_review):
        fake_review.get_parameter.values['max_map_replans'] = 0
        grid = make_costmap(10, 10, resolution=1.0, fill=-1)
        for col in range(10):
            set_cell(grid, col, 0, 0)
        fake_review.costmap = grid
        fake_review._review_map()
        for col in range(10):
            for row in range(10):
                set_cell(grid, col, row, 0)
        fake_review.costmap = grid
        fake_review._review_map()
        assert not fake_review.replanned
        assert fake_review.map_replans == 0

    def test_quiet_while_paused_or_unplanned(self, fake_review):
        fake_review.costmap = self.all_clear()
        for attr in ('estopped', 'holding', 'replanning'):
            setattr(fake_review, attr, True)
            fake_review._review_map()
            assert fake_review.plan_reachable_cells is None
            setattr(fake_review, attr, False)
        fake_review.planned = False
        fake_review._review_map()
        assert fake_review.plan_reachable_cells is None
        fake_review.planned = True
        fake_review.area_set = False
        fake_review._review_map()
        assert fake_review.plan_reachable_cells is None

    def test_disabled_by_period_zero(self, fake_review):
        fake_review.costmap = self.all_clear()
        fake_review.get_parameter.values['map_review_period'] = 0.0
        fake_review._review_map()
        assert fake_review.plan_reachable_cells is None

    def test_quiet_without_a_pose_or_a_costmap(self, fake_review):
        fake_review._review_map()                 # no costmap
        assert fake_review.plan_reachable_cells is None
        fake_review.costmap = self.all_clear()
        fake_review.at = None
        fake_review._review_map()
        assert fake_review.plan_reachable_cells is None


@pytest.fixture
def fake_frontier(search_node):
    """A stand-in SearchNode for the go-and-look logic."""
    import numpy as np
    SearchNode = search_node.SearchNode

    class FakeFrontier:
        _is_blocked = SearchNode._is_blocked
        _costmap_grid = SearchNode._costmap_grid
        _flood = SearchNode._flood
        _reachable_from = SearchNode._reachable_from
        _confirmed_from = SearchNode._confirmed_from
        _is_reachable = SearchNode._is_reachable
        _polygon_mask_costmap = SearchNode._polygon_mask_costmap
        _line_of_sight = SearchNode._line_of_sight
        _in_obstacle_shadow = SearchNode._in_obstacle_shadow
        _shadowed_mask = SearchNode._shadowed_mask
        _lines_clear = SearchNode._lines_clear
        _shadowed_mask = SearchNode._shadowed_mask
        _lines_clear = SearchNode._lines_clear
        _frontier_goal = SearchNode._frontier_goal

        def __init__(self):
            self.costmap = None
            self.confirmed = self.possible = None
            self._shadow_cache = self._shadow_cache_for = None
            self.polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
            self.looked_from = []
            self.at = (0.5, 0.5)
            self.get_parameter = FakeParams(
                obstacle_cost_threshold=INSCRIBED,
                observation_gap_fill=0.0,
                sensor_range=40.0,
                vantage_min_separation=3.0,
                vantage_obstacle_margin=0.0,
                min_unverified_area=0.0,
            )

        def _lookup_robot_xy(self):
            return self.at

        def flood(self):
            self.confirmed = self._confirmed_from(self.at)
            self.possible = self._reachable_from(self.at)

    return FakeFrontier()


class TestFrontier:
    UNKNOWN = -1

    def half_seen(self):
        """Bottom half observed and clear, top half never looked at."""
        grid = make_costmap(10, 10)
        for row in range(5, 10):
            for col in range(10):
                set_cell(grid, col, row, self.UNKNOWN)
        return grid

    def test_goes_to_look_at_unverified_ground(self, fake_frontier):
        """The gap that planning over confirmed ground alone leaves: without
        this the drone searches what it can confirm, reports itself finished,
        and quietly abandons ground that is in fact wide open."""
        fake_frontier.costmap = self.half_seen()
        fake_frontier.flood()
        goal = fake_frontier._frontier_goal()
        assert goal is not None
        # On the frontier: in the observed half, up against the unseen half.
        assert 3.0 <= goal[1] <= 5.5

    def test_nothing_to_look_at_when_all_confirmed(self, fake_frontier):
        fake_frontier.costmap = make_costmap(10, 10)
        fake_frontier.flood()
        assert fake_frontier._frontier_goal() is None

    def test_ignores_unverified_ground_outside_the_area(self, fake_frontier):
        """Unseen ground elsewhere in the world is not the drone's problem."""
        fake_frontier.costmap = self.half_seen()
        fake_frontier.polygon = [(0.0, 0.0), (9.0, 0.0), (9.0, 4.0), (0.0, 4.0)]
        fake_frontier.flood()
        assert fake_frontier._frontier_goal() is None

    def test_will_not_pick_where_it_already_looked(self, fake_frontier):
        """Otherwise it ping-pongs between neighbouring frontier cells for
        ever: a rock's interior is never resolved by looking from outside."""
        fake_frontier.costmap = self.half_seen()
        fake_frontier.flood()
        first = fake_frontier._frontier_goal()
        seen = {first}
        for _ in range(6):
            fake_frontier.looked_from.append(fake_frontier._frontier_goal())
            nxt = fake_frontier._frontier_goal()
            if nxt is None:
                break
            assert all(math.hypot(nxt[0] - lx, nxt[1] - ly) > 3.0
                       for lx, ly in fake_frontier.looked_from)
            seen.add(nxt)
        assert len(seen) > 1

    def test_no_vantage_when_nothing_can_see_the_target(self, fake_frontier):
        """Sealed behind mapped rock: there is no point flying anywhere."""
        grid = make_costmap(10, 10)
        for i in range(4, 9):             # a closed ring of mapped wall
            set_cell(grid, i, 4, LETHAL)
            set_cell(grid, 4, i, LETHAL)
            set_cell(grid, i, 8, LETHAL)
            set_cell(grid, 8, i, LETHAL)
        for col in range(5, 8):           # unknown inside the ring
            for row in range(5, 8):
                set_cell(grid, col, row, self.UNKNOWN)
        fake_frontier.costmap = grid
        fake_frontier.flood()
        # Sealed even optimistically, so there is nothing to go and look at.
        assert fake_frontier._frontier_goal() is None

    def test_quiet_without_floods_or_pose(self, fake_frontier):
        assert fake_frontier._frontier_goal() is None       # no costmap
        fake_frontier.costmap = self.half_seen()
        fake_frontier.flood()
        fake_frontier.at = None
        assert fake_frontier._frontier_goal() is None


class TestLineOfSight:
    def test_mapped_obstacle_blocks_sight(self, fake_frontier):
        import numpy as np
        grid = make_costmap(10, 10)
        for row in range(10):
            set_cell(grid, 5, row, LETHAL)       # a full wall at col 5
        fake_frontier.costmap = grid
        g, info = fake_frontier._costmap_grid()
        solid = g >= INSCRIBED
        xs = np.array([1.5]); ys = np.array([1.5])
        assert not fake_frontier._line_of_sight(xs, ys, (8.5, 1.5), solid, info)[0]
        assert fake_frontier._line_of_sight(xs, ys, (4.0, 1.5), solid, info)[0]

    def test_unknown_does_not_block_sight(self, fake_frontier):
        """Not having looked somewhere is the whole reason to go and look, so
        unknown cells must not count as a wall for this test."""
        import numpy as np
        grid = make_costmap(10, 10, fill=-1)
        fake_frontier.costmap = grid
        g, info = fake_frontier._costmap_grid()
        solid = g >= INSCRIBED
        xs = np.array([0.5]); ys = np.array([0.5])
        assert fake_frontier._line_of_sight(xs, ys, (9.5, 9.5), solid, info)[0]

    def test_vectorised_over_many_origins(self, fake_frontier):
        import numpy as np
        grid = make_costmap(10, 10)
        for row in range(0, 6):
            set_cell(grid, 5, row, LETHAL)       # wall only across the bottom
        fake_frontier.costmap = grid
        g, info = fake_frontier._costmap_grid()
        solid = g >= INSCRIBED
        xs = np.array([1.5, 1.5]); ys = np.array([1.5, 8.5])
        got = fake_frontier._line_of_sight(xs, ys, (8.5, 8.5), solid, info)
        assert not got[0]       # from low down, the wall is in the way
        assert got[1]           # from up high, clear over the top of it


class TestPolygonMask:
    def test_marks_the_cells_inside_the_shape(self, fake_frontier):
        import numpy as np
        fake_frontier.costmap = make_costmap(10, 10)
        fake_frontier.polygon = [(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)]
        mask = fake_frontier._polygon_mask_costmap()
        assert mask[3, 3] and mask[5, 5]
        assert not mask[1, 1] and not mask[7, 7]
        assert np.count_nonzero(mask) == 16          # a 4x4 m square

    def test_handles_a_concave_shape(self, fake_frontier):
        fake_frontier.costmap = make_costmap(12, 12)
        fake_frontier.polygon = [(1.0, 1.0), (9.0, 1.0), (9.0, 3.0),
                                 (3.0, 3.0), (3.0, 9.0), (1.0, 9.0)]
        mask = fake_frontier._polygon_mask_costmap()
        assert mask[2, 7]        # along the bottom arm of the L
        assert mask[7, 2]        # up the side arm
        assert not mask[7, 7]    # the bite out of the corner

    def test_degenerate_polygon(self, fake_frontier):
        fake_frontier.costmap = make_costmap(6, 6)
        fake_frontier.polygon = [(1.0, 1.0), (2.0, 2.0)]
        assert not fake_frontier._polygon_mask_costmap().any()


class TestGapFill:
    """The lidar's 360 rays fan out, leaving unobserved wedges between them
    past about 9 m. Without closing those the confirmed region is a thin fan
    and no leg can be planned at all - measured at 0 legs on open grass."""

    def test_bridges_a_gap_between_two_rays(self, fake_search):
        grid = make_costmap(20, 3, resolution=0.15)
        for col in range(20):            # two observed rays...
            set_cell(grid, col, 0, 0)
            set_cell(grid, col, 2, 0)
        for col in range(20):            # ...with an unobserved wedge between
            set_cell(grid, col, 1, -1)
        fake_search.costmap = grid
        fake_search.get_parameter.values['observation_gap_fill'] = 0.0
        strict = fake_search._confirmed_from((0.1, 0.075))
        assert not strict[1, 10]         # the gap reads as a wall
        fake_search.get_parameter.values['observation_gap_fill'] = 0.75
        filled = fake_search._confirmed_from((0.1, 0.075))
        assert filled[1, 10]             # bridged

    def test_never_fills_in_rock(self, fake_search):
        """The one thing it must not do: invent a way through an obstacle."""
        grid = make_costmap(20, 3, resolution=0.15)
        for col in range(20):
            set_cell(grid, col, 0, 0)
            set_cell(grid, col, 2, 0)
            set_cell(grid, col, 1, LETHAL)
        fake_search.costmap = grid
        fake_search.get_parameter.values['observation_gap_fill'] = 2.0
        filled = fake_search._confirmed_from((0.1, 0.075))
        assert not filled[1, 10]


class TestNoUndefinedNames:
    """Catch the class of bug that killed the scout mid-flight.

    plan_sweep referred to a local called `reached` that an earlier edit had
    renamed to self.confirmed. Nothing here called plan_sweep, and a NameError
    is a RUNTIME error, so both the unit tests and a syntax check passed - and
    the node died the first time an operator drew a search area. A static
    check over the whole file costs nothing and would have caught it.
    """

    def source_files(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        return ([p for p in (root / 'scripts').glob('*.py')]
                + [p for p in (root / 'UI').glob('*.py')])

    def test_no_undefined_names(self):
        api = pytest.importorskip('pyflakes.api')
        from pyflakes import reporter as pyflakes_reporter
        import io

        problems = []
        for path in self.source_files():
            out, err = io.StringIO(), io.StringIO()
            api.checkPath(str(path), pyflakes_reporter.Reporter(out, err))
            for line in (out.getvalue() + err.getvalue()).splitlines():
                # Unused imports are untidy, not dangerous. Undefined names
                # are the ones that take a node down in flight.
                if 'undefined name' in line or 'syntax' in line.lower():
                    problems.append(line)
        assert not problems, 'undefined names:\n' + '\n'.join(problems)


@pytest.fixture
def planning_node(search_node):
    """A SearchNode complete enough to run plan_sweep end to end.

    Worth the setup: plan_sweep is where the sweep, the two floods, the
    frontier trip and the coverage mask all meet, and it was the one function
    with no test at all when it crashed the scout.
    """
    import numpy as np
    SearchNode = search_node.SearchNode

    class PlanningNode:
        for _name in ('_is_blocked', '_costmap_grid', '_flood', '_reachable_from',
                      '_confirmed_from', '_is_reachable', '_project',
                      '_polygon_mask_costmap', '_line_of_sight',
                      '_frontier_goal', '_in_obstacle_shadow',
                      '_shadowed_mask', '_lines_clear',
                      '_unverified_cells', '_worth_looking',
                      '_write_off_unverified',
                      '_build_inside_mask', '_refresh_searchable',
                      '_predicted_coverage', '_already_searched',
                      '_camera_fov', '_camera_altitude', '_footprint_radius',
                      'plan_sweep', '_blocked_test', '_skip_searched_ground',
                      '_choose_spacing', '_nothing_to_sweep', '_finish_plan',
                      '_log_plan', '_review_map', '_publish_coverage',
                      '_publish_vantage',
                      '_maybe_return_home', '_after_last_waypoint',
                      '_start_pass', '_fly_home', '_coverage_summary'):
            locals()[_name] = getattr(SearchNode, _name)
        del _name
        # Still a staticmethod here, or self would be passed as its first
        # argument and every cell would test as outside the polygon.
        point_in_polygon = staticmethod(SearchNode.point_in_polygon)

        def __init__(self, polygon, costmap):
            self.polygon = polygon
            self.costmap = costmap
            self.cam_fx = self.cam_fy = 360.0
            self.cam_width, self.cam_height = 720, 480
            self.confirmed = self.possible = None
            self._shadow_cache = self._shadow_cache_for = None
            self.unverifiable = None
            self.looked_from = []
            self.unverified_at_vantage = None
            self.fruitless_vantages = 0
            self.plan_reachable_cells = None
            self.map_replans = 0
            self.replanning = False
            self.planned = False
            self.replans = 0
            self.points = []
            self.home_index = None
            self.spacing = 0.0
            self.at = (0.5, 0.5)
            self.logged = []
            self.completed = 0
            # Coverage grid over the polygon plus a small pad, 0.5 m cells.
            xs = [p[0] for p in polygon]; ys = [p[1] for p in polygon]
            self.coverage_resolution = 0.5
            self.cov_origin_x = min(xs) - 2.0
            self.cov_origin_y = min(ys) - 2.0
            self.cov_cols = int((max(xs) + 2.0 - self.cov_origin_x) / 0.5) + 1
            self.cov_rows = int((max(ys) + 2.0 - self.cov_origin_y) / 0.5) + 1
            self.coverage = np.zeros((self.cov_rows, self.cov_cols), dtype=np.int8)
            self.cells_in_shape = 0
            self.map_frame = 'parrot1_odom'
            self.vantages = []
            self.vantage_pub = SimpleNamespace(publish=self.vantages.append)
            self.statuses = []
            self.status_state = None
            self.status_detail = ''
            self.returning = False
            self.vantage_index = None
            self.last_coverage_log = 0.0
            from builtin_interfaces.msg import Time as _T
            self.get_clock = lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(to_msg=lambda: _T()))
            self.get_parameter = FakeParams(
                obstacle_cost_threshold=INSCRIBED,
                avoid_obstacles=True,
                drop_unreachable=True,
                observation_gap_fill=0.0,
                sensor_range=40.0,
                vantage_min_separation=3.0,
                vantage_obstacle_margin=0.0,
                min_unverified_area=0.0,
                vantage_patience=5,
                vantage_progress_fraction=0.02,
                spacing=0.0,
                overlap=0.8,
                obstacle_clearance=1.0,
                min_leg_length=1.5,
                return_to_spawn=True,
                flight_altitude=10.0,
                fallback_camera_hfov=1.0472,
                fallback_image_width=720,
                fallback_image_height=480,
            )
            self.inside_mask = self._build_inside_mask()
            self.cells_to_search = int(np.count_nonzero(self.inside_mask))

        def _lookup_robot_xy(self):
            return self.at

        def _set_status(self, state, detail=''):
            self.statuses.append((state, detail))
            self.status_state, self.status_detail = state, detail

        def get_logger(self):
            node = self

            class L:
                def __getattr__(self, level):
                    def log(msg):
                        node.logged.append(msg)
                    return log
            return L()

        class _Pub:
            def __init__(self, node):
                self.node = node

            def publish(self, _msg):
                self.node.completed += 1

        @property
        def sweep_complete_pub(self):
            return self._Pub(self)

        def _publish_path(self):
            pass

    return PlanningNode


class TestPlanSweep:
    """The integration the crash exposed as untested."""

    def box(self):
        return [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]

    def test_plans_a_sweep_over_open_ground(self, planning_node):
        """The exact path that crashed: operator draws an area, node plans."""
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        node = planning_node(self.box(), grid)
        assert node.plan_sweep() is True
        assert node.planned
        assert len(node.points) > 2
        # Return-to-spawn tacked on the end, and no go-and-look trip needed
        # because everything in the area is observed and clear.
        assert node.points[-1] == (0.0, 0.0)
        assert node.home_index == len(node.points) - 1
        assert node.looked_from == []
        assert node.plan_reachable_cells > 0

    def test_waits_for_the_camera(self, planning_node):
        node = planning_node(self.box(), make_costmap(60, 60, resolution=0.5))
        node.cam_fx = None
        assert node.plan_sweep() is False

    def test_waits_for_the_costmap(self, planning_node):
        node = planning_node(self.box(), None)
        assert node.plan_sweep() is False
        assert any('costmap' in m for m in node.logged)

    def test_no_waypoint_lands_in_an_obstacle(self, planning_node):
        """A rock across the middle of the area, mapped."""
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        fill_rect(grid, 20, 36, 20, 30, LETHAL)
        node = planning_node(self.box(), grid)
        assert node.plan_sweep() is True
        for x, y in node.points:
            assert not node._is_blocked(x, y), f'waypoint ({x}, {y}) is in rock'

    def test_unobserved_area_goes_to_look_instead_of_waiting(self, planning_node):
        """On the pad the lidar has seen nothing useful, so nothing is
        confirmed reachable and no leg can be planned.

        That must not be reported as "your area is solid rock", and it must
        not WAIT either. Waiting was a deadlock: the view only improves if the
        drone moves, and it only moves if there is a plan - the drone sat
        parked for 213 s of sim time with nothing commanding it. The answer is
        to fly to one vantage point, which is what produces the better view.
        """
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0), fill=-1)
        fill_rect(grid, 0, 60, 0, 1, 0)    # just a strip of observed ground
        node = planning_node(self.box(), grid)
        node.at = (0.5, -4.75)
        assert node.plan_sweep() is True
        assert node.points, 'no plan at all - the drone would sit still'
        assert len(node.looked_from) == 1
        assert node.points[-1] == node.looked_from[0]
        assert not any('NOT SEARCHABLE' in m for m in node.logged)

    def test_goes_to_look_when_part_cannot_be_judged(self, planning_node):
        """Bottom half observed, top half never seen: plan the confirmed half
        and end at a vantage point rather than going home none the wiser."""
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        fill_rect(grid, 0, 60, 30, 60, -1)
        node = planning_node(self.box(), grid)
        assert node.plan_sweep() is True
        assert len(node.looked_from) == 1
        assert node.points[-1] == node.looked_from[0]
        assert node.home_index is None     # not home yet, still work to do
        assert any('to get a view of it' in m for m in node.logged)

    def test_area_entirely_inside_rock_is_reported(self, planning_node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0),
                            fill=LETHAL)
        fill_rect(grid, 0, 4, 0, 4, 0)     # a scrap of clear ground to sit on
        node = planning_node(self.box(), grid)
        node.at = (-4.0, -4.0)
        assert node.plan_sweep() is True
        assert node.points == []
        assert any('NOT SEARCHABLE' in m for m in node.logged)

    def test_replan_keeps_searched_ground_out(self, planning_node):
        import numpy as np
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        node = planning_node(self.box(), grid)
        node.plan_sweep()
        # Mark the bottom half as already swept, then replan.
        node.coverage[:node.cov_rows // 2, :] = 100
        node.replanning = True
        node.planned = False
        node.points = []
        assert node.plan_sweep() is True
        assert node.points
        # The point is WHERE it replans, not how many waypoints it uses: every
        # sweep waypoint must be on ground the camera has not already covered.
        swept_y = node.cov_origin_y + (node.cov_rows // 2) * node.coverage_resolution
        for x, y in node.points[:-1]:
            assert not node._already_searched(x, y), f'({x}, {y}) already done'
            assert y >= swept_y, f'({x}, {y}) is back in the searched half'
        assert node.points[-1] == (0.0, 0.0)


class TestReviewDoesNotRetrigger:
    """A replan is asked for by cancelling the goal in flight, and
    self.replanning is not set until that cancel lands - which can be several
    seconds. The review must not keep firing in the meantime: in a live run it
    reported the identical "276 m2 has opened up" every few seconds and spent
    map replan 2/20, 3/20, 4/20 on a map that had not changed at all.
    """

    def opening_up(self, fake_review):
        grid = make_costmap(10, 10, resolution=1.0, fill=-1)
        for col in range(10):
            set_cell(grid, col, 0, 0)
        fake_review.costmap = grid
        fake_review._review_map()                 # baseline
        for col in range(10):
            for row in range(10):
                set_cell(grid, col, row, 0)
        fake_review.costmap = grid
        return grid

    def test_one_replan_per_change(self, fake_review):
        class Handle:
            cancels = 0

            def cancel_goal_async(self):
                Handle.cancels += 1

        self.opening_up(fake_review)
        fake_review.goal_handle = Handle()
        for _ in range(5):                        # five review ticks
            fake_review._review_map()
        assert fake_review.map_replans == 1
        assert Handle.cancels == 1

    def test_baseline_moves_on_immediately(self, fake_review):
        """Even with no goal to cancel, the figure acted on must be recorded
        at once, or the next tick reports the same change again."""
        self.opening_up(fake_review)
        fake_review._review_map()
        assert fake_review.replanned
        assert fake_review.plan_reachable_cells == 100
        fake_review.replanned = False
        fake_review._replan_wanted = False        # as plan_sweep would leave it
        fake_review._review_map()
        assert not fake_review.replanned

    def test_a_later_real_change_still_replans(self, fake_review):
        """The guard must not wedge it shut for good."""
        self.opening_up(fake_review)
        fake_review._review_map()
        assert fake_review.map_replans == 1
        fake_review._replan_wanted = False        # the replan has now run
        grid = fake_review.costmap
        for col in range(10):                     # half of it turns to rock
            for row in range(5, 10):
                set_cell(grid, col, row, LETHAL)
        fake_review.costmap = grid
        fake_review._review_map()
        assert fake_review.map_replans == 2


class TestPredictedCoverageIgnoresUnreachable:
    """The self-check marks the plan against its own job.

    Legs are only planned over confirmed ground, so counting ground the drone
    cannot confirm a way to made a sound plan score badly and sent the spacing
    retry loop tightening for nothing - 10.67 m down to 4.0 m, 2.7x the legs,
    with the uncovered part being rock either way.
    """

    def test_unconfirmed_ground_is_not_counted_against_the_plan(self, planning_node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        fill_rect(grid, 0, 60, 30, 60, -1)        # top half never observed
        node = planning_node([(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
                             grid)
        node.plan_sweep()
        assert not any('tightening' in m for m in node.logged), \
            'spacing was tightened over ground the sweep never meant to cover'
        assert not any('will NOT be searched' in m for m in node.logged)

    def test_still_tightens_for_a_genuine_gap(self, planning_node):
        """It must keep working for what it was written for: a shape whose arm
        is reached by a single sweep row and is taller than the footprint."""
        grid = make_costmap(120, 120, resolution=0.5, origin=(-5.0, -5.0))
        cross = [(18.0, 0.0), (26.0, 0.0), (26.0, 18.0), (44.0, 18.0),
                 (44.0, 26.0), (26.0, 26.0), (26.0, 44.0), (18.0, 44.0),
                 (18.0, 26.0), (0.0, 26.0), (0.0, 18.0), (18.0, 18.0)]
        node = planning_node(cross, grid)
        node.get_parameter.values['spacing'] = 0.0
        node.plan_sweep()
        covered = node._predicted_coverage(node.points, node._footprint_radius())
        assert covered >= 0.999, f'cross shape only {100*covered:.1f}% covered'


class TestRealNode:
    """Drive the REAL SearchNode, not a stand-in.

    Both crashes that reached the operator got through because every test used
    a fake self. plan_sweep referred to a local that had been renamed
    (NameError), and _review_map read an attribute that only came into
    existence on the first abandoned waypoint (AttributeError) - and the fakes
    happened to define it. A real object, with the real __init__, is the only
    thing that catches that class of bug.
    """

    @pytest.fixture
    def node(self, search_node):
        rclpy = pytest.importorskip('rclpy')
        if not rclpy.ok():
            rclpy.init()
        n = search_node.SearchNode()
        yield n
        n.destroy_node()

    def test_constructs_and_the_timers_are_safe_to_call(self, node):
        """Every periodic callback must survive being called with no sim
        running - that is the state the node starts in."""
        node.tick()
        node._review_map()
        node.record_trail()
        node.update_coverage()
        node.publish_search_area()
        node._log_footprint()

    def test_plans_for_real_from_a_drawn_area(self, node, search_node):
        """The exact sequence that killed the scout: camera info, a costmap,
        an operator drawing an area, then the node planning."""
        import numpy as np
        node.cam_fx = node.cam_fy = 360.0
        node.cam_width, node.cam_height = 720, 480
        node.polygon = [(0.0, 2.0), (16.0, 2.0), (16.0, 18.0), (0.0, 18.0)]
        node.area_set = True
        node._rebuild_coverage_grid()
        # TF is not available in a unit test, so stand in for it.
        node._lookup_robot_xy = lambda: (0.0, 0.0)

        grid = make_costmap(300, 300, resolution=0.2, origin=(-20.0, -20.0))
        fill_rect(grid, 0, 300, 140, 150, LETHAL)   # a mapped wall across it
        node.costmap = grid

        assert node.plan_sweep() is True
        assert node.points, 'no path planned'
        for x, y in node.points:
            assert not node._is_blocked(x, y), f'waypoint ({x}, {y}) is in rock'
        # And the review runs against that plan without falling over.
        node._review_map()

    def test_review_runs_repeatedly_without_replanning_a_stable_map(self, node):
        node.cam_fx = node.cam_fy = 360.0
        node.cam_width, node.cam_height = 720, 480
        node.polygon = [(0.0, 2.0), (12.0, 2.0), (12.0, 14.0), (0.0, 14.0)]
        node.area_set = True
        node._rebuild_coverage_grid()
        node._lookup_robot_xy = lambda: (0.0, 0.0)
        node.costmap = make_costmap(300, 300, resolution=0.2, origin=(-20.0, -20.0))
        assert node.plan_sweep() is True
        before = node.map_replans
        for _ in range(6):
            node._review_map()
        assert node.map_replans == before, 'replanned on a map that did not change'


class TestUnknownTrackingGuard:
    """The misconfiguration that silently undid the whole feature.

    With track_unknown_space off, Nav2 stores "never observed" and "observed
    and clear" both as 0. The confirmed and possible floods then come out
    identical, the inside of a rock reads as open ground, and waypoints get
    planned in it - with nothing in the log to say why. Measured on a live
    costmap: one summit's footprint was 78.8% "clear" and 0% unknown.
    """

    @pytest.fixture
    def node(self, search_node):
        SearchNode = search_node.SearchNode

        class N:
            _check_unknown_tracking = SearchNode._check_unknown_tracking

            def __init__(self):
                self._unknown_checked = False
                self.errors = []

            def get_logger(self):
                node = self

                class L:
                    def error(self, msg):
                        node.errors.append(msg)

                    def __getattr__(self, _):
                        return lambda *a, **k: None
                return L()

        return N()

    def test_complains_when_nothing_is_unknown(self, node):
        grid = make_costmap(10, 10, fill=0)
        set_cell(grid, 5, 5, LETHAL)          # something has been mapped
        node._check_unknown_tracking(grid)
        assert node.errors and 'UNKNOWN' in node.errors[0]

    def test_silent_when_unknown_is_tracked(self, node):
        grid = make_costmap(10, 10, fill=-1)
        set_cell(grid, 5, 5, LETHAL)
        node._check_unknown_tracking(grid)
        assert not node.errors

    def test_an_empty_costmap_is_not_a_misconfiguration(self, node):
        """All zeros early on just means nothing has been mapped yet."""
        node._check_unknown_tracking(make_costmap(10, 10, fill=0))
        assert not node.errors

    def test_says_it_once_only(self, node):
        grid = make_costmap(10, 10, fill=0)
        set_cell(grid, 5, 5, LETHAL)
        for _ in range(5):
            node._check_unknown_tracking(grid)
        assert len(node.errors) == 1


class TestPendingReplanAlwaysRuns:
    """The freeze that parked the drone for 213 s of sim time.

    A replan is asked for by setting _replan_wanted and cancelling the goal in
    flight; the goal's result callback then carries it out. But not every path
    through that callback runs - a goal Nav2 REJECTS clears busy and returns
    without producing a result - so the replan was never carried out. And
    _review_map suppresses itself while one is pending, so it never asked
    again. Net effect: the drone sat at a vantage point, no active goal,
    nothing commanding it, the node ticking away happily.

    So the tick carries out a pending replan itself, whatever became of the
    goal.
    """

    @pytest.fixture
    def search(self, search_node):
        SearchNode = search_node.SearchNode

        class Fake:
            tick = SearchNode.tick

            def __init__(self):
                self.pending_area = None
                self.area_set = True
                self.planned = True
                self.holding = self.estopped = False
                self.busy = False
                self._replan_wanted = False
                self.returning = False
                self.vantage_index = None
                self.statuses = []
                self.status_state = None
                self.status_detail = ''
                self.index = 0
                self.points = [(1.0, 1.0)]
                self.replanned = 0
                self.stall_checks = 0
                self.sent = []
                self.nav_client = SimpleNamespace(server_is_ready=lambda: True)

            def _set_status(self, state, detail=''):
                self.statuses.append((state, detail))

            def paused(self):
                return self.holding or self.estopped

            def get_logger(self):
                return FakeLogger()

            def send_goal(self, p):
                self.sent.append(p)

            def _check_stalled(self):
                self.stall_checks += 1

            def _after_abandoned_goal(self):
                self.replanned += 1
                self._replan_wanted = False

        return Fake()

    def test_tick_carries_out_a_pending_replan(self, search):
        search._replan_wanted = True
        search.tick()
        assert search.replanned == 1
        assert search.sent == []          # replans, does not fly the old plan

    def test_and_then_gets_on_with_flying(self, search):
        search._replan_wanted = True
        search.tick()                     # does the replan
        search.tick()                     # next tick flies again
        assert search.replanned == 1
        assert search.sent == [(1.0, 1.0)]

    def test_a_goal_in_flight_is_left_alone(self, search):
        """While a goal is still in flight the cancel route handles it; the
        tick must not replan from under it."""
        search.busy = True
        search._replan_wanted = True
        search.tick()
        assert search.replanned == 0
        assert search.stall_checks == 1

    def test_nothing_pending_means_normal_flying(self, search):
        search.tick()
        assert search.replanned == 0
        assert search.sent == [(1.0, 1.0)]

    def test_estop_still_wins(self, search):
        search.estopped = True
        search._replan_wanted = True
        search.tick()
        assert search.replanned == 0
        assert search.sent == []

    def test_exhausted_plan_no_longer_wedges(self, search):
        """The exact frozen state: plan flown out, a replan pending."""
        search.index = len(search.points)
        search._replan_wanted = True
        search.tick()
        assert search.replanned == 1


class TestReplanSelfCheck:
    """The spiral seen in flight at 88% coverage.

    On a replan the sweep only covers what is LEFT, but the self-check was
    still marking it against the whole area. With almost everything already
    searched it scored near zero, so the retry loop tightened the spacing six
    times - 8.98 m down to 3.4 m - and then warned "100.0% will NOT be
    searched" on a search that was nearly finished.
    """

    def box(self):
        return [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]

    def test_no_tightening_when_the_area_is_nearly_done(self, planning_node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        node = planning_node(self.box(), grid)
        node.plan_sweep()
        node.logged.clear()
        node.coverage[:, :] = 100          # the camera has been everywhere
        node.replanning = True
        node.planned = False
        node.points = []
        node.plan_sweep()
        assert not any('tightening' in m for m in node.logged), \
            'tightened the spacing with nothing left to cover'
        assert not any('will NOT be searched' in m for m in node.logged)

    def test_nothing_outstanding_counts_as_covered(self, planning_node):
        """0.0 was the wrong answer for "no ground outstanding" - that is
        what set the retry loop off."""
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        node = planning_node(self.box(), grid)
        node.plan_sweep()
        node.coverage[:, :] = 100
        node.replanning = True
        assert node._predicted_coverage([], node._footprint_radius()) == 1.0

    def test_still_measures_a_real_gap_on_a_replan(self, planning_node):
        """It must not become a rubber stamp: with half the area unsearched
        and a path that covers none of it, the score has to be low."""
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        node = planning_node(self.box(), grid)
        node.plan_sweep()
        node.coverage[:node.cov_rows // 2, :] = 100     # bottom half done
        node.replanning = True
        covered = node._predicted_coverage([(0.0, 1.0), (1.0, 1.0)],
                                           node._footprint_radius())
        assert covered < 0.5, f'scored {covered:.2f} for covering almost none of it'


class TestVantageKeepsClearOfRock:
    """Three of five failure replans went on this in one flight.

    A frontier cell is clear when it is picked, but it sits at the edge of
    what has been seen - so the next scan fills in rock beside it, inflation
    swallows it, and the drone arrives at a waypoint that has turned
    unreachable. That counts as a FAILED waypoint, so it spends one of the
    failure-replan budget rather than the map-replan one.
    """

    UNKNOWN = -1

    def rock_at_the_frontier(self):
        """Observed ground up to row 5, rock along row 5, unknown beyond."""
        grid = make_costmap(30, 30, resolution=1.0)
        fill_rect(grid, 0, 30, 6, 30, self.UNKNOWN)
        fill_rect(grid, 10, 20, 5, 6, LETHAL)
        return grid

    def test_picks_a_vantage_away_from_the_rock(self, fake_frontier):
        import numpy as np
        fake_frontier.costmap = self.rock_at_the_frontier()
        fake_frontier.polygon = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
        fake_frontier.get_parameter.values['vantage_obstacle_margin'] = 3.0
        fake_frontier.flood()
        goal = fake_frontier._frontier_goal()
        assert goal is not None
        g, info = fake_frontier._costmap_grid()
        solid = np.argwhere(g >= INSCRIBED)
        ys = info.origin.position.y + (solid[:, 0] + 0.5) * info.resolution
        xs = info.origin.position.x + (solid[:, 1] + 0.5) * info.resolution
        nearest = float(np.min(np.hypot(xs - goal[0], ys - goal[1])))
        assert nearest >= 3.0, f'vantage only {nearest:.1f} m from mapped rock'

    def test_margin_zero_restores_the_old_behaviour(self, fake_frontier):
        fake_frontier.costmap = self.rock_at_the_frontier()
        fake_frontier.polygon = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
        fake_frontier.get_parameter.values['vantage_obstacle_margin'] = 0.0
        fake_frontier.flood()
        assert fake_frontier._frontier_goal() is not None

    def test_takes_what_there_is_when_nothing_is_roomy(self, fake_frontier):
        """A margin nothing can satisfy must not stop it looking altogether."""
        fake_frontier.costmap = self.rock_at_the_frontier()
        fake_frontier.polygon = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
        fake_frontier.get_parameter.values['vantage_obstacle_margin'] = 500.0
        fake_frontier.flood()
        assert fake_frontier._frontier_goal() is not None


class TestFourStateCoverage:
    """The overlay has to say WHY ground is unpainted.

    Everything used to collapse to -1, so "searched", "could not reach" and
    "never worked out whether I could reach it" drew as blue, blank, blank.
    Measured on one flight: of 619 unpainted cells inside the area, 478 (77%)
    were unknown rather than obstacle - so the category the operator most
    needs was the one being hidden.
    """

    @pytest.fixture
    def node(self, planning_node):
        import numpy as np
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        n = planning_node([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                          grid)
        n.sent = []
        n.coverage_pub = SimpleNamespace(publish=n.sent.append)
        n.last_coverage_log = 0.0
        return n

    def values(self, node):
        import numpy as np
        node._publish_coverage()
        m = node.sent[-1]
        return np.asarray(m.data, dtype=np.int8).reshape(
            m.info.height, m.info.width)

    def test_the_four_states_are_distinguishable(self, search_node, node):
        import numpy as np
        sn = search_node
        node.coverage[:, :] = 0
        # mark a strip as swept
        node.coverage[node.in_shape_mask.any(axis=1), :] = 0
        swept = node.in_shape_mask.copy()
        swept[:, 10:] = False
        node.coverage[swept] = 100
        g = self.values(node)
        assert (g == sn.COVER_OUTSIDE).any(), 'nothing marked outside the area'
        assert (g == sn.COVER_SEARCHED).any(), 'nothing marked searched'
        assert (g == sn.COVER_TO_DO).any(), 'nothing marked still-to-do'
        # every in-shape cell must carry one of the in-area values
        inside = g[node.in_shape_mask]
        assert set(np.unique(inside)) <= {sn.COVER_TO_DO, sn.COVER_UNVERIFIED,
                                          sn.COVER_BLOCKED, sn.COVER_SEARCHED}

    def test_outside_the_area_says_nothing(self, search_node, node):
        g = self.values(node)
        assert (g[~node.in_shape_mask] == search_node.COVER_OUTSIDE).all()

    def test_searched_wins_over_blocked(self, search_node, node):
        """Ground the camera HAS been over stays marked searched even if the
        costmap later calls it rock - we did look at it."""
        import numpy as np
        rows, cols = np.nonzero(node.in_shape_mask)
        r, c = rows[0], cols[0]
        node.blocked_mask[r, c] = True
        node.coverage[r, c] = 100
        assert self.values(node)[r, c] == search_node.COVER_SEARCHED

    def test_blocked_ground_is_marked_blocked_not_blank(self, search_node, node):
        import numpy as np
        rows, cols = np.nonzero(node.in_shape_mask)
        # a block big enough to survive the speckle filter
        for r, c in zip(rows[:9], cols[:9]):
            node.blocked_mask[r, c] = True
        g = self.values(node)
        assert (g == search_node.COVER_BLOCKED).any()

    def test_unverified_ground_is_marked_unverified(self, search_node, node):
        """The important one: not confirmed reachable is its own state, not
        the same as empty space off the edge of the map."""
        import numpy as np
        node.confirmed = np.zeros((60, 60), dtype=bool)   # nothing confirmed
        g = self.values(node)
        assert (g == search_node.COVER_UNVERIFIED).any()
        assert (g[node.in_shape_mask] != search_node.COVER_OUTSIDE).all()


class TestSpeckleFilter:
    """A coverage cell is 0.5 m, the costmap 0.15 m, and the cell is judged by
    its CENTRE - so one inflated costmap cell carved a whole 0.25 m2 square
    out of the search area. Nine holes were measured in one overlay, four of
    them one to four cells.
    """

    def box(self):
        return [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    def test_a_single_inflated_cell_does_not_punch_a_hole(self, planning_node):
        grid = make_costmap(200, 200, resolution=0.15, origin=(-5.0, -5.0))
        # one lethal costmap cell, right on a coverage cell centre
        set_cell(grid, int((5.25 + 5.0) / 0.15), int((5.25 + 5.0) / 0.15), LETHAL)
        node = planning_node(self.box(), grid)
        assert not node.blocked_mask.any(), 'a 0.25 m2 speck was treated as obstacle'

    def test_a_real_obstacle_is_still_excluded(self, planning_node):
        """It must not become a blanket amnesty for small obstacles."""
        grid = make_costmap(200, 200, resolution=0.15, origin=(-5.0, -5.0))
        lo = int((3.0 + 5.0) / 0.15)
        hi = int((6.0 + 5.0) / 0.15)
        fill_rect(grid, lo, hi, lo, hi, LETHAL)       # a 3 x 3 m rock
        node = planning_node(self.box(), grid)
        assert node.blocked_mask.sum() > 20, 'a real obstacle was filtered away'
        assert not node.inside_mask[node.blocked_mask].any()


class TestSaysWhatItIsDoing:
    """The scout flew out of the search area with the green path gone and
    nothing on screen to say why. From the operator's seat a finished search,
    a go-and-look trip and a crash all looked identical. Each one now names
    itself on search_status.
    """

    def box(self):
        return [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]

    def flyable(self, node, search_node):
        """Give the fake just enough to let the real send_goal run."""
        node.goal_best_dist = None
        node.goal_progress_at = node.goal_sent_at = node.last_recheck = 0.0
        node.goal_leg_length = 0.0
        node.rejections = 0
        node._now = lambda: 0.0
        node.nav_client = SimpleNamespace(
            send_goal_async=lambda g: SimpleNamespace(
                add_done_callback=lambda cb: None))
        node.goal_response = lambda f: None
        node.send_goal = search_node.SearchNode.send_goal.__get__(node)

    def test_a_look_around_trip_is_named(self, planning_node, search_node):
        """The one that mattered: a single waypoint outside the search area,
        with no sweep pattern to draw. It has to announce itself as LOOKING,
        or it is indistinguishable from the drone wandering off."""
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        fill_rect(grid, 0, 60, 30, 60, -1)       # top half never observed
        node = planning_node(self.box(), grid)
        assert node.plan_sweep() is True
        assert node.vantage_index == len(node.points) - 1
        self.flyable(node, search_node)
        node.index = node.vantage_index
        node.send_goal(node.points[node.index])
        assert node.status_state == search_node.ST_LOOKING
        assert 'see it' in node.status_detail, node.status_detail

    def test_a_sweep_leg_is_named_searching(self, planning_node, search_node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        fill_rect(grid, 0, 60, 30, 60, -1)
        node = planning_node(self.box(), grid)
        node.plan_sweep()
        self.flyable(node, search_node)
        node.index = 0
        node.send_goal(node.points[0])
        assert node.status_state == search_node.ST_SEARCHING
        assert 'waypoint 1' in node.status_detail

    def test_the_way_home_is_named_returning(self, planning_node, search_node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        node = planning_node(self.box(), grid)
        node.plan_sweep()
        self.flyable(node, search_node)
        node.returning = True
        node.index = 0
        node.send_goal((0.0, 0.0))
        assert node.status_state == search_node.ST_RETURNING

    def test_no_vantage_means_no_vantage_index(self, planning_node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        node = planning_node(self.box(), grid)
        node.plan_sweep()
        assert node.vantage_index is None
        assert node.points[-1] == (0.0, 0.0)         # home instead

    def test_unsearchable_area_says_so(self, planning_node, search_node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0),
                            fill=LETHAL)
        fill_rect(grid, 0, 4, 0, 4, 0)
        node = planning_node(self.box(), grid)
        node.at = (-4.0, -4.0)
        node.plan_sweep()
        assert search_node.ST_UNSEARCHABLE in [st for st, _ in node.statuses]


class TestReturnsHome:
    """The run used to end wherever the last look-around trip left the drone -
    out by the rock, no path drawn, nothing commanding it, which reads as a
    crash. A pass ending on a vantage point gets no home waypoint on purpose,
    because the run is meant to carry on; but when coverage completed DURING
    that trip, nothing ever planned again.
    """

    def box(self):
        return [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]

    def finished(self, planning_node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        node = planning_node(self.box(), grid)
        node.plan_sweep()
        node.coverage[:, :] = 100          # everything swept
        node.at = (18.0, 18.0)             # parked out at the far corner
        node.index = len(node.points)      # plan flown out
        return node

    def test_flies_home_when_finished(self, planning_node, search_node):
        node = self.finished(planning_node)
        node._maybe_return_home()
        assert node.returning
        assert node.points == [(0.0, 0.0)]
        assert node.home_index == 0
        assert node.index == 0
        assert node.planned
        assert search_node.ST_RETURNING in [st for st, _ in node.statuses]

    def test_does_not_go_home_while_there_is_somewhere_left_to_look(
            self, planning_node):
        """Flying home now would abandon ground that is about to become
        reachable - the map review is still expected to reopen the search."""
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        fill_rect(grid, 0, 60, 30, 60, -1)
        node = planning_node(self.box(), grid)
        node.plan_sweep()
        node.coverage[:, :] = 100
        node.at = (10.0, 10.0)
        node.index = len(node.points)
        node.returning = False
        node._maybe_return_home()
        assert not node.returning, 'went home with unverified ground left'

    def test_already_home_reports_complete(self, planning_node, search_node):
        node = self.finished(planning_node)
        node.at = (0.5, 0.5)
        node._maybe_return_home()
        assert search_node.ST_COMPLETE in [st for st, _ in node.statuses]
        assert node.points != [(0.0, 0.0)] or node.index != 0

    def test_only_goes_home_once(self, planning_node):
        node = self.finished(planning_node)
        node._maybe_return_home()
        node.points = []
        node._maybe_return_home()
        assert node.points == []          # not replanned a second time

    def test_respects_return_to_spawn_false(self, planning_node, search_node):
        node = self.finished(planning_node)
        node.get_parameter.values['return_to_spawn'] = False
        node._maybe_return_home()
        assert not node.returning
        assert search_node.ST_COMPLETE in [st for st, _ in node.statuses]

    def test_the_summary_mentions_the_gap(self, planning_node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        fill_rect(grid, 20, 36, 20, 30, LETHAL)
        node = planning_node(self.box(), grid)
        node.plan_sweep()
        node.coverage[:, :] = 100
        text = node._coverage_summary()
        assert 'not checked' in text, text


class TestLookingDoesNotDeadEnd:
    """Measured in flight: the drone reached its vantage point, 0.7 m from the
    target, and stopped there for good - status stuck on LOOKING, no goal,
    nothing commanding it.

    _maybe_return_home declined to go home because there was still somewhere
    worth looking, which was right, but then did nothing. The plan was already
    flown out, so nothing moved the drone again. The map review only replans
    when reachability CHANGES, and a look that resolved nothing changes
    nothing, so it never fired either.
    """

    def box(self):
        return [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]

    def arrived_at_a_vantage(self, planning_node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        fill_rect(grid, 0, 60, 30, 60, -1)      # top half never observed
        node = planning_node(self.box(), grid)
        node.plan_sweep()
        assert node.vantage_index is not None, 'test needs a vantage point'
        node.coverage[:, :] = 100               # the legs have been flown
        node.index = len(node.points)           # and the plan is flown out
        node.at = node.points[node.vantage_index]   # parked at the vantage
        return node

    def test_plans_the_next_pass_instead_of_stopping(self, planning_node):
        node = self.arrived_at_a_vantage(planning_node)
        node._maybe_return_home()
        assert not node.returning, 'went home with ground left to look at'
        assert not node.planned, 'nothing will plan again - the drone is stuck'
        assert node.index == 0
        assert node.replanning

    def test_and_the_next_pass_is_flyable(self, planning_node):
        node = self.arrived_at_a_vantage(planning_node)
        node._maybe_return_home()
        assert node.plan_sweep() is True
        assert node.points, 'the next pass had nothing to fly'

    def test_it_still_goes_home_once_there_is_nothing_to_look_at(
            self, planning_node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        node = planning_node(self.box(), grid)
        node.plan_sweep()
        node.coverage[:, :] = 100
        node.at = (18.0, 18.0)
        node.index = len(node.points)
        node._maybe_return_home()
        assert node.returning
        assert node.points == [(0.0, 0.0)]


    def test_a_vantage_near_base_does_not_end_the_search(self, planning_node):
        """The "already home" check must not pre-empt the "still somewhere to
        look" one, or a pass ending near base stops with ground unresolved."""
        node = self.arrived_at_a_vantage(planning_node)
        node.at = (0.5, 0.5)                    # happens to be near base
        node._maybe_return_home()
        assert not node.returning
        assert not node.planned                 # a new pass is queued instead


class TestTinyRemaindersAreNotChased:
    """Seen at the end of a real run: 1 m2 left unverified, and the drone sat
    still for several seconds before heading home.

    It was willing to fly a whole vantage trip to resolve a square metre -
    when the camera covers about 98 m2 in one go - and with a patch that small
    the answer flickered as the costmap updated, so it lapped between "there
    is something left to look at" and "no there isn't", commanding nothing in
    between.
    """

    @pytest.fixture
    def node(self, planning_node):
        import numpy as np
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        n = planning_node([(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
                          grid)
        n.get_parameter.values['min_unverified_area'] = 2.0
        n.possible = np.ones((60, 60), dtype=bool)
        n.confirmed = np.ones((60, 60), dtype=bool)
        return n

    def unverified_cells(self, node, n):
        """Leave exactly n coverage cells unverified."""
        import numpy as np
        node.confirmed = np.ones((60, 60), dtype=bool)
        # each coverage cell is 0.5 m, the costmap 0.5 m here, so 1:1
        rows, cols = np.nonzero(node.inside_mask)
        for r, c in zip(rows[:n], cols[:n]):
            x = node.cov_origin_x + (c + 0.5) * node.coverage_resolution
            y = node.cov_origin_y + (r + 0.5) * node.coverage_resolution
            cc = int((x + 5.0) / 0.5)
            rr = int((y + 5.0) / 0.5)
            node.confirmed[rr, cc] = False

    def test_a_square_metre_is_not_worth_a_trip(self, node):
        self.unverified_cells(node, 4)          # 4 cells x 0.25 = 1 m2
        assert node._unverified_cells() == 4
        assert not node._worth_looking()

    def test_a_real_gap_still_is(self, node):
        self.unverified_cells(node, 40)         # 10 m2
        assert node._worth_looking()

    def test_nothing_outstanding_is_never_worth_a_trip(self, node):
        """area >= 0.0 was true for zero area, so a threshold of 0 meant
        "chase nothing at all" instead of "chase anything"."""
        node.get_parameter.values['min_unverified_area'] = 0.0
        assert node._unverified_cells() == 0
        assert not node._worth_looking()

    def test_the_remainder_is_reported_and_written_off(self, node):
        """It must not vanish silently - the operator is told the area and it
        stops dragging the percentage down."""
        self.unverified_cells(node, 4)
        node.replanning = True
        node.coverage[:, :] = 100
        node.points = []
        node._nothing_to_sweep()
        assert any('never verified' in m for m in node.logged), node.logged[-3:]
        assert node.unverifiable is not None


class TestPathIsOnlyTheSweep:
    """A waypoint 33 m outside the search area, drawn in the same green
    dashes as the lawnmower, read as a planning fault. It was the vantage
    point: a go-and-look trip, published as the last pose of search_path.

    search_path is the sweep legs and nothing else. The trip goes out on its
    own topic, and the ground station draws it as its own amber leg while the
    status says LOOKING.
    """

    def box(self):
        return [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]

    def published(self, node):
        node._publish_path()
        return [(round(p.pose.position.x, 2), round(p.pose.position.y, 2))
                for p in node.sent[-1].poses]

    @pytest.fixture
    def node(self, planning_node, search_node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        n = planning_node(self.box(), grid)
        n.sent = []
        n.path_pub = SimpleNamespace(publish=n.sent.append)
        # the shared fixture stubs _publish_path out; this class is testing it
        n._publish_path = search_node.SearchNode._publish_path.__get__(n)
        return n

    def test_the_vantage_point_is_not_in_the_path(self, node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        fill_rect(grid, 0, 60, 30, 60, -1)       # top half never observed
        node.costmap = grid
        node.inside_mask = node._build_inside_mask()
        node.cells_to_search = int(node.inside_mask.sum())
        assert node.plan_sweep() is True
        assert node.vantage_index is not None, 'test needs a vantage point'
        vantage = node.points[node.vantage_index]
        assert vantage not in self.published(node), \
            'the go-and-look trip is being drawn as a sweep leg'

    def test_it_goes_out_on_its_own_topic(self, node):
        grid = make_costmap(60, 60, resolution=0.5, origin=(-5.0, -5.0))
        fill_rect(grid, 0, 60, 30, 60, -1)
        node.costmap = grid
        node.inside_mask = node._build_inside_mask()
        node.cells_to_search = int(node.inside_mask.sum())
        node.plan_sweep()
        assert node.vantages, 'nothing published on search_vantage'
        pt = node.vantages[-1].point
        assert (pt.x, pt.y) == node.points[node.vantage_index]

    def test_the_way_home_is_not_a_sweep_leg_either(self, node):
        node.plan_sweep()
        assert node.home_index is not None
        assert (0.0, 0.0) not in self.published(node)

    def test_the_sweep_legs_themselves_are_published(self, node):
        node.plan_sweep()
        legs = self.published(node)
        assert legs, 'no sweep legs published at all'
        assert len(legs) == node.home_index


class TestObstacleShadow:
    """Which unknown ground sits behind the mapped face of an obstacle.

    Used to ORDER the targets a look-around trip aims at - behind-the-face
    ground last - and for nothing else. It must not be used to exclude that
    ground or to colour it "cannot check": measured on a live costmap, the
    test caught only 32% of an actual rock interior while condemning 86% of
    the genuinely open ground north of the ridge. From one side the inside of
    a rock and open ground beyond it are the same thing to the map, so no
    local test can separate them - the information is not there.

    It was both. The optimistic flood counts unknown as free, so until a rock
    is mapped all the way round that flood gets into its interior round the
    ends of the mapped face. The nearest unverified cell was then an interior
    cell, the vantage point landed on the near face where nothing can be seen,
    and the drone hopped along the face until the patience counter stopped it.
    Measured on a live costmap: 44% of the cells being targeted sat behind
    mapped rock.

    Depth into unknown does NOT separate the two cases - shadowed and
    unshadowed cells had almost the same median depth, 2.9 m against 2.6 m.
    What separates them is whether a straight line from the nearest known-free
    ground reaches the cell without crossing mapped obstacle.
    """

    UNKNOWN = -1

    def rock_with_a_hollow(self):
        """A mapped wall across the middle; unknown behind it, and unknown
        off to the side that nothing blocks."""
        grid = make_costmap(40, 40, resolution=1.0)
        fill_rect(grid, 0, 40, 20, 40, self.UNKNOWN)   # everything above y=20
        fill_rect(grid, 10, 30, 19, 20, LETHAL)        # a wall at y=19, x 10..30
        return grid

    @pytest.fixture
    def node(self, search_node):
        SearchNode = search_node.SearchNode

        class N:
            for _n in ('_is_blocked', '_costmap_grid', '_flood', '_reachable_from',
                       '_confirmed_from', '_is_reachable', '_polygon_mask_costmap',
                       '_line_of_sight', '_lines_clear', '_shadowed_mask',
                       '_in_obstacle_shadow'):
                locals()[_n] = getattr(SearchNode, _n)
            del _n

            def __init__(self):
                self.costmap = None
                self.confirmed = self.possible = None
                self._shadow_cache = self._shadow_cache_for = None
                self.polygon = [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)]
                self.get_parameter = FakeParams(
                    obstacle_cost_threshold=INSCRIBED,
                    observation_gap_fill=0.0)

            def flood(self, at):
                self.confirmed = self._confirmed_from(at)
                self.possible = self._reachable_from(at)
        return N()

    def test_ground_behind_the_wall_is_shadowed(self, node):
        node.costmap = self.rock_with_a_hollow()
        node.flood((20.5, 5.5))
        grid, info = node._costmap_grid()
        solid = grid >= INSCRIBED
        # directly behind the wall
        assert node._in_obstacle_shadow(20.5, 25.5, solid, info)

    def test_ground_merely_unvisited_is_not_shadowed(self, node):
        """Unknown off to the side, with nothing mapped in between. It is just
        as deep in unknown, and must still be chased."""
        node.costmap = self.rock_with_a_hollow()
        node.flood((20.5, 5.5))
        grid, info = node._costmap_grid()
        solid = grid >= INSCRIBED
        assert not node._in_obstacle_shadow(3.5, 25.5, solid, info)

    def test_depth_alone_would_not_separate_them(self, node):
        """The point of the test: both sample points sit the same distance
        into unknown, and only one of them is behind something."""
        import numpy as np
        from scipy import ndimage
        node.costmap = self.rock_with_a_hollow()
        node.flood((20.5, 5.5))
        d = ndimage.distance_transform_edt(~node.confirmed)
        behind = d[25, 20]
        beside = d[25, 3]
        assert abs(behind - beside) < 4.0, (behind, beside)

    def test_no_floods_means_no_claim(self, node):
        node.costmap = self.rock_with_a_hollow()
        assert node._shadowed_mask() is None

    def test_the_mask_is_cached_per_costmap(self, node):
        node.costmap = self.rock_with_a_hollow()
        node.flood((20.5, 5.5))
        first = node._shadowed_mask()
        assert node._shadowed_mask() is first      # same object, not recomputed
        node.costmap = self.rock_with_a_hollow()   # a new message
        assert node._shadowed_mask() is not first


class TestShadowOnlyOrdersTargets:
    """The shadow test must never write ground off.

    Measured on a live costmap: used as an exclusion it condemned 86% of the
    open ground north of the ridge - exactly the ground the look-around trips
    exist to recover - while catching only 32% of the rock interior it was
    aimed at. It survives as a preference and nothing more.
    """

    UNKNOWN = -1

    def test_shadowed_ground_is_still_a_valid_target(self, fake_frontier):
        """A search area lying ENTIRELY behind the mapped face, with a way
        round at one end. Every target in it is shadowed, so if shadowing
        excluded targets there would be nothing to aim at and the ground
        would be abandoned - when the drone can in fact reach it by going
        round, which is the whole point of a look-around trip.
        """
        grid = make_costmap(30, 30, resolution=1.0)
        fill_rect(grid, 0, 30, 15, 30, self.UNKNOWN)
        fill_rect(grid, 10, 30, 14, 15, LETHAL)   # wall, open at x < 10
        fake_frontier.costmap = grid
        fake_frontier.polygon = [(12.0, 16.0), (28.0, 16.0),
                                 (28.0, 28.0), (12.0, 28.0)]
        fake_frontier.flood()
        g2, info = fake_frontier._costmap_grid()
        solid = g2 >= INSCRIBED
        assert fake_frontier._in_obstacle_shadow(20.5, 20.5, solid, info), \
            'test needs the area to be behind the mapped face'
        assert fake_frontier._frontier_goal() is not None, \
            'ground behind the face was written off, not deprioritised'

    def test_unshadowed_ground_is_preferred(self, fake_frontier):
        """With a gap at one end, the reachable side should be aimed at
        first - that is what cut the hopping along the rock face."""
        grid = make_costmap(30, 30, resolution=1.0)
        fill_rect(grid, 0, 30, 15, 30, self.UNKNOWN)
        fill_rect(grid, 10, 30, 14, 15, LETHAL)      # wall, open at x < 10
        fake_frontier.costmap = grid
        fake_frontier.polygon = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
        fake_frontier.flood()
        goal = fake_frontier._frontier_goal()
        assert goal is not None
        # the open end is at low x; the chosen vantage should be over that side
        assert goal[0] < 15.0, f'aimed at the blocked side first: {goal}'


class TestNewAreaIsJudgedAfresh:
    """"SEARCH AREA NOT SEARCHABLE" for an area that was perfectly reachable.

    _build_inside_mask filters by self.possible, and on_area_request rebuilt
    the coverage grid BEFORE anything recomputed it - so a new box was judged
    against the reachability picture of the previous one, taken at a different
    position against an older map. A box outside that picture came out with
    zero searchable cells, the node announced the area unreachable and set
    planned = True, and never looked again. Measured on the live costmap at
    the time: the box was 100% unknown, 0% blocked, and a fresh flood reached
    all 960 of its cells.
    """

    def test_a_new_area_is_not_judged_by_the_old_one(self, planning_node,
                                                     search_node):
        import numpy as np
        grid = make_costmap(80, 80, resolution=0.5, origin=(-5.0, -5.0))
        node = planning_node([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                             grid)
        # a stale picture that covers nothing at all, as a previous area
        # at a previous position could easily leave behind
        node.possible = np.zeros((80, 80), dtype=bool)
        node.confirmed = np.zeros((80, 80), dtype=bool)

        # the operator draws somewhere new
        node.polygon = [(20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)]
        node.confirmed = None          # what on_area_request now does
        node.possible = None
        node._shadow_cache = node._shadow_cache_for = None
        node._rebuild_coverage_grid = search_node.SearchNode._rebuild_coverage_grid.__get__(node)
        node._rebuild_coverage_grid()

        assert node.cells_to_search > 0, \
            'the new area was written off using the old reachability picture'

    def test_a_stale_picture_would_have_written_it_off(self, planning_node):
        """The counterpart: with the stale picture left in place, the mask is
        empty - which is the bug, pinned down so it cannot come back."""
        import numpy as np
        grid = make_costmap(80, 80, resolution=0.5, origin=(-5.0, -5.0))
        node = planning_node([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                             grid)
        node.possible = np.zeros((80, 80), dtype=bool)
        node.polygon = [(20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)]
        node.inside_mask = node._build_inside_mask()
        assert int(node.inside_mask.sum()) == 0
