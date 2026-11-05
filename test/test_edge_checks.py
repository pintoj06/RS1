"""A person cut off by the camera's edge, on a boundary leg of the search.

The detector only counts a person fully in frame. Mid-area that is fine,
the next leg overlaps and sees them whole, but on a boundary leg there is
no next leg, so someone just past the edge would be passed by for good. The
scout flies over them, hovers for a look, and goes back to where it left
the sweep.
"""

import math
from types import SimpleNamespace

import pytest
from action_msgs.msg import GoalStatus
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from conftest import FakeLogger, FakeParams, make_costmap, set_cell

WIDTH, HEIGHT = 720, 480


class FakeGoalHandle:
    def __init__(self):
        self.cancelled = False

    def cancel_goal_async(self):
        self.cancelled = True


def result(status):
    return SimpleNamespace(result=lambda: SimpleNamespace(status=status))


def edge_box(u=700.0, v=240.0, score=0.8):
    """One detection centred at pixel (u, v); the default is the right edge."""
    det = Detection2D()
    det.bbox.center.position.x = u
    det.bbox.center.position.y = v
    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.score = score
    det.results.append(hyp)
    msg = Detection2DArray()
    msg.detections.append(det)
    return msg


@pytest.fixture
def scout(search_node):
    """A SearchNode flying a sweep leg along the bottom edge of a 40 m box."""
    SearchNode = search_node.SearchNode

    class Scout:
        for _name in ('_on_edge_detections', '_edge_check_allowed',
                      '_edge_target', '_start_edge_check', 'next_waypoint',
                      'goal_finished', 'tick', 'paused', '_camera_fov',
                      '_camera_altitude', '_footprint_radius', '_is_blocked',
                      '_start_pass'):
            locals()[_name] = getattr(SearchNode, _name)
        del _name

        def __init__(self):
            self.polygon = [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)]
            self.area_set = True
            self.holding = self.estopped = False
            self.pending_area = None
            self.goal_handle = FakeGoalHandle()
            self.cam_fx = self.cam_fy = None
            self.cam_width, self.cam_height = WIDTH, HEIGHT
            self.costmap = None
            self.edge_checked = []
            self.t = 0.0
            self.at = (20.0, 2.0, 0.0)        # x, y, yaw: facing +x, 2 m in
            self.sent = []
            self.statuses = []
            self.get_logger = FakeLogger
            self.get_parameter = FakeParams(
                investigate_edges=True, edge_band_margin=1.0,
                edge_max_detour=10.0, edge_dwell_s=4.0, edge_spot_radius=3.0,
                edge_stop_short=2.5,
                avoid_obstacles=True, obstacle_cost_threshold=99,
                fallback_camera_hfov=1.0472, fallback_image_width=WIDTH,
                fallback_image_height=HEIGHT, flight_altitude=10.0)
            self.sweep_complete_pub = SimpleNamespace(publish=lambda _m: None)
            self.nav_client = SimpleNamespace(server_is_ready=lambda: True)
            self._start_pass()
            # Leg along y = 2 heading for (38, 2), then the far side; home last.
            self.points = [(2.0, 2.0), (38.0, 2.0), (38.0, 9.0), (0.0, 0.0)]
            self.index = 1
            self.home_index = 3
            self.planned = True
            self.busy = True

        def _now(self):
            return self.t

        def _pose_at(self, _stamp):
            return self.at

        def _lookup_robot_xy(self):
            return self.at[:2]

        def _publish_path(self):
            pass

        def _set_status(self, state, detail=''):
            self.statuses.append(state)

        def _check_stalled(self):
            pass

        def send_goal(self, point):
            self.sent.append(point)
            self.busy = True

    return Scout()


class TestGeometry:
    def test_image_centre_is_straight_below(self, search_node):
        assert search_node.pixel_to_ground(
            360, 240, WIDTH, HEIGHT, 1.0472, 10.0, 3.0, 4.0, 0.0) == (3.0, 4.0)

    def test_right_edge_is_to_the_drones_right(self, search_node):
        # Facing +x, the right of the image is -y. Half width at 10 m with a
        # 60 deg lens is 10 tan(30 deg) = 5.77 m.
        x, y = search_node.pixel_to_ground(
            720, 240, WIDTH, HEIGHT, 1.0472, 10.0, 0.0, 0.0, 0.0)
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(-10.0 * math.tan(0.5236), abs=1e-3)

    def test_distance_to_the_outline_inside_and_out(self, search_node):
        square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        d = search_node.distance_to_polygon_edge
        assert d(5.0, 1.0, square) == pytest.approx(1.0)
        assert d(5.0, -2.0, square) == pytest.approx(2.0)
        assert d(13.0, 14.0, square) == pytest.approx(5.0)   # off a corner


class TestStartsACheck:
    def test_steps_off_the_sweep_and_back(self, scout):
        scout._on_edge_detections(edge_box())
        look, back = scout.points[1], scout.points[2]
        # The person is past the right of the frame, about 5.5 m to -y and
        # outside the area, where no later leg would see them. The drone
        # stops 2.5 m short: enough to have them well inside both cameras.
        assert look[0] == pytest.approx(20.0, abs=0.1)
        assert look[1] == pytest.approx(-0.95, abs=0.1)
        assert back == (20.0, 2.0)
        # Then the waypoint it was flying to, and home still last.
        assert scout.points[3] == (38.0, 2.0)
        assert scout.home_index == 5 and scout.points[5] == (0.0, 0.0)
        assert scout.goal_handle.cancelled

    def test_mid_area_it_carries_on(self, scout):
        """The next leg overlaps this one and will see them whole."""
        scout.at = (20.0, 20.0, 0.0)
        scout._on_edge_detections(edge_box())
        assert len(scout.points) == 4
        assert not scout.goal_handle.cancelled

    def test_same_spot_is_only_checked_once(self, scout):
        scout._on_edge_detections(edge_box())
        scout._start_pass()                       # e.g. a replan since
        scout.points, scout.index, scout.planned = [(38.0, 2.0)], 0, True
        scout.goal_handle = FakeGoalHandle()
        scout._on_edge_detections(edge_box(u=705.0))
        assert scout.points == [(38.0, 2.0)]

    def test_not_into_a_mapped_obstacle(self, scout):
        grid = make_costmap(60, 60, resolution=1.0, origin=(-10.0, -10.0))
        set_cell(grid, 30, 9, 100)                # (20.5, -0.5): the cell the look spot is in
        scout.costmap = grid
        scout._on_edge_detections(edge_box())
        assert len(scout.points) == 4

    @pytest.mark.parametrize('state', ['holding', 'returning', 'replanning'])
    def test_not_while_doing_something_else(self, scout, state):
        setattr(scout, state, True)
        scout._on_edge_detections(edge_box())
        assert len(scout.points) == 4

    def test_not_while_a_goal_is_still_being_accepted(self, scout):
        """No handle to cancel yet: the old waypoint would be flown and then
        the index advanced past the check."""
        scout.goal_handle = None
        scout._on_edge_detections(edge_box())
        assert len(scout.points) == 4

    def test_already_close_means_look_from_here(self, scout):
        """Inside edge_stop_short already: hover where it is, do not lunge."""
        scout.get_parameter.values['edge_stop_short'] = 10.0
        scout._on_edge_detections(edge_box())
        assert scout.points[1] == (20.0, 2.0)

    def test_one_check_at_a_time(self, scout):
        scout._on_edge_detections(edge_box())
        scout._on_edge_detections(edge_box(u=360.0, v=5.0))
        assert len(scout.points) == 6


class TestFliesTheCheck:
    def started(self, scout):
        scout._on_edge_detections(edge_box())
        scout.goal_finished(result(GoalStatus.STATUS_CANCELED))
        return scout

    def test_a_goal_that_lands_as_it_is_cancelled_does_not_skip_it(self, scout):
        scout._on_edge_detections(edge_box())
        scout.goal_finished(result(GoalStatus.STATUS_SUCCEEDED))
        assert scout.index == 1 and not scout.busy

    def test_whole_round_trip(self, scout):
        scout = self.started(scout)
        scout.tick()
        assert scout.sent[-1] == scout.points[1]          # over the person
        scout.goal_finished(result(GoalStatus.STATUS_SUCCEEDED))

        scout.t = 2.0                                     # hovering
        scout.tick()
        assert len(scout.sent) == 1, 'left before the cameras had a look'
        assert 'INVESTIGATING' in scout.statuses

        scout.t = 4.5
        scout.tick()
        assert scout.sent[-1] == (20.0, 2.0)              # back where it left
        scout.goal_finished(result(GoalStatus.STATUS_SUCCEEDED))
        assert scout.edge_look_index is None

        scout.tick()
        assert scout.sent[-1] == (38.0, 2.0)              # and on with the leg

    def test_a_hold_over_the_person_does_not_cut_the_look_short(self, scout):
        """Thermal sees heat and the ground station holds the search. Nothing
        moves until the operator rules, then the drone goes back."""
        scout = self.started(scout)
        scout.tick()
        scout.goal_finished(result(GoalStatus.STATUS_SUCCEEDED))
        scout.holding = True
        scout.t = 60.0
        scout.tick()
        assert len(scout.sent) == 1
        scout.holding = False
        scout.tick()
        assert scout.sent[-1] == (20.0, 2.0)
