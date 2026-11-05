"""After the drop both drones fly back, one after the other: the courier first
while the scout keeps watch, then the scout once the courier reports in.

Uses the real nodes where it can. Nothing else in this suite constructed a real
CourierNode, and a QoS mismatch on the new topic would fail in flight with no
error at all, so one test sends the message over a real topic.
"""

import time
from types import SimpleNamespace

import pytest
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point32, PointStamped, PolygonStamped
from std_msgs.msg import Bool, String

from conftest import make_costmap

HOME = (2.47, -1.10)
CONTACT = (10.0, 5.0)


class FakeGoalHandle:
    def __init__(self):
        self.cancelled = False

    def cancel_goal_async(self):
        self.cancelled = True


def result(status):
    return SimpleNamespace(result=lambda: SimpleNamespace(status=status))


@pytest.fixture
def real_nodes():
    if not rclpy.ok():
        rclpy.init()
    made = []
    yield made
    for node in made:
        node.destroy_node()


@pytest.fixture
def courier(courier_node, real_nodes):
    """A real CourierNode with Nav2, TF and the drop stubbed out."""
    node = courier_node.CourierNode()
    real_nodes.append(node)
    node.goals, node.states, node.drops = [], [], []
    node.at = HOME
    node.own_xy = lambda: node.at
    node.nav_client = SimpleNamespace(server_is_ready=lambda: True)

    def send_goal(point):
        node.busy = node.sent = True
        node.goals.append(point)
    node.send_goal = send_goal
    node.publish_courier_state = node.states.append
    node.release_pub = SimpleNamespace(publish=node.drops.append)
    node.tick()                     # on the pad: records home
    return node


def deliver(courier):
    """Fly to the contact and drop the package, up to the cancel landing."""
    courier.target = CONTACT
    courier.tick()
    assert courier.goals == [CONTACT]
    courier.goal_handle = FakeGoalHandle()
    courier.at = (9.5, 5.0)
    courier.check_release()
    assert courier.released and courier.goal_handle.cancelled
    courier.goal_finished(result(GoalStatus.STATUS_CANCELED))


class TestCourierFliesBack:
    def test_home_is_where_it_was_before_takeoff(self, courier):
        courier.at = (30.0, 30.0)
        courier.tick()
        assert courier.home == HOME

    def test_stays_put_until_the_package_is_dropped(self, courier):
        courier.tick()
        assert courier.goals == []
        assert courier.states == []

    def test_flies_home_straight_after_the_drop(self, courier):
        deliver(courier)
        assert courier.goals == [CONTACT, HOME]
        assert courier.states == ['RETURNING']
        assert len(courier.drops) == 1

    def test_arriving_on_the_contact_also_starts_the_return(self, courier):
        courier.target = CONTACT
        courier.tick()
        courier.goal_finished(result(GoalStatus.STATUS_SUCCEEDED))
        assert len(courier.drops) == 1
        assert courier.goals == [CONTACT, HOME]

    def test_reports_home_on_arrival_and_stops(self, courier):
        deliver(courier)
        courier.goal_finished(result(GoalStatus.STATUS_SUCCEEDED))
        assert courier.states == ['RETURNING', 'HOME']
        courier.tick()
        assert courier.goals == [CONTACT, HOME]        # not sent again
        assert len(courier.drops) == 1

    def test_reports_failure_so_the_scout_is_not_left_waiting(self, courier):
        deliver(courier)
        courier.goal_finished(result(GoalStatus.STATUS_ABORTED))
        assert courier.states == ['RETURNING', 'RETURN_FAILED']

    def test_no_home_recorded_counts_as_failed(self, courier):
        courier.home = None
        courier.own_xy = lambda: None
        courier.released = True
        courier.target = CONTACT
        courier.tick()
        assert courier.states == ['RETURN_FAILED']

    def test_can_be_switched_off(self, courier):
        courier.set_parameters([rclpy.parameter.Parameter(
            'return_after_delivery', value=False)])
        deliver(courier)
        assert courier.goals == [CONTACT]
        assert courier.states == []


class TestCourierEstopOnTheWayBack:
    @pytest.mark.parametrize('resume_before_result', [False, True])
    def test_return_is_flown_again(self, courier, resume_before_result):
        deliver(courier)
        courier.goal_handle = FakeGoalHandle()
        courier.on_estop(Bool(data=True))
        assert courier.goal_handle.cancelled
        cancelled = result(GoalStatus.STATUS_CANCELED)
        if resume_before_result:
            courier.on_estop(Bool(data=False))
            courier.goal_finished(cancelled)
            courier.tick()
        else:
            courier.goal_finished(cancelled)
            assert courier.goals == [CONTACT, HOME]    # still stopped
            courier.on_estop(Bool(data=False))
        assert courier.goals == [CONTACT, HOME, HOME]
        assert 'HOME' not in courier.states
        assert 'RETURN_FAILED' not in courier.states


class TestDeliveryEstopRace:
    def test_resume_before_the_cancel_lands_still_delivers(self, courier):
        """RESUME clears the flag before Nav2 reports the cancel. This used to
        read as a failed delivery, and the courier never flew again."""
        courier.target = CONTACT
        courier.tick()
        courier.goal_handle = FakeGoalHandle()
        courier.on_estop(Bool(data=True))
        courier.on_estop(Bool(data=False))
        courier.goal_finished(result(GoalStatus.STATUS_CANCELED))
        courier.tick()
        assert courier.goals == [CONTACT, CONTACT]
        assert courier.drops == []


@pytest.fixture
def scout(search_node, real_nodes):
    """A real SearchNode part way through a sweep, with Nav2 stubbed out."""
    node = search_node.SearchNode()
    real_nodes.append(node)
    node.cam_fx = node.cam_fy = 360.0
    node.cam_width, node.cam_height = 720, 480
    node.polygon = [(0.0, 2.0), (16.0, 2.0), (16.0, 18.0), (0.0, 18.0)]
    node.area_set = True
    node._rebuild_coverage_grid()
    node._lookup_robot_xy = lambda: (8.0, 8.0)
    node.costmap = make_costmap(300, 300, resolution=0.2, origin=(-20.0, -20.0))
    assert node.plan_sweep() is True
    node.nav_client = SimpleNamespace(server_is_ready=lambda: True)
    node.goals = []

    def send_goal(point):
        node.busy = True
        node.goals.append(point)
    node.send_goal = send_goal
    return node


class TestScoutFliesBackSecond:
    @pytest.mark.parametrize('state', ['HOME', 'RETURN_FAILED'])
    def test_goes_home_once_the_courier_is_done(self, scout, state):
        scout.on_courier_state(String(data=state))
        scout.tick()
        scout.tick()
        assert scout.returning
        assert scout.goals == [(0.0, 0.0)]

    def test_keeps_searching_while_the_courier_is_on_its_way_back(self, scout):
        scout.on_courier_state(String(data='RETURNING'))
        scout.tick()
        assert not scout.returning
        assert scout.goals and scout.goals[0] != (0.0, 0.0)

    def test_cancels_a_sweep_leg_first_and_waits_for_it(self, scout):
        scout.tick()
        scout.goal_handle = FakeGoalHandle()
        scout.on_courier_state(String(data='HOME'))
        scout.tick()
        assert scout.goal_handle.cancelled
        assert not scout.returning            # the old result is still to come
        scout.goal_finished(result(GoalStatus.STATUS_CANCELED))
        scout.tick()
        scout.tick()
        assert scout.goals[-1] == (0.0, 0.0)

    def test_estop_still_wins(self, scout):
        scout.on_estop(Bool(data=True))
        scout.on_courier_state(String(data='HOME'))
        scout.tick()
        assert not scout.returning and scout.goals == []
        scout.on_estop(Bool(data=False))
        scout.tick()
        assert scout.goals == [(0.0, 0.0)]

    def test_a_hold_keeps_it_until_released(self, scout):
        scout.on_hold(Bool(data=True))
        scout.on_courier_state(String(data='HOME'))
        scout.tick()
        assert not scout.returning
        scout.on_hold(Bool(data=False))
        scout.tick()
        assert scout.goals == [(0.0, 0.0)]

    def test_a_restarted_pass_does_not_resume_the_search(self, scout):
        scout.on_courier_state(String(data='HOME'))
        scout.tick()
        scout._start_pass(replanning=True)
        scout.tick()
        assert scout.returning
        assert scout.points == [(0.0, 0.0)]

    def test_map_changes_do_not_replan_on_the_way_home(self, scout):
        scout.on_courier_state(String(data='HOME'))
        scout.tick()
        before = scout.map_replans
        scout.plan_reachable_cells = 0        # looks like a big change
        scout._review_map()
        assert scout.map_replans == before


class TestOverARealTopic:
    def test_courier_state_reaches_the_scout(self, courier_node, search_node,
                                             real_nodes):
        if not rclpy.ok():
            rclpy.init()
        sender = courier_node.CourierNode()
        sender.publish_courier_state('HOME')
        # Subscribes only after that went out: it has to be latched, or a
        # scout node that starts late would never hear it.
        receiver = search_node.SearchNode()
        real_nodes += [sender, receiver]
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(sender)
        executor.add_node(receiver)
        deadline = time.monotonic() + 8.0
        while not receiver.recalled and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        executor.shutdown()
        assert receiver.recalled, 'never arrived: check the QoS on both ends'


class TestGroundStationLetsGo:
    @pytest.fixture
    def window(self, ground_station):
        holds = []
        node = SimpleNamespace(courier_state=None, set_hold=holds.append)
        win = SimpleNamespace(
            node=node, contacts=[], hold_for_confirmed=True, _mission_over=False,
            _check_until=None, _no_check_until=0.0, _area_since_confirm=False)
        cls = ground_station.MainWindow
        win._update_hold = lambda: cls._update_hold(win)
        win.step = lambda: cls._end_mission_if_courier_done(win)
        win.holds = holds
        return win

    @pytest.mark.parametrize('state', ['HOME', 'RETURN_FAILED'])
    def test_releases_the_hold_and_stops_checking(self, window, state):
        window.node.courier_state = state
        window.step()
        assert window.holds == [False]
        assert window._no_check_until == float('inf')

    @pytest.mark.parametrize('state', [None, 'RETURNING'])
    def test_holds_over_the_casualty_until_then(self, window, state):
        window.node.courier_state = state
        window.step()
        assert window.holds == []
        assert window.hold_for_confirmed

    def test_a_contact_awaiting_a_ruling_still_holds(self, window, ground_station):
        window.contacts = [ground_station.Contact(
            id=2, x=0.0, y=0.0, rgb_confidence=0.9, thermal_confidence=0.9)]
        window.node.courier_state = 'HOME'
        window.step()
        assert window.holds == [True]


def transform(x=0.0, y=0.0):
    return SimpleNamespace(transform=SimpleNamespace(
        translation=SimpleNamespace(x=x, y=y)))


def area_msg(x0, y0, x1, y1):
    msg = PolygonStamped()
    msg.header.frame_id = 'parrot1_map'
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        msg.polygon.points.append(Point32(x=float(x), y=float(y), z=0.0))
    return msg


class TestSearchAgain:
    """Draw another area once they are back, and the scout goes out again."""

    @pytest.fixture
    def home_scout(self, scout):
        """Recalled by the courier, on its way home, goal in flight."""
        scout.tf_buffer = SimpleNamespace(
            lookup_transform=lambda *a, **k: transform())
        scout.on_confirmed_point(PointStamped())
        scout.on_courier_state(String(data='HOME'))
        scout.tick()
        assert scout.returning
        return scout

    def land(self, scout):
        scout.goal_finished(result(GoalStatus.STATUS_SUCCEEDED))
        scout.tick()

    def test_a_new_area_sends_it_back_out(self, home_scout):
        self.land(home_scout)
        home_scout.goals.clear()
        home_scout.on_area_request(area_msg(-14, 2, -2, 14))
        assert not home_scout.recalled and not home_scout.returning
        home_scout.tick()
        assert home_scout.goals
        x, y = home_scout.goals[0]
        assert -14.5 <= x <= -1.5 and 1.5 <= y <= 14.5

    def test_the_new_search_starts_from_nothing(self, home_scout):
        home_scout.coverage[:, :] = 100
        home_scout.replans = home_scout.map_replans = 3
        self.land(home_scout)
        home_scout.on_area_request(area_msg(-14, 2, -2, 14))
        assert not home_scout.coverage.any()
        assert home_scout.replans == 0 and home_scout.map_replans == 0

    def test_it_comes_home_after_the_new_search_too(self, home_scout):
        self.land(home_scout)
        home_scout.on_area_request(area_msg(-14, 2, -2, 14))
        home_scout.tick()
        assert home_scout.points[-1] == (0.0, 0.0)

    def test_a_smaller_area_than_the_first(self, home_scout):
        self.land(home_scout)
        home_scout.on_area_request(area_msg(2, 2, 6, 6))
        home_scout.tick()
        assert home_scout.goals

    def test_drawn_while_flying_home_waits_for_the_cancel(self, home_scout):
        """The way home's late result must not be taken for the new goal's."""
        home_scout.goal_handle = old = FakeGoalHandle()
        home_scout.goals.clear()
        home_scout.on_area_request(area_msg(-14, 2, -2, 14))
        assert old.cancelled
        home_scout.tick()
        assert home_scout.goals == []              # still waiting on it
        home_scout.goal_finished(result(GoalStatus.STATUS_CANCELED))
        home_scout.tick()
        assert home_scout.goals and home_scout.index == 0

    def test_drawn_before_the_courier_is_back_wins(self, scout):
        scout.tf_buffer = SimpleNamespace(
            lookup_transform=lambda *a, **k: transform())
        scout.on_confirmed_point(PointStamped())
        scout.on_area_request(area_msg(-14, 2, -2, 14))
        scout.on_courier_state(String(data='HOME'))
        assert not scout.recalled

    def test_an_area_from_before_the_confirm_does_not(self, scout):
        scout.tf_buffer = SimpleNamespace(
            lookup_transform=lambda *a, **k: transform())
        scout.on_area_request(area_msg(-14, 2, -2, 14))
        scout.on_confirmed_point(PointStamped())
        scout.on_courier_state(String(data='HOME'))
        assert scout.recalled


class TestGroundStationSearchAgain:
    @pytest.fixture
    def window(self, ground_station):
        sent, holds = [], []
        node = SimpleNamespace(courier_state='HOME', set_hold=holds.append,
                               publish_search_area=lambda *a: sent.append(a))
        win = SimpleNamespace(
            node=node, contacts=[], hold_for_confirmed=True, _mission_over=False,
            _check_until=None, _no_check_until=0.0, _area_since_confirm=False)
        cls = ground_station.MainWindow
        win._update_hold = lambda: cls._update_hold(win)
        win.finish = lambda: cls._end_mission_if_courier_done(win)
        win.draw = lambda: cls._set_area(win, 0.0, 0.0, 5.0, 5.0)
        win.sent = sent
        return win

    def test_a_new_area_turns_heat_checks_back_on(self, window):
        window.finish()
        assert window._no_check_until == float('inf')
        window.draw()
        assert window._no_check_until == 0.0
        assert window.sent == [(0.0, 0.0, 5.0, 5.0)]

    def test_drawn_while_the_courier_is_out_keeps_them_on(self, window):
        window.draw()
        window.finish()
        assert window._no_check_until == 0.0
        assert not window.hold_for_confirmed

    def test_it_only_lets_go_once(self, window):
        window.finish()
        window.draw()
        window.hold_for_confirmed = True      # a second person confirmed
        window.finish()
        assert window.hold_for_confirmed      # still held for RESUME

    @pytest.mark.parametrize('state, used', [
        (None, False), ('RETURNING', False),
        ('HOME', True), ('RETURN_FAILED', True)])
    def test_the_operator_is_told_the_package_is_gone(
            self, ground_station, state, used):
        node = SimpleNamespace(courier_state=state)
        assert ground_station.GroundStationNode.courier_used(node) is used
