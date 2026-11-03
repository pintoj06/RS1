"""The ground station's E-Stop: every drone stops where it is and only the
E-Stop's own RESUME restarts it, never a contact hold being released."""

from types import SimpleNamespace

import pytest
from action_msgs.msg import GoalStatus
from std_msgs.msg import Bool

from conftest import FakeLogger, load_module


class FakeGoalHandle:
    def __init__(self):
        self.cancelled = False

    def cancel_goal_async(self):
        self.cancelled = True


def result(status):
    return SimpleNamespace(result=lambda: SimpleNamespace(status=status))


@pytest.fixture
def takeoff_node():
    return load_module('takeoff_node', 'scripts/takeoff_node.py')


@pytest.fixture
def search(search_node):
    SearchNode = search_node.SearchNode

    class FakeSearch:
        on_estop = SearchNode.on_estop
        on_hold = SearchNode.on_hold
        paused = SearchNode.paused
        goal_finished = SearchNode.goal_finished
        goal_response = SearchNode.goal_response

        def __init__(self):
            self.holding = False
            self.estopped = False
            self.goal_handle = None
            self.index = 2
            self.points = [(0, 0)] * 6
            self.busy = False
            self.timed_out = False
            self.rejections = 0
            self.ticks = 0
            self.advanced = 0
            self.get_logger = FakeLogger

        def tick(self):
            # Only counts calls; the real tick returns early when paused().
            self.ticks += 1

        def next_waypoint(self):
            self.advanced += 1

    return FakeSearch()


class TestSearchNode:
    def test_estop_cancels_the_current_waypoint(self, search):
        search.goal_handle = FakeGoalHandle()
        search.on_estop(Bool(data=True))
        assert search.estopped and search.paused()
        assert search.goal_handle.cancelled

    def test_cancelled_waypoint_is_flown_again_not_skipped(self, search):
        search.busy = True
        search.on_estop(Bool(data=True))
        search.goal_finished(result(GoalStatus.STATUS_CANCELED))
        assert search.advanced == 0
        assert search.index == 2
        assert not search.busy

    def test_goal_accepted_during_estop_is_cancelled(self, search):
        search.on_estop(Bool(data=True))
        handle = FakeGoalHandle()
        handle.accepted = True
        handle.get_result_async = lambda: SimpleNamespace(add_done_callback=lambda cb: None)
        search.goal_response(SimpleNamespace(result=lambda: handle))
        assert handle.cancelled

    def test_releasing_a_hold_does_not_release_the_estop(self, search):
        search.on_hold(Bool(data=True))
        search.on_estop(Bool(data=True))
        search.on_hold(Bool(data=False))
        assert search.paused()

    def test_resume_clears_the_estop_but_not_a_contact_hold(self, search):
        search.on_hold(Bool(data=True))
        search.on_estop(Bool(data=True))
        search.on_estop(Bool(data=False))
        assert not search.estopped
        assert search.paused()            # still holding for the contact
        search.on_hold(Bool(data=False))
        assert not search.paused()

    def test_resume_restarts_straight_away(self, search):
        search.on_estop(Bool(data=True))
        before = search.ticks
        search.on_estop(Bool(data=False))
        assert search.ticks == before + 1

    def test_tick_does_nothing_while_estopped(self, search_node, search):
        # The real tick, with everything else ready to fly.
        search.area_set, search.pending_area, search.planned = True, None, True
        search.nav_client = SimpleNamespace(server_is_ready=lambda: True)
        sent = []
        search.send_goal = sent.append
        search.on_estop(Bool(data=True))
        search_node.SearchNode.tick(search)
        assert sent == []
        search.estopped = False
        search_node.SearchNode.tick(search)
        assert sent == [(0, 0)]


@pytest.fixture
def courier(courier_node):
    CourierNode = courier_node.CourierNode

    class FakeCourier:
        on_estop = CourierNode.on_estop
        goal_finished = CourierNode.goal_finished

        def __init__(self):
            self.estopped = False
            self.goal_handle = FakeGoalHandle()
            self.busy = True
            self.sent = True
            self.target = (5.0, 5.0)
            self.released = 0
            self.release_pub = SimpleNamespace(publish=lambda _: self._release())
            self.get_logger = FakeLogger

        def _release(self):
            self.released += 1

        def tick(self):
            CourierNode.tick(self)

    return FakeCourier()


class TestCourier:
    def test_estop_cancels_the_delivery_and_keeps_the_package(self, courier):
        courier.on_estop(Bool(data=True))
        assert courier.goal_handle.cancelled
        courier.goal_finished(result(GoalStatus.STATUS_CANCELED))
        assert courier.released == 0
        assert not courier.busy
        assert not courier.sent           # so it is flown again on RESUME

    def test_resume_flies_the_delivery_again(self, courier):
        courier.on_estop(Bool(data=True))
        courier.goal_finished(result(GoalStatus.STATUS_CANCELED))
        sent = []
        courier.nav_client = SimpleNamespace(server_is_ready=lambda: True)
        courier.send_goal = sent.append
        courier.on_estop(Bool(data=False))
        assert sent == [(5.0, 5.0)]

    def test_normal_arrival_still_releases(self, courier):
        courier.goal_finished(result(GoalStatus.STATUS_SUCCEEDED))
        assert courier.released == 1


@pytest.fixture
def takeoff(takeoff_node):
    TakeoffNode = takeoff_node.TakeoffNode
    sent = []
    node = SimpleNamespace(
        estopped=False, done=False, launched=True, altitude=2.0,
        target=10.0, tolerance=0.05, timeout=60.0, max_speed=1.5, gain=1.5,
        start_time=None, sent=sent,
        cmd_pub=SimpleNamespace(publish=sent.append),
        get_logger=FakeLogger,
        get_clock=lambda: SimpleNamespace(now=lambda: _Time(0)),
    )
    node.on_estop = lambda msg: TakeoffNode.on_estop(node, msg)
    node.tick = lambda: TakeoffNode.tick(node)
    return node


class _Time:
    def __init__(self, ns):
        self.nanoseconds = ns

    def __sub__(self, other):
        return _Time(self.nanoseconds - other.nanoseconds)


class TestTakeoff:
    def test_estop_sends_zero_velocity_and_stops_climbing(self, takeoff):
        takeoff.tick()
        assert takeoff.sent[-1].linear.z > 0
        takeoff.on_estop(Bool(data=True))
        assert takeoff.sent[-1].linear.z == 0.0
        count = len(takeoff.sent)
        takeoff.tick()
        assert len(takeoff.sent) == count     # nothing sent while stopped

    def test_resume_carries_on_climbing(self, takeoff):
        takeoff.on_estop(Bool(data=True))
        takeoff.on_estop(Bool(data=False))
        assert takeoff.start_time is None     # timeout restarts
        takeoff.tick()
        assert takeoff.sent[-1].linear.z > 0


class TestGroundStation:
    def test_set_estop_publishes_only_on_change(self, ground_station):
        published = []
        station = SimpleNamespace(
            estopped=False,
            estop_pub=SimpleNamespace(publish=lambda m: published.append(m.data)),
            get_logger=FakeLogger,
        )
        set_estop = ground_station.GroundStationNode.set_estop
        set_estop(station, True)
        set_estop(station, True)
        set_estop(station, False)
        assert published == [True, False]
        assert station.estopped is False
