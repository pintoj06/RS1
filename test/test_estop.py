"""The ground station's E-Stop: every drone stops where it is and only the
E-Stop's own RESUME restarts it, never a contact hold being released."""

from types import SimpleNamespace

import pytest
from action_msgs.msg import GoalStatus
from std_msgs.msg import Bool

from conftest import FakeLogger, FakeParams, load_module


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
            self.recalled = False
            self.goal_handle = None
            self.index = 2
            self.points = [(0, 0)] * 6
            self.busy = False
            self._replan_wanted = False
            self.returning = False
            self.vantage_index = None
            self.edge_look_index = None
            self.edge_back_index = None
            self._dwell_until = None
            self._goal_superseded = False
            self.statuses = []
            self.timed_out = False
            self.rejections = 0
            self.ticks = 0
            self.advanced = 0
            self.get_logger = FakeLogger

        def _set_status(self, state, detail=''):
            # tick() reports what the scout is doing; nothing here reads it.
            pass

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
        release = CourierNode.release

        def __init__(self):
            self.estopped = False
            self.goal_handle = FakeGoalHandle()
            self.busy = True
            self.sent = True
            self.target = (5.0, 5.0)
            self.released = False       # the node's own flag
            self.going_home = False
            self.home = (0.0, 0.0)
            self.drops = 0              # how many times the package went
            self.release_pub = SimpleNamespace(publish=lambda _: self._release())
            self.get_logger = FakeLogger

        def _release(self):
            self.drops += 1

        def fly_home(self):
            pass                        # see test_return_to_base.py

        def tick(self):
            CourierNode.tick(self)

    return FakeCourier()


class TestCourier:
    def test_estop_cancels_the_delivery_and_keeps_the_package(self, courier):
        courier.on_estop(Bool(data=True))
        assert courier.goal_handle.cancelled
        courier.goal_finished(result(GoalStatus.STATUS_CANCELED))
        assert courier.drops == 0
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
        assert courier.drops == 1


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


class FakeCancelClient:
    """Stands in for the Nav2 navigate_to_pose cancel_goal service client."""

    def __init__(self, ready=True):
        self.ready = ready
        self.requests = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        return SimpleNamespace()


class FakeCmdPub:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


def make_station(ground_station, nav_ready=True, cancel=True, zero=True):
    """A ground station carrying only the fields E-Stop touches."""
    node = ground_station.GroundStationNode.__new__(
        ground_station.GroundStationNode)
    node.estopped = False
    node.published = []
    node.estop_pub = SimpleNamespace(
        publish=lambda m: node.published.append(m.data))
    node.estop_cancel = {"parrot1": FakeCancelClient(nav_ready),
                         "parrot2": FakeCancelClient(nav_ready)}
    node.estop_cmd = {"parrot1": FakeCmdPub(), "parrot2": FakeCmdPub()}
    node.estop_zero_timer = None
    node.made_timers = []
    node.destroyed = []

    def create_timer(period, callback):
        handle = SimpleNamespace(period=period, callback=callback,
                                 cancelled=False)
        handle.cancel = lambda: setattr(handle, "cancelled", True)
        node.made_timers.append(handle)
        return handle

    node.create_timer = create_timer
    node.destroy_timer = node.destroyed.append
    node.get_logger = FakeLogger
    node.get_parameter = FakeParams(
        estop_cancel_nav=cancel, estop_zero_velocity=zero,
        estop_zero_period_s=0.1)
    return node


class TestGroundStation:
    def test_set_estop_publishes_only_on_change(self, ground_station):
        station = make_station(ground_station)
        station.set_estop(True)
        station.set_estop(True)
        station.set_estop(False)
        assert station.published == [True, False]
        assert station.estopped is False


class TestTheButtonDoesNotTrustTheDrones:
    """E-Stop was pressed in flight and the scout flew on.

    It worked by publishing /aeroaid/estop and leaving each node to cancel
    its own Nav2 goal. The search node had been killed by a MemoryError, so
    there was nothing left to hear the topic - a stop that needs the thing
    it is stopping to still be working. The ground station now stops the
    drones itself as well, by two routes, and these tests are about the
    routes working with no drone node in the picture at all.
    """

    def test_pressing_it_acts_and_does_not_only_announce(self, ground_station):
        station = make_station(ground_station)
        station.set_estop(True)
        assert station.estop_cancel["parrot1"].requests, \
            'the button published the topic and did nothing else'
        assert station.estop_cmd["parrot1"].sent

    def test_nav2_goals_are_cancelled_for_every_drone(self, ground_station):
        station = make_station(ground_station)
        station.stop_the_drones()
        for name in ("parrot1", "parrot2"):
            assert len(station.estop_cancel[name].requests) == 1, name

    def test_it_cancels_goals_this_process_never_sent(self, ground_station):
        """The point of going through the service: the goal belongs to the
        search node, which may be gone. A zero goal id and a zero stamp mean
        'every goal' in the action protocol."""
        station = make_station(ground_station)
        station.stop_the_drones()
        info = station.estop_cancel["parrot1"].requests[0].goal_info
        assert not any(info.goal_id.uuid), 'not a cancel-all request'
        assert info.stamp.sec == 0 and info.stamp.nanosec == 0

    def test_the_velocity_is_zeroed_for_every_drone(self, ground_station):
        station = make_station(ground_station)
        station.stop_the_drones()
        for name in ("parrot1", "parrot2"):
            sent = station.estop_cmd[name].sent
            assert sent, name
            assert sent[-1].linear.x == 0.0
            assert sent[-1].linear.z == 0.0
            assert sent[-1].angular.z == 0.0

    def test_the_velocity_is_held_down_not_sent_once(self, ground_station):
        """The drone keeps the last velocity it was given, so one zero is
        not a stop if anything publishes again afterwards."""
        station = make_station(ground_station)
        station.stop_the_drones()
        assert len(station.made_timers) == 1, 'nothing keeps the velocity at zero'
        assert station.made_timers[0].callback == station.publish_zero_velocity

    def test_pressing_it_twice_does_not_stack_timers(self, ground_station):
        station = make_station(ground_station)
        station.stop_the_drones()
        station.stop_the_drones()
        assert len(station.made_timers) == 1

    def test_a_dead_nav2_still_gets_the_velocity_zeroed(self, ground_station):
        """The case that matters most: Nav2 gone, or never up, and a drone
        still climbing on raw cmd_vel from takeoff_node."""
        station = make_station(ground_station, nav_ready=False)
        station.stop_the_drones()
        assert not station.estop_cancel["parrot1"].requests
        assert station.estop_cmd["parrot1"].sent, \
            'no cancel service and no zero velocity: nothing stopped the drone'

    def test_resume_stops_holding_the_velocity_down(self, ground_station):
        """Otherwise RESUME leaves a 10 Hz stream of zeroes fighting Nav2
        and nothing can fly again."""
        station = make_station(ground_station)
        station.set_estop(True)
        held = station.made_timers[0]
        station.set_estop(False)
        assert held.cancelled
        assert station.destroyed == [held]
        assert station.estop_zero_timer is None

    def test_it_can_stop_again_after_a_resume(self, ground_station):
        station = make_station(ground_station)
        station.set_estop(True)
        station.set_estop(False)
        station.set_estop(True)
        assert len(station.made_timers) == 2
        assert not station.made_timers[1].cancelled
