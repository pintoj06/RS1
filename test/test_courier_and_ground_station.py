"""Courier frame conversion and the ground station's thermal check."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy.time
import tf2_ros
from geometry_msgs.msg import PointStamped, TransformStamped

from conftest import FakeLogger


def contact(x, y, frame):
    msg = PointStamped()
    msg.header.frame_id = frame
    msg.point.x, msg.point.y = float(x), float(y)
    return msg


def map_to_odom(tx, ty, yaw):
    t = TransformStamped()
    t.header.frame_id = 'odom'
    t.child_frame_id = 'map'
    t.transform.translation.x, t.transform.translation.y = float(tx), float(ty)
    t.transform.rotation.z = math.sin(yaw / 2)
    t.transform.rotation.w = math.cos(yaw / 2)
    return t


@pytest.fixture
def fake_courier(courier_node):
    buffer = tf2_ros.Buffer()
    courier = SimpleNamespace(
        scout_odom_frame='odom',
        scout_tf_buffer=buffer,
        get_logger=FakeLogger,
    )
    convert = courier_node.CourierNode.contact_to_own_frame
    return courier, lambda msg: convert(courier, msg)


class TestContactToOwnFrame:
    def test_already_in_odom_passes_through(self, fake_courier):
        _, convert = fake_courier
        assert convert(contact(3, -4, 'odom')) == (3.0, -4.0)

    def test_waits_for_missing_transform(self, fake_courier):
        _, convert = fake_courier
        assert convert(contact(1, 1, 'map')) is None

    def test_translation_only(self, fake_courier):
        courier, convert = fake_courier
        courier.scout_tf_buffer.set_transform_static(map_to_odom(2, -1, 0.0), 'test')
        x, y = convert(contact(5, 5, 'map'))
        assert (x, y) == pytest.approx((7.0, 4.0))

    def test_rotation_and_translation(self, fake_courier):
        courier, convert = fake_courier
        courier.scout_tf_buffer.set_transform_static(
            map_to_odom(10, 0, math.pi / 2), 'test')
        # (1, 0) in map rotates to (0, 1), then shifts by (10, 0).
        x, y = convert(contact(1, 0, 'map'))
        assert (x, y) == pytest.approx((10.0, 1.0), abs=1e-9)



class TestScoutCostmapRelay:
    """The courier plans round what the scout has mapped, not only round what
    its own 10 m lidar can see."""

    @pytest.fixture
    def relay(self, courier_node):
        from conftest import FakeParams, make_costmap
        sent = []
        clock = SimpleNamespace(t=0.0)
        courier = SimpleNamespace(
            map_frame='parrot2_odom',
            _scout_costmap_sent=None,
            scout_costmap_pub=SimpleNamespace(publish=sent.append),
            get_parameter=FakeParams(scout_costmap_period=2.0),
            get_logger=FakeLogger,
            get_clock=lambda: SimpleNamespace(
                now=lambda: rclpy.time.Time(nanoseconds=int(clock.t * 1e9))),
        )
        run = courier_node.CourierNode.relay_scout_costmap

        def scout_map():
            grid = make_costmap(4, 4, resolution=0.15, origin=(-30.0, -25.0))
            grid.header.frame_id = 'parrot1_odom'
            return grid
        return SimpleNamespace(run=lambda m: run(courier, m), sent=sent,
                               clock=clock, scout_map=scout_map)

    def test_relabelled_into_the_couriers_frame(self, relay):
        """Same world coordinates; only the label changes. There is no TF
        between the two robots' trees for Nav2 to convert with."""
        grid = relay.scout_map()
        relay.run(grid)
        assert len(relay.sent) == 1
        out = relay.sent[0]
        assert out.header.frame_id == 'parrot2_odom'
        assert out.info.origin.position.x == -30.0
        assert out.info.origin.position.y == -25.0
        assert list(out.data) == list(grid.data)

    def test_throttled(self, relay):
        relay.run(relay.scout_map())
        relay.clock.t = 0.2                 # the scout's next 5 Hz frame
        relay.run(relay.scout_map())
        assert len(relay.sent) == 1
        relay.clock.t = 2.1
        relay.run(relay.scout_map())
        assert len(relay.sent) == 2


@pytest.fixture
def detect(ground_station):
    station = SimpleNamespace(person_min_k=305.0, person_min_pixels=600,
                              tracker=ground_station.GroundTracker(2.0, 10.0))
    return lambda kelvin, frame: \
        ground_station.GroundStationNode._detect_person(station, kelvin, frame)


class TestThermalDetection:
    def test_cold_scene_gives_nothing(self, detect):
        kelvin = np.full((480, 640), 290.0, dtype=np.float32)
        assert detect(kelvin, np.zeros((480, 640, 3), np.uint8)) is None

    def test_small_hot_spot_is_ignored(self, detect):
        kelvin = np.full((480, 640), 290.0, dtype=np.float32)
        kelvin[100:110, 100:110] = 310.0          # 100 px < 600
        assert detect(kelvin, np.zeros((480, 640, 3), np.uint8)) is None

    def test_person_sized_blob_is_found_and_boxed(self, detect):
        kelvin = np.full((480, 640), 290.0, dtype=np.float32)
        kelvin[200:240, 300:330] = 309.0          # 1200 px
        kelvin[220, 315] = 311.5
        frame = np.zeros((480, 640, 3), np.uint8)
        peak, count = detect(kelvin, frame)
        assert peak == pytest.approx(311.5)
        assert count == 1200
        # The box is drawn round the blob, not elsewhere in the frame.
        assert frame[200, 300].any()
        assert not frame[10, 10].any()

    def test_threshold_is_inclusive(self, detect):
        kelvin = np.full((200, 200), 290.0, dtype=np.float32)
        kelvin[80:120, 80:120] = 305.0            # 1600 px, clear of the edges
        assert detect(kelvin, np.zeros((200, 200, 3), np.uint8)) is not None

    def test_body_at_frame_edge_is_ignored_until_fully_in_view(self, detect):
        # The run that prompted this: 604 px of a person just entering the
        # frame held the search with them at the edge, where the RGB
        # detector won't count them, so the check timed out and missed them.
        kelvin = np.full((480, 720), 290.0, dtype=np.float32)
        kelvin[0:30, 300:340] = 310.0             # 1200 px, cut off at the top
        assert detect(kelvin, np.zeros((480, 720, 3), np.uint8)) is None
        kelvin[:] = 290.0
        kelvin[60:90, 300:340] = 310.0            # same body, fully in view
        assert detect(kelvin, np.zeros((480, 720, 3), np.uint8)) is not None


class TestContactRadius:
    def test_near(self, ground_station):
        window = SimpleNamespace(CONTACT_RADIUS=5.0)
        near = ground_station.MainWindow._near
        spots = [(0.0, 0.0), (20.0, 20.0)]
        assert near(window, 3.0, 3.0, spots)          # 4.2 m away
        assert not near(window, 4.0, 4.0, spots)      # 5.7 m away
        assert not near(window, 1.0, 1.0, [])


class TestOverlaysSurviveALateTransform:
    """The sweep vanished from the map after a restart.

    search_path and search_vantage are published ONCE per plan. The ground
    station converted them into its own frame on arrival and returned early
    if the transform was not ready - which it is not for the first seconds,
    when the ground station starts before the scout and SLAM. The route was
    then lost for good, because nothing republishes it until the next replan.

    They are now kept in the frame they arrived in and converted when drawn.
    """

    def path_msg(self, *points):
        from nav_msgs.msg import Path
        from geometry_msgs.msg import PoseStamped
        msg = Path()
        msg.header.frame_id = 'parrot1_odom'
        for x, y in points:
            p = PoseStamped()
            p.pose.position.x, p.pose.position.y = x, y
            msg.poses.append(p)
        return msg

    def test_a_sweep_arriving_before_tf_is_kept(self, ground_station):
        node = ground_station.GroundStationNode.__new__(
            ground_station.GroundStationNode)
        node.search_path_raw, node.search_path_frame = [], ""
        node._on_path(self.path_msg((1.0, 2.0), (3.0, 4.0)))
        assert node.search_path_raw == [(1.0, 2.0), (3.0, 4.0)], \
            'the sweep was dropped instead of kept for later conversion'
        assert node.search_path_frame == 'parrot1_odom'

    def test_a_vantage_arriving_before_tf_is_kept(self, ground_station):
        from geometry_msgs.msg import PointStamped
        node = ground_station.GroundStationNode.__new__(
            ground_station.GroundStationNode)
        node.vantage_raw, node.vantage_frame = None, ""
        msg = PointStamped()
        msg.header.frame_id = 'parrot1_odom'
        msg.point.x, msg.point.y = 5.0, 6.0
        node._on_vantage(msg)
        assert node.vantage_raw == (5.0, 6.0)
        assert node.vantage_frame == 'parrot1_odom'

    def test_an_empty_sweep_clears_it(self, ground_station):
        node = ground_station.GroundStationNode.__new__(
            ground_station.GroundStationNode)
        node.search_path_raw, node.search_path_frame = [(1.0, 1.0)], 'x'
        node._on_path(self.path_msg())
        assert node.search_path_raw == []
class TestReleaseRadius:
    """The package drops once the courier is within release_radius of the
    contact, not only when Nav2 calls the goal reached at 0.25 m."""

    @pytest.fixture
    def courier(self, courier_node):
        from action_msgs.msg import GoalStatus
        from conftest import FakeParams
        Courier = courier_node.CourierNode

        class Fake:
            check_release = Courier.check_release
            release = Courier.release
            goal_finished = Courier.goal_finished

            def __init__(self):
                self.target = (10.0, 0.0)
                self.at = (0.0, 0.0)
                self.busy = True
                self.sent = True
                self.released = False
                self.estopped = False
                self.drops = 0
                self.cancelled = False
                handle = SimpleNamespace(cancel_goal_async=self._cancel)
                self.goal_handle = handle
                self.release_pub = SimpleNamespace(publish=self._drop)
                self.get_parameter = FakeParams(release_radius=1.5)
                self.get_logger = FakeLogger

            def _drop(self, _msg):
                self.drops += 1

            def _cancel(self):
                self.cancelled = True

            def own_xy(self):
                return self.at

        fake = Fake()
        fake.result = lambda status: SimpleNamespace(
            result=lambda: SimpleNamespace(status=status))
        fake.GoalStatus = GoalStatus
        return fake

    def test_holds_on_to_it_while_far_off(self, courier):
        courier.at = (8.0, 0.0)              # 2 m out
        courier.check_release()
        assert courier.drops == 0 and not courier.cancelled

    def test_drops_inside_the_radius_and_stops_there(self, courier):
        courier.at = (8.6, 0.5)              # 1.49 m out
        courier.check_release()
        assert courier.drops == 1
        assert courier.cancelled             # hovers, no creeping in
        courier.check_release()
        assert courier.drops == 1            # once only

    def test_the_cancel_is_not_reported_as_a_miss(self, courier):
        courier.at = (9.0, 0.0)
        courier.check_release()
        courier.goal_finished(courier.result(courier.GoalStatus.STATUS_CANCELED))
        assert courier.released and courier.sent   # no retry either
        assert courier.drops == 1

    def test_arriving_still_drops_if_the_check_missed_it(self, courier):
        courier.goal_finished(courier.result(courier.GoalStatus.STATUS_SUCCEEDED))
        assert courier.drops == 1

    def test_not_while_estopped(self, courier):
        courier.at = (9.5, 0.0)
        courier.estopped = True
        courier.check_release()
        assert courier.drops == 0
