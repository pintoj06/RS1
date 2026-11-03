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
        kelvin = np.full((100, 100), 305.0, dtype=np.float32)
        assert detect(kelvin, np.zeros((100, 100, 3), np.uint8)) is not None


class TestContactRadius:
    def test_near(self, ground_station):
        window = SimpleNamespace(CONTACT_RADIUS=5.0)
        near = ground_station.MainWindow._near
        spots = [(0.0, 0.0), (20.0, 20.0)]
        assert near(window, 3.0, 3.0, spots)          # 4.2 m away
        assert not near(window, 4.0, 4.0, spots)      # 5.7 m away
        assert not near(window, 1.0, 1.0, [])
