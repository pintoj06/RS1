"""The ground station's SETUP page: flashlight mode, body-heat threshold and
switching a detector off.

The node and window tests build the real GroundStationNode and MainWindow
(offscreen Qt, no simulation), so they check the wiring as well as the logic.
"""

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from PyQt5.QtWidgets import QApplication
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


@pytest.fixture
def ros():
    # Same as the other test files: other modules may already have started
    # rclpy, and it is left running for the ones after.
    if not rclpy.ok():
        rclpy.init()


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def node(ros, ground_station):
    """A real ground station node, its flashlight messages captured."""
    n = ground_station.GroundStationNode()
    n.lights = []
    n.light_pub = SimpleNamespace(publish=n.lights.append)
    yield n
    # Stop the TF listener's own spin thread now. Left to its __del__, that
    # runs after rclpy has shut down and prints a traceback per test.
    tl = n.tf_listener
    tl.executor.shutdown()
    tl.dedicated_listener_thread.join()
    del tl.dedicated_listener_thread
    tl.unregister()
    n.destroy_node()


@pytest.fixture
def window(app, node, ground_station):
    win = ground_station.MainWindow(node)
    yield win
    win.ui_timer.stop()
    win.blink_timer.stop()
    win.close()


def human(u=400.0, v=200.0, score=0.9):
    msg = Detection2DArray()
    d = Detection2D()
    d.bbox.center.position.x, d.bbox.center.position.y = u, v
    d.bbox.size_x = d.bbox.size_y = 40.0
    r = ObjectHypothesisWithPose()
    r.hypothesis.class_id, r.hypothesis.score = 'Human', score
    d.results.append(r)
    msg.detections.append(d)
    return msg


class TestDefaultsUnchanged:
    """Left alone, the settings must behave exactly as before they existed."""

    def test_day_defaults(self, node):
        assert node.flashlight_mode == 'OFF'
        assert node.thermal_enabled and node.rgb_enabled
        assert node.person_min_k == 305.0

    def test_day_run_never_publishes_to_the_light(self, node):
        node._resend_light()
        node.set_hold(True)
        node._resend_light()
        assert node.lights == []


class TestFlashlight:
    def test_on_is_lit_without_a_hold(self, node):
        node.set_flashlight_mode('ON')
        assert [m.intensity > 0 for m in node.lights] == [True]
        node._resend_light()
        assert node.lights[-1].intensity > 0

    def test_thermal_follows_the_hold(self, node):
        node.set_flashlight_mode('THERMAL')
        node.set_hold(True)
        node.set_hold(False)
        assert [m.intensity > 0 for m in node.lights] == [False, True, False]

    def test_off_after_on_turns_it_off_and_keeps_it_off(self, node):
        node.set_flashlight_mode('ON')
        node.set_flashlight_mode('OFF')
        node.set_hold(True)          # OFF ignores holds
        node._resend_light()         # and re-sends "off", not nothing
        assert [m.intensity > 0 for m in node.lights] == [True, False, False]

    def test_only_the_brightness_changes(self, node):
        node.set_flashlight_mode('ON')
        node.set_flashlight_mode('OFF')
        lit, dark = node.lights
        for field in ('name', 'type', 'range', 'spot_inner_angle',
                      'spot_outer_angle', 'attenuation_constant'):
            assert getattr(lit, field) == getattr(dark, field)
        assert lit.direction.z == dark.direction.z == -1.0

    def test_bad_mode_is_ignored(self, node):
        node.set_flashlight_mode('STROBE')
        assert node.flashlight_mode == 'OFF' and node.lights == []


class TestHeatThreshold:
    def test_clamped_to_the_display_range(self, node):
        node.set_person_min_k(1000.0)
        assert node.person_min_k == node.thermal_max_k
        node.set_person_min_k(0.0)
        assert node.person_min_k == node.thermal_min_k

    def test_detection_uses_the_new_threshold(self, node):
        kelvin = np.full((480, 720), 290.0, dtype=np.float32)
        kelvin[200:240, 340:370] = 308.0       # 1200 px at 308 K
        frame = np.zeros((480, 720, 3), np.uint8)
        assert node._detect_person(kelvin, frame) is not None
        node.set_person_min_k(309.0)
        assert node._detect_person(kelvin, frame) is None


class TestOneDetectorStaysOn:
    def test_cannot_switch_both_off(self, node):
        assert node.set_thermal_detection(False)
        assert not node.set_rgb_detection(False)
        assert node.rgb_enabled and not node.thermal_enabled

    def test_switching_thermal_off_forgets_old_heat(self, node):
        node.thermal_hit, node.thermal_mask = (310.0, 900), np.ones((4, 4), bool)
        node.set_thermal_detection(False)
        assert node.thermal_hit is None and node.thermal_mask is None

    def test_switching_yolo_off_forgets_old_humans(self, node):
        node.rgb_hit = (0.9, time.monotonic())
        node.set_rgb_detection(False)
        assert node.fresh_rgb_hit() is None


class TestYoloOnly:
    """Thermal off: a YOLO human raises the contact without any heat."""

    def test_human_counts_without_heat_under_it(self, node):
        node.set_thermal_detection(False)
        node._on_detections(human())
        assert node.fresh_rgb_hit() == pytest.approx(0.9)

    def test_with_thermal_on_the_same_human_does_not(self, node):
        node.thermal_mask = np.zeros((480, 720), bool)
        node._on_detections(human())
        assert node.fresh_rgb_hit() is None

    def test_contact_raised_on_yolo_alone(self, window, node):
        node.set_thermal_detection(False)
        node.drone_xy = (5.0, 5.0)
        node._on_detections(human())
        window._check_detections()
        assert len(window.contacts) == 1
        assert window.contacts[0].thermal_confidence == 0.0
        assert node.holding         # an unruled contact holds the search

    def test_running_check_ends_when_thermal_goes_off(self, window, node):
        window._check_until = time.monotonic() + 6.0
        window._update_hold()
        assert node.holding
        node.set_thermal_detection(False)
        window._check_detections()
        assert window._check_until is None and not node.holding


class TestThermalOnly:
    """YOLO off: body heat raises the contact, at the heat's position."""

    def test_yolo_boxes_ignored_but_detector_still_alive(self, node):
        node.set_rgb_detection(False)
        node._on_detections(human())
        assert node.rgb_hit is None and node.detector_alive()

    def test_animal_boxes_ignored(self, node):
        node.set_rgb_detection(False)
        node._on_animal_detections(human())
        assert node.animal_hit is None

    def test_contact_raised_on_heat_alone(self, window, node):
        node.set_rgb_detection(False)
        node.drone_xy = (5.0, 5.0)
        node.thermal_hit = (310.0, 1200)
        node.thermal_hit_xy = (7.0, 3.0)
        window._check_detections()
        assert len(window.contacts) == 1
        c = window.contacts[0]
        assert (c.x, c.y) == (7.0, 3.0) and (c.obj_x, c.obj_y) == (7.0, 3.0)
        assert c.rgb_confidence == 0.0 and c.track_id is None
        assert window._check_until is None      # no RGB check to wait on

    def test_dismissed_heat_is_not_raised_again(self, window, node):
        node.set_rgb_detection(False)
        node.drone_xy = (5.0, 5.0)
        node.thermal_hit, node.thermal_hit_xy = (310.0, 1200), (7.0, 3.0)
        window._check_detections()
        window.contacts_page.refresh(window.contacts)
        window.contacts_page.list.setCurrentRow(0)
        window._rule('DISMISSED')
        # The dismiss left a dismissed track at the heat, which is what
        # _detect_person uses to drop that blob from now on.
        assert [(t.x, t.y) for t in node.tracker.dismissed()] == [(7.0, 3.0)]


class TestSetupPage:
    def test_left_column_and_key_reach_it(self, window, ground_station):
        window.left_buttons[4].click()
        assert window.page == ground_station.PAGE_SETTINGS
        window.refresh()
        assert window.page_label.text() == 'SETUP'
        assert window.left_buttons[4].text() == 'SETUP'

    def test_right_column_changes_every_setting(self, window, node, ground_station):
        window.select_page(ground_station.PAGE_SETTINGS)
        window._press_right(0)
        assert node.flashlight_mode == 'ON'
        window._press_right(0)
        assert node.flashlight_mode == 'THERMAL'
        window._press_right(0)
        assert node.flashlight_mode == 'OFF'
        window._press_right(1)
        assert node.person_min_k == 305.5
        window._press_right(2)
        window._press_right(2)
        assert node.person_min_k == 304.5
        window._press_right(3)
        assert not node.thermal_enabled
        window._press_right(4)               # refused: thermal is off
        assert node.rgb_enabled
        window._press_right(3)
        window._press_right(4)
        assert node.thermal_enabled and not node.rgb_enabled

    def test_labels_show_the_state(self, window, node, ground_station):
        window.select_page(ground_station.PAGE_SETTINGS)
        node.set_thermal_detection(False)
        window.refresh()
        labels = [b.text() for b in window.right_buttons]
        assert labels[:5] == ['LIGHT\nOFF', 'HEAT+\n305.0K', 'HEAT-',
                              'IR DET', 'AI DET']
        assert window.ir_label.text() == 'IR OFF'
        assert 'Thermal off' in window.settings_page.text.text()

    def test_other_pages_unchanged(self, window, ground_station):
        window.select_page(ground_station.PAGE_CONTACTS)
        assert window._right_actions()[0][0] == 'RESUME'
        window.select_page(ground_station.PAGE_VIS)
        assert all(not a[0] for a in window._right_actions())
