"""A dismissed contact must not come back.

Each YOLO human is projected to the ground and tracked frame to frame
(GroundTracker). Dismissing a contact dismisses its track, which keeps
following that person however the projection drifts, so they are never
raised again; anyone else, even close by, still counts.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from conftest import FakeLogger

W, H = 720, 480
THERMAL_FOV = 1.0472


class TestPixelToGround:
    def project(self, gs, u, v, yaw=0.0, alt=10.0, x=0.0, y=0.0):
        return gs.pixel_to_ground(u, v, W, H, THERMAL_FOV, alt, x, y, yaw)

    def test_centre_is_below_the_drone(self, ground_station):
        assert self.project(ground_station, 360, 240, x=3, y=4) == pytest.approx((3, 4))

    def test_top_of_image_is_forward(self, ground_station):
        gx, gy = self.project(ground_station, 360, 0)
        assert gx == pytest.approx(240 * 2 * 10 * math.tan(THERMAL_FOV / 2) / W)
        assert gy == pytest.approx(0)

    def test_right_of_image_is_drone_right(self, ground_station):
        gx, gy = self.project(ground_station, 720, 240)
        assert gx == pytest.approx(0)
        assert gy < 0                      # right of a drone facing +x is -y

    def test_yaw_rotates_the_offset(self, ground_station):
        gx, gy = self.project(ground_station, 360, 0, yaw=math.pi / 2)
        assert gx == pytest.approx(0, abs=1e-9)
        assert gy > 0                      # forward is now +y

    def test_matches_live_thermal_frame(self, ground_station):
        # Measured on the running sim, when the thermal camera was 60 deg
        # (THERMAL_FOV): scout at (-8.72, -1.90), z 9.95, yaw 0; the wombat's
        # blob centred at pixel (437.1, 171.8). The wombat is at (-7.5, -3.1)
        # in large_demo.sdf.
        gx, gy = self.project(ground_station, 437.1, 171.8,
                              alt=9.95 - 0.05, x=-8.72, y=-1.90)
        assert math.hypot(gx + 7.5, gy + 3.1) < 0.3


class TestMatchOneToOne:
    def test_closest_pairs_first(self, ground_station):
        m = ground_station.match_one_to_one(
            [(0.0, 0.0), (1.0, 0.0)], [(0.9, 0.0)], gate=2.0)
        assert m == {1: 0}                 # only one point gets the anchor

    def test_gate(self, ground_station):
        assert ground_station.match_one_to_one([(3.0, 0.0)], [(0.0, 0.0)], 2.0) == {}

    def test_unprojectable_points_skipped(self, ground_station):
        assert ground_station.match_one_to_one([None], [(0.0, 0.0)], 2.0) == {}


class TestGroundTracker:
    def tracker(self, gs):
        return gs.GroundTracker(gate=2.0, timeout_s=10.0)

    def test_same_person_keeps_track(self, ground_station):
        t = self.tracker(ground_station)
        a, = t.update([(0.0, 0.0)], now=0.0)
        b, = t.update([(0.8, 0.3)], now=0.5)
        assert a.id == b.id
        assert (b.x, b.y) == (0.8, 0.3)    # follows the latest projection

    def test_far_detection_is_new_track(self, ground_station):
        t = self.tracker(ground_station)
        a, = t.update([(0.0, 0.0)], now=0.0)
        b, = t.update([(5.0, 0.0)], now=0.5)
        assert a.id != b.id

    def test_replay_of_live_drift(self, ground_station):
        # The run that prompted this: one person (person1, really at
        # (-6.36, -3.07)) projected to (-5.9, -3.4), then (-7.0, -3.5), then
        # (-4.8, -3.3) over successive frames - 2.2 m of spread, so a fixed
        # 1 m radius missed two of the three. Frame to frame it moves less.
        t = self.tracker(ground_station)
        first, = t.update([(-5.9, -3.4)], now=0.0)
        t.dismiss(first.id)
        drift = [(-6.4, -3.5), (-7.0, -3.5), (-6.3, -3.4), (-5.5, -3.3), (-4.8, -3.3)]
        for k, p in enumerate(drift, start=1):
            track, = t.update([p], now=0.5 * k)
            assert track.id == first.id and track.dismissed

    def test_dismissed_track_survives_timeout(self, ground_station):
        t = self.tracker(ground_station)
        a, = t.update([(0.0, 0.0)], now=0.0)
        t.dismiss(a.id)
        b, = t.update([(1.0, 0.0)], now=100.0)   # back on a later sweep row
        assert b.id == a.id and b.dismissed

    def test_undismissed_track_times_out(self, ground_station):
        t = self.tracker(ground_station)
        a, = t.update([(0.0, 0.0)], now=0.0)
        b, = t.update([(0.5, 0.0)], now=20.0)
        assert b.id != a.id

    def test_second_person_beside_dismissed_one(self, ground_station):
        t = self.tracker(ground_station)
        a, = t.update([(0.0, 0.0)], now=0.0)
        t.dismiss(a.id)
        p1, p2 = t.update([(0.2, 0.0), (1.2, 0.0)], now=0.5)
        assert p1.id == a.id and p1.dismissed
        assert p2.id != a.id and not p2.dismissed

    def test_dismiss_fallback_position(self, ground_station):
        t = self.tracker(ground_station)
        track = t.dismiss(None, fallback_xy=(3.0, 4.0))
        assert track.dismissed
        again, = t.update([(3.5, 4.0)], now=1.0)
        assert again.id == track.id


def make_node(ground_station, project=None):
    """A node whose ground_point maps pixel (u, v) to (u / 100, v / 100)."""
    node = SimpleNamespace(
        tracker=ground_station.GroundTracker(2.0, 10.0), rgb_frame=None,
        rgb_hfov=1.4, thermal_hfov=THERMAL_FOV, person_min_k=305.0,
        person_min_pixels=600, rgb_hit=None, rgb_hit_xy=None,
        rgb_hit_track=None, _detections_time=0.0,
        thermal_mask=np.ones((H, W), bool),     # heat everywhere by default
        thermal_enabled=True, rgb_enabled=True,  # SETUP page defaults
    )
    node.ground_point = project or (lambda u, v, w, h, fov, stamp: (u / 100, v / 100))
    return node


def detections(*boxes):
    msg = Detection2DArray()
    for u, v, score in boxes:
        d = Detection2D()
        d.bbox.center.position.x, d.bbox.center.position.y = float(u), float(v)
        d.bbox.size_x, d.bbox.size_y = 40.0, 40.0
        r = ObjectHypothesisWithPose()
        r.hypothesis.class_id, r.hypothesis.score = 'Human', float(score)
        d.results.append(r)
        msg.detections.append(d)
    return msg


class TestHumanMustBeOnTheHeat:
    """A contact's heat and its YOLO human must be the same object: a false
    human on one side of the frame plus a wombat's heat on the other is not
    a person."""

    def run(self, ground_station, node, msg):
        ground_station.GroundStationNode._on_detections(node, msg)

    def heat_at(self, x1, y1, x2, y2):
        mask = np.zeros((H, W), bool)
        mask[y1:y2, x1:x2] = True
        return mask

    def test_human_on_heat_counts(self, ground_station):
        node = make_node(ground_station)
        node.thermal_mask = self.heat_at(380, 180, 420, 220)
        self.run(ground_station, node, detections((400, 200, 0.9)))
        assert node.rgb_hit is not None

    def test_human_away_from_heat_is_not_a_hit(self, ground_station):
        node = make_node(ground_station)
        node.thermal_mask = self.heat_at(600, 380, 640, 420)   # wombat, far side
        self.run(ground_station, node, detections((100, 100, 0.9)))
        assert node.rgb_hit is None

    def test_only_the_human_on_heat_counts(self, ground_station):
        node = make_node(ground_station)
        node.thermal_mask = self.heat_at(380, 180, 420, 220)
        self.run(ground_station, node,
                 detections((100, 100, 0.95), (400, 200, 0.6)))
        assert node.rgb_hit[0] == pytest.approx(0.6)

    def test_no_thermal_yet_means_no_hit(self, ground_station):
        node = make_node(ground_station)
        node.thermal_mask = None
        self.run(ground_station, node, detections((400, 200, 0.9)))
        assert node.rgb_hit is None

    def test_scales_when_thermal_resolution_differs(self, ground_station):
        node = make_node(ground_station)
        mask = np.zeros((H // 2, W // 2), bool)      # half-size thermal image
        mask[90:110, 190:210] = True                 # = RGB (400, 200)
        node.thermal_mask = mask
        self.run(ground_station, node, detections((400, 200, 0.9)))
        assert node.rgb_hit is not None


class TestYoloAfterDismiss:
    def run(self, ground_station, node, msg):
        ground_station.GroundStationNode._on_detections(node, msg)

    def test_dismissed_person_is_not_a_hit(self, ground_station):
        node = make_node(ground_station)
        self.run(ground_station, node, detections((400, 200, 0.9)))
        node.tracker.dismiss(node.rgb_hit_track)
        node.rgb_hit = None
        # Projection drifts 1.5 m on the next frame: still the same person.
        self.run(ground_station, node, detections((550, 200, 0.9)))
        assert node.rgb_hit is None

    def test_other_person_still_counts(self, ground_station):
        node = make_node(ground_station)
        self.run(ground_station, node, detections((400, 200, 0.9)))
        dismissed = node.rgb_hit_track
        node.tracker.dismiss(dismissed)
        # Dismissed person scores higher, but the person 1.2 m away wins.
        self.run(ground_station, node,
                 detections((400, 200, 0.95), (520, 200, 0.6)))
        assert node.rgb_hit[0] == pytest.approx(0.6)
        assert node.rgb_hit_track != dismissed
        assert node.rgb_hit_xy == pytest.approx((5.2, 2.0))


class TestThermalAfterDismiss:
    def detect(self, ground_station, node, kelvin):
        frame = np.zeros(kelvin.shape + (3,), np.uint8)
        return ground_station.GroundStationNode._detect_person(
            node, kelvin, frame, stamp=object())

    def scene(self):
        kelvin = np.full((H, W), 290.0, dtype=np.float32)
        kelvin[100:140, 100:130] = 310.0      # blob A, ground ~(1.15, 1.20)
        kelvin[300:340, 500:530] = 310.0      # blob B, ground ~(5.15, 3.20)
        return kelvin

    def test_dismissed_blob_removed_other_kept(self, ground_station):
        node = make_node(ground_station)
        node.tracker.dismiss(None, fallback_xy=(2.0, 1.5))   # 0.9 m from A
        peak, count = self.detect(ground_station, node, self.scene())
        assert count == 1200                  # blob B only

    def test_one_blob_per_dismissed_track(self, ground_station):
        # Two blobs both inside the gate of one dismissed track: it may only
        # claim one, the other (a second warm body) still counts.
        kelvin = np.full((H, W), 290.0, dtype=np.float32)
        kelvin[100:140, 100:130] = 310.0      # ~(1.15, 1.20)
        kelvin[100:140, 200:230] = 310.0      # ~(2.15, 1.20)
        node = make_node(ground_station)
        node.tracker.dismiss(None, fallback_xy=(1.2, 1.2))
        peak, count = self.detect(ground_station, node, kelvin)
        assert count == 1200

    def test_nothing_dismissed_counts_everything(self, ground_station):
        node = make_node(ground_station)
        peak, count = self.detect(ground_station, node, self.scene())
        assert count == 2400


class TestRuling:
    def window(self, ground_station, contact):
        node = SimpleNamespace(
            tracker=ground_station.GroundTracker(2.0, 10.0),
            dismiss_cooldown_s=2.0, get_logger=FakeLogger,
            send_decision=lambda *a: None, set_hold=lambda hold: None)
        win = SimpleNamespace(
            node=node, contacts=[contact], hold_for_confirmed=False,
            _no_check_until=0.0, _check_until=None,
            vis_page=SimpleNamespace(frozen=None),
            ir_page=SimpleNamespace(frozen=None))
        win._selected = lambda: contact
        cls = ground_station.MainWindow
        win._update_hold = lambda: cls._update_hold(win)
        return win

    def contact(self, gs, **kw):
        return gs.Contact(id=1, x=0.0, y=0.0, rgb_confidence=0.8,
                          thermal_confidence=0.5, **kw)

    def test_dismiss_dismisses_the_track(self, ground_station):
        c = self.contact(ground_station)
        win = self.window(ground_station, c)
        track, = win.node.tracker.update([(2.5, -1.0)], now=0.0)
        c.track_id = track.id
        ground_station.MainWindow._rule(win, "DISMISSED")
        assert track.dismissed

    def test_dismiss_with_only_a_position(self, ground_station):
        c = self.contact(ground_station, obj_x=2.5, obj_y=-1.0)
        win = self.window(ground_station, c)
        ground_station.MainWindow._rule(win, "DISMISSED")
        [t] = win.node.tracker.dismissed()
        assert (t.x, t.y) == (2.5, -1.0)

    def test_confirm_does_not_dismiss(self, ground_station):
        c = self.contact(ground_station)
        win = self.window(ground_station, c)
        win.node.publish_confirmed_point = lambda x, y: None
        track, = win.node.tracker.update([(2.5, -1.0)], now=0.0)
        c.track_id = track.id
        ground_station.MainWindow._rule(win, "CONFIRMED")
        assert not track.dismissed


class TestNightFloodlight:
    """Night mode: the scout's floodlight follows the search hold."""

    def node(self, ground_station, night):
        sent = []
        node = SimpleNamespace(
            night_mode=night, holding=False, get_logger=FakeLogger,
            # What the node picks when flashlight_mode is left unset.
            flashlight_mode='THERMAL' if night else 'OFF',
            hold_pub=SimpleNamespace(publish=lambda m: None),
            light_pub=SimpleNamespace(publish=sent.append))
        cls = ground_station.GroundStationNode
        node._send_light = lambda on: cls._send_light(node, on)
        return node, sent, lambda hold: cls.set_hold(node, hold)

    def test_light_on_while_held_off_on_resume(self, ground_station):
        node, sent, set_hold = self.node(ground_station, night=True)
        set_hold(True)
        set_hold(False)
        assert [m.intensity > 0 for m in sent] == [True, False]
        assert sent[0].name == 'flashlight' and sent[0].direction.z == -1.0

    def test_day_mode_never_touches_the_light(self, ground_station):
        node, sent, set_hold = self.node(ground_station, night=False)
        set_hold(True)
        assert sent == []
