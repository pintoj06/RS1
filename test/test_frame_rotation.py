"""map -> odom is SLAM's drift correction, and it is a shift AND a twist.

Four places applied only the shift: the drawn search area (which decides the
ground actually swept), the sweep and vantage overlays, the courier's route
and marker, and the coverage overlay. Measured on a live run the twist was
-1.894 degrees, which costs r*sin(theta) - nothing at the frame origin, about
a metre 30 m out. So it hid near the spawn point and grew with distance, and
showed up as the drone flying a line parallel to its own drawn path.

courier_node.contact_to_own_frame already did it correctly, which is both
where the formula came from and the proof the team agreed it was needed.
"""
import math

import numpy as np
import pytest

from conftest import load_module


@pytest.fixture(scope='module')
def gs():
    return load_module('ground_station', 'UI/ground_station.py')


@pytest.fixture(scope='module')
def sn():
    return load_module('search_node', 'scripts/search_node.py')


def tf_for(yaw=0.0, tx=0.0, ty=0.0):
    from geometry_msgs.msg import Transform
    t = Transform()
    t.translation.x, t.translation.y = float(tx), float(ty)
    t.rotation.z = math.sin(yaw / 2.0)
    t.rotation.w = math.cos(yaw / 2.0)
    return t


class TestApplyTransform2d:
    def test_identity_changes_nothing(self, gs):
        assert gs.apply_transform_2d(tf_for(), 3.0, 4.0) == pytest.approx((3.0, 4.0))

    def test_pure_translation(self, gs):
        got = gs.apply_transform_2d(tf_for(tx=1.0, ty=-2.0), 3.0, 4.0)
        assert got == pytest.approx((4.0, 2.0))

    def test_quarter_turn(self, gs):
        """A quarter turn sends +x to +y. A sign error here is unmissable."""
        got = gs.apply_transform_2d(tf_for(yaw=math.pi / 2), 1.0, 0.0)
        assert got == pytest.approx((0.0, 1.0), abs=1e-9)

    def test_rotate_then_translate_not_the_other_way(self, gs):
        got = gs.apply_transform_2d(tf_for(yaw=math.pi / 2, tx=10.0), 1.0, 0.0)
        assert got == pytest.approx((10.0, 1.0), abs=1e-9)

    def test_the_error_the_old_code_made(self, gs):
        """Translation only, at the live -1.894 deg: about a metre at 30 m."""
        yaw = math.radians(-1.894)
        t = tf_for(yaw=yaw, tx=-0.438, ty=-0.355)
        for r, expect in ((5, 0.17), (30, 0.99)):
            right = gs.apply_transform_2d(t, r, 0.0)
            wrong = (r + t.translation.x, 0.0 + t.translation.y)
            off = math.hypot(right[0] - wrong[0], right[1] - wrong[1])
            assert off == pytest.approx(expect, abs=0.05), (r, off)

    def test_the_search_node_agrees_with_the_ground_station(self, gs, sn):
        t = tf_for(yaw=0.4, tx=2.0, ty=-3.0)
        for x, y in ((0.0, 0.0), (10.0, 0.0), (-7.5, 21.25)):
            assert sn.apply_transform_2d(t, x, y) == \
                pytest.approx(gs.apply_transform_2d(t, x, y))

    def test_the_courier_agrees_too(self, gs):
        """courier_node had the only correct copy. It must stay in step."""
        from types import SimpleNamespace
        from geometry_msgs.msg import PointStamped
        courier = load_module('courier_node', 'scripts/courier_node.py')
        t = tf_for(yaw=0.4, tx=2.0, ty=-3.0)
        node = SimpleNamespace(
            scout_odom_frame='parrot1_odom',
            scout_tf_buffer=SimpleNamespace(
                lookup_transform=lambda *a, **k: SimpleNamespace(transform=t)),
            get_logger=lambda: SimpleNamespace(info=lambda *a, **k: None))
        msg = PointStamped()
        msg.header.frame_id = 'parrot1_map'
        msg.point.x, msg.point.y = 10.0, 4.0
        got = courier.CourierNode.contact_to_own_frame(node, msg)
        assert got == pytest.approx(gs.apply_transform_2d(t, 10.0, 4.0))


class TestPlaceOverlay:
    """The coverage overlay is drawn as an axis-aligned bitmap, so the twist
    has to move the pixels - shifting the bounds cannot express it."""

    def marked(self, h=20, w=20, r0=2, r1=5, c0=3, c1=6):
        img = np.zeros((h, w, 4), dtype=np.uint8)
        img[r0:r1, c0:c1] = 255
        return img

    def centre_in_grid(self, img, res, ox, oy, r0, r1, c0, c1):
        """Where the marked block's centre is, in the grid's own frame.
        Row 0 of the image is NORTH, so y counts down the rows."""
        h = img.shape[0]
        col_mid = (c0 + c1) / 2.0
        row_mid = (r0 + r1) / 2.0
        return (ox + col_mid * res, oy + (h - row_mid) * res)

    def centre_in_output(self, out, bounds, res):
        ys, xs = np.nonzero(out[:, :, 3])
        assert ys.size, 'the marked block vanished'
        x0, _y0, _x1, y1 = bounds
        return (x0 + (xs.mean() + 0.5) * res, y1 - (ys.mean() + 0.5) * res)

    def test_no_transform_leaves_it_alone(self, gs):
        img = self.marked()
        out, bounds = gs.place_overlay(img, 0.5, -3.0, -4.0, None)
        assert out is img
        assert bounds == pytest.approx((-3.0, -4.0, 7.0, 6.0))

    def test_pure_translation_only_moves_the_bounds(self, gs):
        img = self.marked()
        out, bounds = gs.place_overlay(img, 0.5, 0.0, 0.0, tf_for(tx=2.0, ty=1.0))
        assert np.array_equal(out, img), 'pixels moved for a plain shift'
        assert bounds == pytest.approx((2.0, 1.0, 12.0, 11.0))

    def test_a_quarter_turn_moves_the_pixels_correctly(self, gs):
        """The test that pins the sign. Compared against apply_transform_2d,
        which the tests above check independently."""
        res, ox, oy = 0.5, -3.0, -4.0
        img = self.marked()
        t = tf_for(yaw=math.pi / 2, tx=1.0, ty=-2.0)
        before = self.centre_in_grid(img, res, ox, oy, 2, 5, 3, 6)
        expect = gs.apply_transform_2d(t, *before)
        out, bounds = gs.place_overlay(img, res, ox, oy, t)
        got = self.centre_in_output(out, bounds, res)
        assert got == pytest.approx(expect, abs=res), f'{got} vs {expect}'

    def test_the_real_small_angle(self, gs):
        res, ox, oy = 0.5, -3.0, -4.0
        img = self.marked()
        t = tf_for(yaw=math.radians(-1.894), tx=-0.438, ty=-0.355)
        before = self.centre_in_grid(img, res, ox, oy, 2, 5, 3, 6)
        expect = gs.apply_transform_2d(t, *before)
        out, bounds = gs.place_overlay(img, res, ox, oy, t)
        got = self.centre_in_output(out, bounds, res)
        assert got == pytest.approx(expect, abs=res), f'{got} vs {expect}'

    def test_ground_beyond_the_grid_stays_transparent(self, gs):
        """A rotated grid does not fill its own bounding box; the corners
        must not be painted as searched ground."""
        img = np.full((20, 20, 4), 255, dtype=np.uint8)
        out, _bounds = gs.place_overlay(img, 0.5, 0.0, 0.0,
                                        tf_for(yaw=math.radians(30)))
        assert out[0, 0, 3] == 0, 'a corner outside the grid was painted'
        assert out[out.shape[0] // 2, out.shape[1] // 2, 3] == 255


class TestTheDrawnAreaIsRotatedToo:
    """The one with a functional effect rather than a visual one.

    The operator draws in the MAP frame; the sweep is planned in ODOM. With
    the twist dropped, the ground actually searched came out rotated from the
    box drawn - about a metre out at the corners of a 30 m box. The overlay
    fixes make the picture honest; this one makes the drone search the right
    ground.
    """

    def fake(self, sn, tf):
        from types import SimpleNamespace
        node = SimpleNamespace(
            pending_area=None, polygon=[], area_set=False, recalled=True,
            area_since_confirm=False, goal_handle=None, busy=False,
            _goal_superseded=False, planned=True, replans=3,
            plan_reachable_cells=7, map_replans=2, looked_from=[(1.0, 1.0)],
            edge_checked=[(2.0, 2.0)], fruitless_vantages=4,
            map_frame='parrot1_odom')
        node.tf_buffer = SimpleNamespace(
            lookup_transform=lambda *a, **k: SimpleNamespace(transform=tf))
        node.get_logger = lambda: SimpleNamespace(
            info=lambda *a, **k: None, warn=lambda *a, **k: None,
            error=lambda *a, **k: None)
        node._rebuild_coverage_grid = lambda: None
        node._start_pass = lambda **k: None
        node._set_status = lambda *a, **k: None
        node._apply_area_request = lambda m: sn.SearchNode._apply_area_request(
            node, m)
        return node

    def area(self, *corners):
        from geometry_msgs.msg import Point32, PolygonStamped
        msg = PolygonStamped()
        msg.header.frame_id = 'parrot1_map'
        for x, y in corners:
            msg.polygon.points.append(Point32(x=float(x), y=float(y), z=0.0))
        return msg

    def test_a_quarter_turn_turns_the_box(self, sn):
        tf = tf_for(yaw=math.pi / 2)
        node = self.fake(sn, tf)
        sn.SearchNode._apply_area_request(
            node, self.area((10.0, 0.0), (20.0, 0.0), (20.0, 10.0)))
        assert node.polygon[0] == pytest.approx((0.0, 10.0), abs=1e-9), \
            'the drawn area was shifted but not turned'
        assert node.area_set is True

    def test_the_real_angle_moves_a_far_corner_about_a_metre(self, sn, gs):
        """Quantifies what was wrong: at 30 m the corner was ~1 m out."""
        tf = tf_for(yaw=math.radians(-1.894), tx=-0.438, ty=-0.355)
        node = self.fake(sn, tf)
        corner = (30.0, 0.0)
        sn.SearchNode._apply_area_request(node, self.area(
            (0.0, 0.0), corner, (30.0, 30.0)))
        right = node.polygon[1]
        wrong = (corner[0] + tf.translation.x, corner[1] + tf.translation.y)
        off = math.hypot(right[0] - wrong[0], right[1] - wrong[1])
        assert off == pytest.approx(0.99, abs=0.05), off

    def test_a_bad_polygon_is_still_refused_not_crashed(self, sn):
        """The safety net must stay in front of this."""
        node = self.fake(sn, tf_for())
        node.area_set = True
        sn.SearchNode.on_area_request(node, self.area((0.0, 0.0), (1.0, 1.0)))
        assert node.area_set is True, 'a two-corner area should be ignored'
