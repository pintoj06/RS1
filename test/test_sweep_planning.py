"""Lawnmower planning in search_node.py: the pure geometry functions."""

import math

import pytest


def legs(waypoints):
    """Waypoints come in (start, end) pairs, one pair per straight leg."""
    return [(waypoints[i], waypoints[i + 1]) for i in range(0, len(waypoints), 2)]


def distance_to_segment(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def sample_inside(polygon, point_in_polygon, step=0.25):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    points = []
    x = min(xs)
    while x <= max(xs):
        y = min(ys)
        while y <= max(ys):
            if point_in_polygon(x, y, polygon):
                points.append((x, y))
            y += step
        x += step
    return points


def distance_to_boundary(p, polygon):
    n = len(polygon)
    return min(distance_to_segment(p, polygon[i], polygon[(i + 1) % n])
               for i in range(n))


def assert_full_coverage(search_node, polygon, spacing, corner_margin=0.0):
    """Every point inside the shape must be within half a row spacing of a
    leg - the condition the planner's docstring promises for full coverage.

    corner_margin skips points that close to the boundary. A sharp corner
    narrower than the footprint can never be reached by any lawnmower, so
    shapes with acute angles are only checked away from the edge.
    """
    waypoints = search_node.make_sweep_polygon(polygon, spacing)
    assert waypoints, 'planner returned no waypoints'
    segments = legs(waypoints)
    pip = search_node.SearchNode.point_in_polygon
    for p in sample_inside(polygon, pip):
        if distance_to_boundary(p, polygon) < corner_margin:
            continue
        nearest = min(distance_to_segment(p, a, b) for a, b in segments)
        # 1e-3 allows for the planner nudging rows off vertex heights.
        assert nearest <= spacing / 2 + 1e-3, f'{p} is {nearest:.2f} m from any leg'


SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
L_SHAPE = [(0, 0), (20, 0), (20, 8), (8, 8), (8, 20), (0, 20)]


class TestRotatePoints:
    def test_quarter_turn(self, search_node):
        (x, y), = search_node.rotate_points([(1.0, 0.0)], math.pi / 2)
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(1.0)

    def test_round_trip(self, search_node):
        points = [(3.0, -2.0), (0.5, 7.25)]
        back = search_node.rotate_points(
            search_node.rotate_points(points, 0.7), -0.7)
        for (x0, y0), (x1, y1) in zip(points, back):
            assert x1 == pytest.approx(x0)
            assert y1 == pytest.approx(y0)


class TestBestSweepAngle:
    def test_wide_rectangle_sweeps_along_x(self, search_node):
        angle = search_node.best_sweep_angle([(0, 0), (40, 0), (40, 5), (0, 5)])
        # Parallel to the x axis, either direction.
        assert math.sin(angle) == pytest.approx(0.0, abs=1e-9)

    def test_tall_rectangle_sweeps_along_y(self, search_node):
        angle = search_node.best_sweep_angle([(0, 0), (5, 0), (5, 40), (0, 40)])
        assert math.cos(angle) == pytest.approx(0.0, abs=1e-9)

    def test_diagonal_strip_follows_its_edge(self, search_node):
        # A long thin strip at 45 degrees.
        strip = search_node.rotate_points([(0, 0), (30, 0), (30, 3), (0, 3)],
                                          math.pi / 4)
        angle = search_node.best_sweep_angle(strip)
        assert abs(math.sin(2 * (angle - math.pi / 4))) == pytest.approx(0.0, abs=1e-9)


class TestMakeSweepPolygon:
    @pytest.mark.parametrize('polygon, spacing', [
        ([(0, 0), (1, 0)], 2.0),     # fewer than three vertices
        (SQUARE, 0.0),
        (SQUARE, -1.0),
        (SQUARE, None),
    ])
    def test_degenerate_input_gives_no_plan(self, search_node, polygon, spacing):
        assert search_node.make_sweep_polygon(polygon, spacing) == []

    def test_square_row_count_and_snaking(self, search_node):
        waypoints = search_node.make_sweep_polygon(SQUARE, 4.0, sweep_angle=0.0)
        # ceil(10 / 4) = 3 rows, two waypoints each.
        assert len(waypoints) == 6
        ys = [round(y, 4) for _, y in waypoints]
        assert ys[0] == ys[1] and ys[2] == ys[3] and ys[4] == ys[5]
        # Rows evenly spread with a half-step margin: step = 10/3.
        step = 10 / 3
        assert waypoints[0][1] == pytest.approx(step / 2, abs=1e-4)
        assert waypoints[4][1] == pytest.approx(10 - step / 2, abs=1e-4)
        # Alternate direction each row: each leg starts where the last ended.
        assert waypoints[0][0] == pytest.approx(0) and waypoints[1][0] == pytest.approx(10)
        assert waypoints[2][0] == pytest.approx(10) and waypoints[3][0] == pytest.approx(0)
        assert waypoints[4][0] == pytest.approx(0) and waypoints[5][0] == pytest.approx(10)

    def test_real_step_never_exceeds_requested_spacing(self, search_node):
        for spacing in (1.5, 3.0, 4.0, 7.0):
            waypoints = search_node.make_sweep_polygon(SQUARE, spacing, sweep_angle=0.0)
            row_ys = sorted({round(y, 6) for _, y in waypoints})
            gaps = [b - a for a, b in zip(row_ys, row_ys[1:])]
            assert all(g <= spacing + 1e-9 for g in gaps)

    def test_waypoints_stay_inside_the_area(self, search_node):
        pip = search_node.SearchNode.point_in_polygon
        for polygon in (SQUARE, L_SHAPE):
            for x, y in search_node.make_sweep_polygon(polygon, 3.0):
                # Leg ends sit exactly on the edge, so on-boundary counts.
                assert pip(x, y, polygon) or distance_to_boundary((x, y), polygon) < 1e-6

    @pytest.mark.parametrize('polygon, spacing', [
        (SQUARE, 4.0),
        (L_SHAPE, 4.0),
        (L_SHAPE, 5.0),
        ([(0, 0), (20, 0), (25, 10), (5, 10)], 3.0),                # parallelogram
    ])
    def test_covers_the_whole_shape(self, search_node, polygon, spacing):
        assert_full_coverage(search_node, polygon, spacing)

    @pytest.mark.parametrize('polygon, spacing', [
        ([(0, 0), (12, 0), (6, 9)], 3.0),                           # triangle
        ([(0, 0), (10, 0), (10, 10), (5, 4), (0, 10)], 2.5),        # concave notch
    ])
    def test_covers_shapes_with_sharp_corners(self, search_node, polygon, spacing):
        assert_full_coverage(search_node, polygon, spacing, corner_margin=spacing)

    def test_l_shape_corner_row_is_full_width(self, search_node):
        # Regression: a row landing exactly on the L's inner edge height used
        # to be clipped to the narrow arm, leaving the corner unsearched.
        # Spacing 16/2 puts the row boundary on y = 8 with sweep along x.
        polygon = [(0, 0), (20, 0), (20, 8), (8, 8), (8, 16), (0, 16)]
        waypoints = search_node.make_sweep_polygon(polygon, 8.0, sweep_angle=0.0)
        widths = [abs(b[0] - a[0]) for a, b in legs(waypoints)]
        assert max(widths) == pytest.approx(20.0, abs=1e-3)
        assert_full_coverage(search_node, polygon, 8.0)

    def test_concave_row_splits_into_two_legs(self, search_node):
        # A U shape: rows through the arms must not cross the gap between them.
        u_shape = [(0, 0), (12, 0), (12, 10), (8, 10), (8, 4), (4, 4), (4, 10), (0, 10)]
        waypoints = search_node.make_sweep_polygon(u_shape, 2.0, sweep_angle=0.0)
        for (ax, ay), (bx, by) in legs(waypoints):
            if ay > 4.0:
                lo, hi = sorted((ax, bx))
                assert hi <= 4.0 + 1e-6 or lo >= 8.0 - 1e-6, 'leg crosses the gap'


class TestObstacleAvoidance:
    # Rows on a 2 m spacing over 0..20 sit at y = 1, 3, ..., 19. The radius
    # keeps the rock's edge off those lines: a row that only grazes it sees a
    # chord narrower than the 0.1 m probe step, which the planner can miss
    # (Nav2's inflation covers that case in flight).
    RADIUS = 2.5

    @classmethod
    def rock(cls, x, y):
        return math.hypot(x - 10, y - 10) < cls.RADIUS

    def test_no_leg_passes_through_obstacle(self, search_node):
        # With no clearance a leg ends at the first blocked probe, up to one
        # probe step (0.1 m) inside the rock; the node's default 1 m clearance
        # is what keeps it out, so test with that.
        area = [(0, 0), (20, 0), (20, 20), (0, 20)]
        waypoints = search_node.make_sweep_polygon(
            area, 2.0, sweep_angle=0.0, is_blocked=self.rock, clearance=1.0)
        assert waypoints
        for (ax, ay), (bx, by) in legs(waypoints):
            for k in range(101):
                t = k / 100
                assert not self.rock(ax + t * (bx - ax), ay + t * (by - ay))

    def test_clearance_keeps_back_from_obstacle(self, search_node):
        area = [(0, 0), (20, 0), (20, 20), (0, 20)]
        clearance = 1.0
        waypoints = search_node.make_sweep_polygon(
            area, 2.0, sweep_angle=0.0, is_blocked=self.rock, clearance=clearance)
        for x, y in waypoints:
            # The probe is 0.1 m, so allow that much slack on top.
            assert math.hypot(x - 10, y - 10) >= self.RADIUS + clearance - 0.15 \
                or x in (0, 20)

    def test_min_leg_drops_short_pieces(self, search_node):
        # A wall at x in [1, 19] leaves 1 m stubs at each end of every row.
        def wall(x, _y):
            return 1.0 < x < 19.0
        area = [(0, 0), (20, 0), (20, 6), (0, 6)]
        kept = search_node.make_sweep_polygon(
            area, 2.0, sweep_angle=0.0, is_blocked=wall, min_leg=0.0)
        dropped = search_node.make_sweep_polygon(
            area, 2.0, sweep_angle=0.0, is_blocked=wall, min_leg=2.0)
        assert kept
        assert dropped == []

    def test_fully_blocked_area_gives_no_plan(self, search_node):
        plan = search_node.make_sweep_polygon(
            SQUARE, 2.0, is_blocked=lambda x, y: True)
        assert plan == []


class TestMakeSweepRectangle:
    def test_matches_polygon_version(self, search_node):
        rect = search_node.make_sweep(-8, 8, -8, 8, 4.0)
        poly = search_node.make_sweep_polygon(
            [(-8, -8), (8, -8), (8, 8), (-8, 8)], 4.0)
        assert rect == poly
        assert len(rect) == 8      # 16 m / 4 m = 4 rows


class TestPointInPolygon:
    @pytest.mark.parametrize('point, expected', [
        ((5, 5), True),
        ((15, 5), True),       # lower arm
        ((5, 15), True),       # upper arm
        ((15, 15), False),     # the missing corner of the L
        ((-1, 5), False),
        ((25, 25), False),
    ])
    def test_l_shape(self, search_node, point, expected):
        assert search_node.SearchNode.point_in_polygon(*point, L_SHAPE) is expected
