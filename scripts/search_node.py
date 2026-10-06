#!/usr/bin/env python3


"Fly a lawnmower search patter over a rectangle, using NAV2"

import math

import numpy as np

import rclpy
from action_msgs.msg import GoalStatus
# Point is a plain (x, y, z) location. Markers are built out of lists of these.
# PolygonStamped carries the search area the operator draws in the ground
# station: a list of corners plus the frame they were measured in.
from geometry_msgs.msg import Point, PolygonStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
# OccupancyGrid is the standard ROS "2D grid of values" message. RViz draws it
# as a coloured overlay, so we use it to show which ground has been searched.
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
# CameraInfo carries the camera's focal length and image size, from which the
# real field of view can be worked out - no need to copy numbers from the xacro.
from sensor_msgs.msg import CameraInfo
# Duration is used to put a time limit on TF lookups so they can't block.
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
# ColorRGBA lets us colour each waypoint individually (done vs still to do).
# Empty carries no data - it is just a "this happened" signal for the courier.
from std_msgs.msg import Bool, ColorRGBA, Empty
# tf2_ros answers "where is the drone right now?" by reading the transform tree.
import tf2_ros
# Marker = one drawing in RViz. MarkerArray = several sent together.
from visualization_msgs.msg import Marker, MarkerArray

def rotate_points(points, angle):
    """Rotate a list of (x, y) points about the origin by angle radians."""
    c, s = math.cos(angle), math.sin(angle)
    return [(x * c - y * s, x * s + y * c) for x, y in points]


def best_sweep_angle(polygon):
    """Pick the sweep direction that needs the fewest rows.

    Turning is the expensive part of a coverage pattern, so we want long legs
    and few of them. That means sweeping along the shape's LONG axis.

    For a polygon, the direction that minimises the number of rows is always
    parallel to one of its edges, so we simply try each edge in turn and keep
    whichever gives the smallest extent measured across the sweep direction.
    A tall narrow area therefore sweeps lengthways instead of making dozens of
    short hops across its width.
    """
    best_angle, best_extent = 0.0, float('inf')
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        edge_angle = math.atan2(y2 - y1, x2 - x1)
        # Rotate so this edge is horizontal, then measure the height needed.
        rotated = rotate_points(polygon, -edge_angle)
        extent = max(p[1] for p in rotated) - min(p[1] for p in rotated)
        if extent < best_extent:
            best_extent, best_angle = extent, edge_angle
    return best_angle


def make_sweep_polygon(polygon, spacing, sweep_angle=None):
    """Lawnmower pattern covering ANY simple polygon.

    This is boustrophedon ("as the ox ploughs") coverage. The method:

      1. Choose a sweep direction - by default the one needing fewest rows.
      2. Rotate the polygon so those sweep lines become horizontal, which
         makes the geometry a simple scanline problem.
      3. For each row, find where the horizontal line crosses the polygon's
         edges, sort those crossings, and pair them up. Because a line
         entering a shape must also leave it, consecutive pairs are exactly
         the parts INSIDE the polygon. This is why concave shapes work too:
         an L-shape simply produces more than one pair on some rows.
      4. Alternate the direction of travel each row, so the drone ends each
         leg next to the start of the next one instead of flying back.
      5. Rotate the resulting waypoints back into world coordinates.

    Rows are spread evenly across the shape with a half-step margin at each
    edge, so the real step is always <= the requested spacing. Coverage is
    complete when the camera's footprint radius is at least half that step.

    polygon: list of (x, y) vertices, in order around the shape.
    Returns a flat list of (x, y) waypoints to visit in order.
    """
    if len(polygon) < 3 or spacing is None or spacing <= 0:
        return []

    if sweep_angle is None:
        sweep_angle = best_sweep_angle(polygon)

    rotated = rotate_points(polygon, -sweep_angle)
    min_y = min(p[1] for p in rotated)
    max_y = max(p[1] for p in rotated)
    extent = max_y - min_y

    # Spread the rows evenly rather than stepping from one edge, so the far
    # edge is always reached instead of being left short.
    n_rows = max(1, math.ceil(extent / spacing))
    step = extent / n_rows

    def row_intervals(y):
        """The spans of this horizontal line that lie inside the polygon."""
        crossings = []
        for i in range(len(rotated)):
            x1, y1 = rotated[i]
            x2, y2 = rotated[(i + 1) % len(rotated)]
            # Half-open test: counts a vertex once, not twice. Note this
            # deliberately ignores an edge that FINISHES at exactly y, which is
            # what makes the nudge below necessary.
            if (y1 <= y < y2) or (y2 <= y < y1):
                t = (y - y1) / (y2 - y1)
                crossings.append(x1 + t * (x2 - x1))
        crossings.sort()
        # Pair them: crossing 0->1 is inside, 1->2 is outside, 2->3 inside...
        spans = []
        for j in range(0, len(crossings) - 1, 2):
            left, right = crossings[j], crossings[j + 1]
            if right - left >= 1e-6:           # skip slivers at a vertex
                spans.append((left, right))
        return spans

    waypoints = []
    flip = False
    for row in range(n_rows):
        y = min_y + (row + 0.5) * step

        # A scanline landing exactly on a corner height is a degenerate case.
        # The crossing test above counts an edge's lower end but not its upper
        # one (otherwise a corner counts twice and the inside/outside pairing
        # inverts), so an edge finishing exactly at this height is missed and
        # the row comes out short.
        #
        # This bit us on an L-shape: its middle row landed on y = 0, exactly
        # where the L's inner edge sits, and the row was clipped to half width,
        # leaving the corner of the lower arm unsearched.
        #
        # So shift the line a hair off the corner - but which way matters. At a
        # junction between a wide part and a narrow part, one side gives a short
        # leg and the other a full-width one, and the full-width leg is the one
        # that covers both (the camera footprint reaches above AND below the
        # line). Rather than guess, try both and keep whichever sweeps more.
        nudge = step * 1e-6
        below, above = row_intervals(y - nudge), row_intervals(y + nudge)
        if sum(r - l for l, r in above) > sum(r - l for l, r in below):
            y, spans = y + nudge, above
        else:
            y, spans = y - nudge, below

        for left, right in spans:
            if flip:
                waypoints.extend([(right, y), (left, y)])
            else:
                waypoints.extend([(left, y), (right, y)])
        flip = not flip

    return rotate_points(waypoints, sweep_angle)


def make_sweep(min_x, max_x, min_y, max_y, spacing):
    """Lawnmower pattern over a rectangle.

    Kept so existing launch arguments and parameters still work. It just
    expresses the rectangle as a polygon and hands it to the general version.
    """
    rectangle = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]
    return make_sweep_polygon(rectangle, spacing)


class SearchNode(Node):
    def __init__(self):
        super().__init__('search_node')

        # Things that can be changed without editing the code
        self.declare_parameter('robot_name', 'parrot1')
        self.declare_parameter('min_x', -4.0)
        self.declare_parameter('max_x', 4.0)
        self.declare_parameter('min_y', -4.0)
        self.declare_parameter('max_y', 4.0)
        self.declare_parameter('return_to_spawn', True)

        # Leg spacing is normally worked out from the live camera footprint, so
        # it does not need setting by hand. Set this to a positive number only
        # to FORCE a fixed spacing, which is useful for testing.
        self.declare_parameter('spacing', 0.0)

        # How much neighbouring camera passes overlap. 0.8 means each leg sits
        # 80% of a footprint width from the last, so consecutive passes share a
        # 20% strip. That margin is what stops small position errors from
        # opening a gap of unseen ground between two legs.
        self.declare_parameter('overlap', 0.8)

        # An arbitrary search polygon, as a flat list: [x1, y1, x2, y2, ...].
        # Leave it empty to use the min_x/max_x/min_y/max_y rectangle instead.
        # This is the STARTING area; the operator can redraw it at any time
        # from the ground station (see on_area_request).
        self.declare_parameter('polygon', [0.0])

        robot_name= self.get_parameter('robot_name').value
        # Goals are stamped in the frame Nav2 plans in. That is now the odom
        # frame, not the SLAM map frame - see global_frame in the nav2 params.
        # Must stay in step with those, or Nav2 will not understand our goals.
        self.map_frame= f'{robot_name}_odom' # e.g. "parrot1_odom", NOT "odom"

        # Build the search shape: an explicit polygon if one was given,
        # otherwise the rectangle from the min/max parameters.
        self.polygon = self._resolve_polygon()

        # The sweep is deliberately NOT built here. Leg spacing comes from how
        # much ground the camera can see, which depends on the field of view -
        # and that arrives over CameraInfo a moment after startup, not yet.
        # plan_sweep() builds the path from tick() instead, on the first tick
        # where the camera has reported itself. That is also where the
        # return-to-spawn waypoint gets appended (see plan_sweep) - doing it
        # here would just be overwritten once plan_sweep() replaces self.points.
        self.points = []
        self.planned = False
        self.spacing = None
        self.home_index = None

        self.index = 0
        self.busy= False
        # Hold = stop sweeping and hover where we are. The ground station sets
        # it when the thermal camera picks up a person, and clears it once the
        # operator has ruled on the contact.
        self.holding = False
        self.goal_handle = None

        self.nav_client= ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Tells other robots (e.g. the courier drone) that the sweep is done.
        # Transient-local + depth 1 means a subscriber that starts up late
        # still gets this message, instead of needing to be listening at the
        # exact moment it is published.
        latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.sweep_complete_pub = self.create_publisher(Empty, 'search_node/sweep_complete', latched_qos)

        self.timer = self.create_timer(1.0, self.tick) # runs once a seocnd

        # Transient local so we still get the current hold state if the ground
        # station published it before this node started.
        hold_qos = QoSProfile(depth=1,
                              reliability=QoSReliabilityPolicy.RELIABLE,
                              durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, 'search/hold', self.on_hold, hold_qos)

        # ------------------------------------------------------------------
        # VISUALISATION - everything below is only for RViz. It does not
        # affect how the drone flies; it just draws what is going on so the
        # search can be seen (and screenshotted) rather than only logged.
        # ------------------------------------------------------------------

        # The drone's own body frame, e.g. "parrot1_base_link". Asking TF for
        # map -> base_link tells us where the drone is on the map.
        self.base_frame = f'{robot_name}_base_link'

        # The buffer stores recent transforms; the listener fills it from the
        # /tf topic in the background. Both are needed for lookups to work.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher for the red breadcrumb trail of where the drone has been.
        self.trail_pub = self.create_publisher(Marker, 'search_trail', 10)

        # One Marker that we keep appending points to, rather than creating a
        # new one each time. Same ns + id every publish means RViz replaces
        # the old drawing instead of stacking copies on top of each other.
        self.trail = Marker()
        self.trail.header.frame_id = self.map_frame  # coordinates are map-frame
        self.trail.ns = 'flight_trail'
        self.trail.id = 0
        self.trail.type = Marker.POINTS   # dots; use LINE_STRIP for a line
        self.trail.action = Marker.ADD
        self.trail.scale.x = 0.2          # POINTS needs BOTH x and y set:
        self.trail.scale.y = 0.2          # they are the dot's width/height
        self.trail.color.r = 1.0          # red
        self.trail.color.g = 0.0
        self.trail.color.b = 0.0
        self.trail.color.a = 1.0          # opacity - defaults to 0 = INVISIBLE
        self.trail.pose.orientation.w = 1.0  # identity rotation (points are absolute)

        # Publisher for the search area, planned route and waypoint markers.
        # A MarkerArray because it carries several separate drawings at once.
        self.area_pub = self.create_publisher(MarkerArray, 'search_area', 10)

        # Two extra timers. Redrawing on a timer (instead of publishing once)
        # means the displays appear correctly no matter when you add them in
        # RViz, and lets the waypoint colours update as progress is made.
        self.trail_timer = self.create_timer(0.5, self.record_trail)
        self.area_timer = self.create_timer(1.0, self.publish_search_area)

        # ------------------------------------------------------------------
        # COVERAGE MAP - which ground the camera has actually looked at.
        #
        # The camera points straight down with a 60 degree horizontal field of
        # view (1.0472 rad, set in parrot.gazebo.xacro) and a 720x480 image,
        # which gives roughly a 42 degree vertical field of view. So at
        # altitude h the ground patch in view is about:
        #
        #     width  = 2h * tan(30 deg) = 1.155 * h
        #     height = 2h * tan(21 deg) = 0.770 * h
        #
        # Rather than track a rotating rectangle, we mark a CIRCLE whose radius
        # is half the SHORTER side. That is independent of which way the drone
        # is facing, and it under-claims coverage rather than over-claiming it,
        # which is the honest direction to be wrong in.
        # ------------------------------------------------------------------
        # Nothing about the camera is hardcoded. The lens properties come from
        # the camera itself over CameraInfo, and the altitude comes from TF.
        # Change the drone's spawn height, move the camera, or swap the lens,
        # and the coverage maths follows automatically.
        #
        # The parameters below are only FALLBACKS, used in the first moments
        # before CameraInfo/TF have arrived. They match the current xacro.
        self.declare_parameter('fallback_camera_hfov', 1.0472)   # radians
        self.declare_parameter('fallback_image_width', 720)
        self.declare_parameter('fallback_image_height', 480)
        # Height the drone flies at, in metres above the ground. Together with
        # the camera's field of view this is what sets how wide a strip of ground
        # the camera covers, and therefore how far apart the sweep legs go.
        #
        # It is a parameter rather than a measurement because nothing in flight
        # changes it: Nav2 is a 2D planner and never commands height, so the
        # drone holds whatever altitude it was spawned at. Keep this in step with
        # the Parrot's spawn z in 41068_ignition.launch.py.
        self.declare_parameter('flight_altitude', 10.0)           # metres
        self.declare_parameter('coverage_resolution', 0.5)       # metres per cell

        self.camera_frame = f'{robot_name}_camera_link'

        # Filled in by the CameraInfo callback. None until the first message.
        self.cam_fx = None
        self.cam_fy = None
        self.cam_width = None
        self.cam_height = None
        self.logged_camera_info = False

        self.create_subscription(
            CameraInfo, 'camera/camera_info', self._camera_info_callback, 1
        )

        self.coverage_resolution = float(self.get_parameter('coverage_resolution').value)

        # Sized to the search polygon. Built here for the starting area and
        # rebuilt whenever the operator redraws it (see on_area_request).
        self.last_coverage_log = 0.0
        self._rebuild_coverage_grid()

        self.coverage_pub = self.create_publisher(OccupancyGrid, 'search_coverage', 10)
        self.coverage_timer = self.create_timer(0.5, self.update_coverage)

        # Report the footprint periodically rather than once at startup: at
        # startup neither CameraInfo nor TF has arrived, so the numbers would
        # be the fallbacks rather than the real ones. This also means the log
        # follows the drone if its altitude changes mid-flight.
        self.footprint_timer = self.create_timer(10.0, self._log_footprint)

        # Live search area from the ground station: the operator drags a box on
        # the map and this replaces whatever the launch parameter set.
        #
        # Registered LAST in __init__ on purpose. It is transient-local, so a
        # message already waiting on the topic is delivered the moment we
        # subscribe - and the callback touches the TF buffer, the coverage grid
        # and the planner state, all of which are built above.
        area_qos = QoSProfile(depth=1,
                              reliability=QoSReliabilityPolicy.RELIABLE,
                              durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            PolygonStamped, 'search_area_request', self.on_area_request, area_qos)

    def on_area_request(self, msg):
        """Operator drew a new search area in the ground station.

        Rather than building the path here, this resets self.planned so
        plan_sweep() redoes it on the next tick - that way the new area still
        gets footprint-derived spacing, the pre-flight coverage check and the
        return-to-spawn waypoint, instead of a bare sweep that skips all three.
        """
        pts = [(p.x, p.y) for p in msg.polygon.points]
        if len(pts) < 3:
            self.get_logger().warn(
                f'Search area needs at least three corners, got {len(pts)}.')
            return

        # The ground station stamps the polygon in the scout's MAP frame; this
        # node plans in odom. Translation only: the two frames differ by an
        # offset with no rotation here, so shifting the corners is enough.
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, msg.header.frame_id,
                rclpy.time.Time(), timeout=Duration(seconds=0.5))
        except Exception as exc:
            self.get_logger().warn(f'Cannot transform search area: {exc}')
            return

        dx = tf.transform.translation.x
        dy = tf.transform.translation.y
        self.polygon = [(x + dx, y + dy) for x, y in pts]

        # The coverage grid is sized to the polygon, so it has to be rebuilt -
        # otherwise progress would be measured against the old shape.
        self._rebuild_coverage_grid()

        # Abandon the current leg. The operator has new information, so
        # finishing a waypoint from the old area is wasted flying.
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.busy = False
        self.index = 0
        self.home_index = None
        self.planned = False

        self.get_logger().info(
            f'Operator set a new {len(self.polygon)}-sided search area; '
            f'replanning on the next tick.')

    def _rebuild_coverage_grid(self):
        """Size and clear the coverage grid for the current polygon.

        Pads a little beyond the search area so edge passes still land inside
        the grid, and recomputes which cells are actually inside the shape -
        the percentage is measured against those only, or a non-rectangular
        area could never reach 100%.
        """
        pad = 5.0
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        self.cov_origin_x = min(xs) - pad
        self.cov_origin_y = min(ys) - pad
        span_x = (max(xs) - min(xs)) + 2 * pad
        span_y = (max(ys) - min(ys)) + 2 * pad
        self.cov_cols = max(1, int(span_x / self.coverage_resolution))
        self.cov_rows = max(1, int(span_y / self.coverage_resolution))

        # 0 = not yet searched, 100 = searched. Same convention OccupancyGrid
        # uses, so it can be published directly.
        self.coverage = np.zeros((self.cov_rows, self.cov_cols), dtype=np.int8)
        self.inside_mask = self._build_inside_mask()
        self.cells_to_search = int(np.count_nonzero(self.inside_mask))
        self.last_coverage_log = 0.0

    def _log_footprint(self):
        """Log what the camera can currently see, and the spacing in use.

        While the search is on hold this says so instead. Otherwise the only
        sign the drone has parked itself is one HOLD line that scrolls away,
        and this message keeps reporting spacing as though it were still
        flying - which reads like nothing is wrong.
        """
        if not self.planned:
            return
        if self.holding:
            self.get_logger().warn(
                f'STILL ON HOLD at waypoint {self.index + 1}/{len(self.points)} '
                f'- hovering, waiting for the operator to confirm or dismiss '
                f'the contact.'
            )
            return
        self.get_logger().info(
            f'Camera sees {self._footprint_radius() * 2:.1f} m across at '
            f'{self._camera_altitude():.1f} m above ground. '
            f'Leg spacing in use: {self.spacing:.1f} m.'
        )

    def _predicted_coverage(self, points, radius, res=0.5):
        """Fraction of the search polygon the camera would see flying `points`.

        This is the planner marking its own homework, before the drone moves.
        We lay a grid over the polygon, fly the proposed path in simulation, and
        count which cells the camera footprint would pass over.

        It exists because spacing alone cannot guarantee coverage. On shapes
        whose width changes sharply along the sweep direction - a cross or a
        plus, say - the outer arms may be reached by only a single sweep row,
        and if that arm is taller than the footprint a strip goes unseen. The
        textbook cure is cell decomposition (splitting the shape and sweeping
        each part with its own rows), which is a much bigger change. Measuring
        the result and tightening the spacing gets the same guarantee.
        """
        if len(points) < 2:
            return 0.0

        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        pad = radius + res
        ox, oy = min(xs) - pad, min(ys) - pad
        cols = max(1, int(((max(xs) - min(xs)) + 2 * pad) / res))
        rows = max(1, int(((max(ys) - min(ys)) + 2 * pad) / res))

        # Cell-centre coordinates as two 2D arrays, so the tests below can be
        # done on the whole grid at once instead of cell by cell.
        gx, gy = np.meshgrid(ox + (np.arange(cols) + 0.5) * res,
                            oy + (np.arange(rows) + 0.5) * res)

        # Which cells are inside the polygon - ray casting, vectorised. Each
        # edge flips the cells whose rightward ray crosses it; an odd number of
        # crossings means inside.
        inside = np.zeros((rows, cols), dtype=bool)
        n = len(self.polygon)
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]
            if y1 == y2:
                continue                      # horizontal edges never cross
            inside ^= (((y1 > gy) != (y2 > gy))
                       & (gx < x1 + (gy - y1) / (y2 - y1) * (x2 - x1)))

        total = int(inside.sum())
        if total == 0:
            return 0.0

        # Fly the path and mark everything the footprint circle passes over.
        seen = np.zeros((rows, cols), dtype=bool)
        sample = max(res * 0.5, 0.1)          # how finely to step along a leg
        r2 = radius * radius
        for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
            legs = max(1, int(math.hypot(bx - ax, by - ay) / sample))
            for k in range(legs + 1):
                t = k / legs
                px, py = ax + t * (bx - ax), ay + t * (by - ay)
                seen |= (gx - px) ** 2 + (gy - py) ** 2 <= r2

        return float((seen & inside).sum()) / total

    def plan_sweep(self):
        """Build the sweep path, with leg spacing set by the camera footprint.

        This is the point of the whole search pattern: the drone should never
        have to be told how far apart to fly its legs. It can see how wide a
        strip of ground its own camera covers, so it works the spacing out from
        that and gaps become impossible by construction rather than something
        we check for afterwards.

        Called from tick() rather than __init__ because it needs live sensor
        data. Returns True once the path has been built.
        """
        if self.cam_fx is None:
            self.get_logger().info('Waiting for camera info before planning the sweep...')
            return False
        footprint = self._footprint_radius() * 2.0      # metres across on the ground

        # A positive 'spacing' parameter forces a fixed value. Otherwise the
        # spacing is the footprint shrunk by the overlap factor, so neighbouring
        # camera passes are guaranteed to touch instead of leaving a strip of
        # ground nobody looked at.
        forced = float(self.get_parameter('spacing').value)
        overlap = float(self.get_parameter('overlap').value)
        if forced > 0.0:
            self.spacing = forced
            source = 'forced by the spacing parameter'
            self.points = make_sweep_polygon(self.polygon, self.spacing)
            predicted = self._predicted_coverage(self.points, self._footprint_radius())
        else:
            source = f'{footprint:.1f} m camera footprint x {overlap:.2f} overlap'
            # Start from the footprint-derived spacing, then CHECK it. Spacing
            # on its own does not guarantee coverage on awkward shapes, so if
            # the check predicts a gap we tighten and try again rather than
            # taking off and finding out afterwards.
            self.spacing = footprint * overlap
            radius = self._footprint_radius()
            for attempt in range(6):
                self.points = make_sweep_polygon(self.polygon, self.spacing)
                predicted = self._predicted_coverage(self.points, radius)
                if predicted >= 0.999:
                    break
                self.get_logger().warn(
                    f'Spacing {self.spacing:.2f} m would leave '
                    f'{100 * (1 - predicted):.1f}% of the area unsearched - '
                    f'tightening.'
                )
                self.spacing *= 0.85

        # (0, 0) in the odom frame is the spawn point (see map_frame comment
        # in __init__), so "return to spawn" is just one more waypoint tacked
        # onto the end of the sweep - no separate flight mode needed. Done
        # here, after self.points has its final value, rather than in
        # __init__ - the retry loop above replaces self.points outright, so
        # appending any earlier would just get overwritten.
        if self.get_parameter('return_to_spawn').value:
            self.points.append((0.0, 0.0))
            self.home_index = len(self.points) - 1

        self.planned = True

        sweep_deg = math.degrees(best_sweep_angle(self.polygon))
        self.get_logger().info(
            f'Search area: {len(self.polygon)}-sided polygon, sweeping along '
            f'{sweep_deg:.0f} degrees (chosen to minimise turns).'
        )
        self.get_logger().info(
            f'Camera sees {footprint:.1f} m across at '
            f'{self._camera_altitude():.1f} m above ground. '
            f'Leg spacing {self.spacing:.1f} m ({source}): '
            f'{len(self.points) // 2} legs, {len(self.points)} waypoints.'
        )
        if predicted >= 0.999:
            self.get_logger().info(
                f'Checked before take-off: this path covers '
                f'{100 * predicted:.1f}% of the search area.'
            )
        else:
            self.get_logger().warn(
                f'Best achievable with this shape is {100 * predicted:.1f}% '
                f'coverage - {100 * (1 - predicted):.1f}% will NOT be searched.'
            )
        return True

    def tick(self):
        """Called every second; sends the next waypoint to Nav2 once it's ready and we're not already flying to one."""
        if self.holding:
            return
        if self.busy: #goal is already in flight
            return
        # Plan on the first tick that has camera data, not in __init__.
        if not self.planned and not self.plan_sweep():
            return
        if self.index>= len(self.points):
            return
        if not self.nav_client.server_is_ready(): #NAv2 sevrer not up yet
            self.get_logger().info('Waiting for Nav2 to be ready...')
            return
        self.send_goal(self.points[self.index])

    def send_goal (self, point):
        """Send one waypoint to Nav2."""
        x,y =point

        #Face the next waypoint, so the camera looks where we're going
        yaw= 0.0

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        # An angle as a quaternion. For a flat 2D turn, only z and w matter.
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.busy = True
        self.get_logger().info (
            f'Waypoint {self.index +1}/{len(self.points)}: flying to ({x:.1f}, {y:.1f})'
        )

        # Stage 1: send it. goal_response() gets called wehen Nav2 answers.
        send_future= self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self.goal_response)



    def goal_response(self,future):
        """Stage 2: Nav2 has accepted or rejected the goal. """
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected that goal. Skipping it.')
            self.next_waypoint()
            return

        self.goal_handle = goal_handle
        if self.holding:
            # Hold arrived while this goal was still being accepted.
            goal_handle.cancel_goal_async()

        #Stage 3: ask to be told when its finished flying.
        result_future= goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_finished)

    def goal_finished(self,future):
        """Stage 3: Nav2 has arrived, failed or given up"""
        self.goal_handle = None
        status = future.result().status
        if self.holding or status == GoalStatus.STATUS_CANCELED:
            # Cancelled by a hold, or by the operator redrawing the search
            # area. Don't advance: on a hold, resuming re-flies the same
            # waypoint instead of skipping it, and on a redraw the index has
            # already been reset to the start of the new pattern.
            self.busy = False
            return
        if future.result().status ==GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Arrived. ')
        else:
            self.get_logger().warn('Could not reach that one. Skipping it. ')
        self.next_waypoint()

    def on_hold(self, msg):
        if msg.data == self.holding:
            return
        self.holding = msg.data
        if self.holding:
            self.get_logger().warn(
                f'HOLD: contact reported, stopping search at waypoint '
                f'{self.index + 1}/{len(self.points)}. Hovering until the '
                f'operator rules on it.'
            )
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
        else:
            self.get_logger().info(
                f'Hold released, resuming search - re-flying waypoint '
                f'{self.index + 1}/{len(self.points)}.'
            )
            # Don't wait up to a second for the timer; the ground station only
            # gives the drone a short window to move before it checks again.
            self.tick()

    def next_waypoint(self):
        completed_index = self.index
        self.index +=1
        self.busy = False
        if self.index >=len(self.points):
            if self.home_index is not None and completed_index == self.home_index:
                self.get_logger().info('Sweep complete and returned to spawn.')
            else:
                self.get_logger().info('Sweep complete. ')
            self.sweep_complete_pub.publish(Empty())

    # ----------------------------------------------------------------------
    # VISUALISATION METHODS
    # ----------------------------------------------------------------------

    def record_trail(self):
        """Add the drone's current position to the red trail.

        Runs twice a second. Each call asks TF where the drone is and appends
        that position to a growing list of dots, so afterwards you can see
        exactly where it actually flew (as opposed to where it was told to go).
        """
        try:
            # "Where is base_frame, as seen from map_frame, right now?"
            # rclpy.time.Time() with no argument means "the latest available".
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
        except Exception:
            # TF lookups fail constantly at startup before transforms start
            # flowing. That is normal - just skip this tick and try again.
            return

        # transform.translation is the position part of the transform.
        p = Point()
        p.x = t.transform.translation.x
        p.y = t.transform.translation.y
        p.z = 0.0                       # draw flat on the ground
        self.trail.points.append(p)

        # Stop the list growing forever on a long mission (drops oldest first).
        if len(self.trail.points) > 2000:
            self.trail.points.pop(0)

        self.trail.header.stamp = self.get_clock().now().to_msg()
        self.trail_pub.publish(self.trail)

    def publish_search_area(self):
        """Draw the search rectangle, the planned route, and the waypoints.

        Rebuilt and republished every second so the waypoint colours update
        as the sweep progresses.
        """
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        def new_marker(marker_id, name, marker_type, size):
            """Small helper - fills in the fields every marker needs."""
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = stamp
            m.ns = name                  # ns + id together identify a marker,
            m.id = marker_id             # so reusing them replaces the drawing
            m.type = marker_type
            m.action = Marker.ADD
            m.scale.x = size             # for lines this is the line width;
            m.scale.y = size             # for spheres it is the diameter
            m.scale.z = size
            m.pose.orientation.w = 1.0
            return m

        def point(x, y, z=0.0):
            """Build a Point from plain numbers (markers need float values)."""
            p = Point()
            p.x, p.y, p.z = float(x), float(y), float(z)
            return p

        # 1. GREEN OUTLINE of the search area - the actual polygon, whatever
        #    its shape. A LINE_STRIP joins its points in order, so we repeat
        #    the first vertex at the end to close the loop.
        box = new_marker(0, 'search_area', Marker.LINE_STRIP, 0.15)
        box.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        for x, y in list(self.polygon) + [self.polygon[0]]:
            box.points.append(point(x, y))
        markers.markers.append(box)

        # 2. CYAN LINE showing the planned lawnmower route, in visiting order.
        #    Compare this against the red trail to see how closely the drone
        #    actually followed the intended sweep lines.
        route = new_marker(1, 'planned_route', Marker.LINE_STRIP, 0.08)
        route.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        for x, y in self.points:
            route.points.append(point(x, y))
        markers.markers.append(route)

        # 3. WAYPOINT SPHERES. A SPHERE_LIST draws one sphere per point, and
        #    the parallel 'colors' list lets each one have its own colour -
        #    so completed waypoints turn green while pending ones stay yellow.
        dots = new_marker(2, 'waypoints', Marker.SPHERE_LIST, 0.4)
        dots.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)   # fallback colour
        for i, (x, y) in enumerate(self.points):
            dots.points.append(point(x, y, 0.2))   # lifted slightly off ground
            if i < self.index:
                dots.colors.append(ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0))  # done
            else:
                dots.colors.append(ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0))  # to do
        markers.markers.append(dots)

        self.area_pub.publish(markers)

    # ----------------------------------------------------------------------
    # COVERAGE TRACKING
    # ----------------------------------------------------------------------

    @staticmethod
    def point_in_polygon(x, y, polygon):
        """True if (x, y) lies inside the polygon.

        Ray casting: fire a ray to the right and count how many edges it
        crosses. Odd means inside, even means outside. Works for concave
        shapes as well as convex ones.
        """
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                x_cross = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
                if x < x_cross:
                    inside = not inside
        return inside

    def _build_inside_mask(self):
        """Boolean grid marking which coverage cells fall inside the polygon."""
        mask = np.zeros((self.cov_rows, self.cov_cols), dtype=bool)
        for row in range(self.cov_rows):
            cell_y = self.cov_origin_y + (row + 0.5) * self.coverage_resolution
            for col in range(self.cov_cols):
                cell_x = self.cov_origin_x + (col + 0.5) * self.coverage_resolution
                mask[row, col] = self.point_in_polygon(cell_x, cell_y, self.polygon)
        return mask

    def _resolve_polygon(self):
        """Return the search shape as a list of (x, y) vertices.

        Uses the 'polygon' parameter if it holds a sensible shape, otherwise
        falls back to the min_x/max_x/min_y/max_y rectangle so existing launch
        arguments keep working unchanged.
        """
        flat = list(self.get_parameter('polygon').value or [])

        if len(flat) >= 6 and len(flat) % 2 == 0:
            return [(float(flat[i]), float(flat[i + 1]))
                    for i in range(0, len(flat), 2)]

        if len(flat) > 1:
            self.get_logger().warn(
                f'polygon parameter has {len(flat)} values; it needs an even '
                'count of at least 6 (three x,y pairs). Using the rectangle.'
            )

        min_x = self.get_parameter('min_x').value
        max_x = self.get_parameter('max_x').value
        min_y = self.get_parameter('min_y').value
        max_y = self.get_parameter('max_y').value
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

    def _lookup_robot_xy(self):
        """Return the drone's current (x, y) in the map frame, or None.

        Same TF lookup record_trail does, pulled out so the coverage map can
        use it too. Returns None while TF is not ready yet, which is normal
        for the first few seconds after startup.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
        except Exception:
            return None

        return (t.transform.translation.x, t.transform.translation.y)

    def _camera_info_callback(self, msg: CameraInfo):
        """Store the camera's real lens properties as reported by the camera.

        CameraInfo.k is the 3x3 intrinsic matrix, laid out as a flat list:

            k = [fx,  0, cx,
                  0, fy, cy,
                  0,  0,  1]

        fx and fy are the focal lengths in PIXELS. Combined with the image
        size they give the field of view directly, so we never have to copy
        numbers out of the xacro and keep them in step by hand.
        """
        self.cam_fx = msg.k[0]
        self.cam_fy = msg.k[4]
        self.cam_width = msg.width
        self.cam_height = msg.height

        if not self.logged_camera_info and self.cam_fx:
            self.logged_camera_info = True
            hfov, vfov = self._camera_fov()
            self.get_logger().info(
                f'Camera info received: {self.cam_width}x{self.cam_height}, '
                f'FOV {math.degrees(hfov):.1f} x {math.degrees(vfov):.1f} degrees.'
            )

    def _camera_fov(self):
        """Return (horizontal, vertical) field of view in radians.

        Uses the live CameraInfo when it has arrived, otherwise the fallback
        parameters so the node still works in the first second or two.
        """
        if self.cam_fx and self.cam_fy and self.cam_width and self.cam_height:
            hfov = 2.0 * math.atan(self.cam_width / (2.0 * self.cam_fx))
            vfov = 2.0 * math.atan(self.cam_height / (2.0 * self.cam_fy))
            return hfov, vfov

        hfov = float(self.get_parameter('fallback_camera_hfov').value)
        width = float(self.get_parameter('fallback_image_width').value)
        height = float(self.get_parameter('fallback_image_height').value)
        vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (height / width))
        return hfov, vfov

    def _camera_altitude(self):
        """Height of the camera above the ground, in metres.

        Read from the flight_altitude parameter. It deliberately does NOT come
        from TF: Ignition's odometry publisher is planar, so /parrot1/odometry
        reports z = 0.0 however high the drone flies, and the only z in the
        odom -> camera_link chain is the camera's own 0.2 m mounting offset.
        Reading that gave a 0.2 m "altitude", a camera footprint smaller than
        one grid cell, and a coverage map that never marked anything at all.

        A parameter is the right answer rather than a stopgap: Nav2 is a 2D
        planner and never commands height, so the drone holds its spawn
        altitude for the whole flight. The figure is constant and known.
        """
        return float(self.get_parameter('flight_altitude').value)

    def _footprint_radius(self):
        """Radius on the ground the camera can see, in metres.

        Everything here is derived live: the lens from CameraInfo, the height
        from TF. Raise the drone, move the camera, or change the lens and this
        updates on its own.

        The camera looks straight down, so its footprint is a rectangle. We use
        half the SHORTER side as a circle radius: that is independent of which
        way the drone is facing, and it under-claims coverage rather than
        over-claiming it, which is the honest direction to be wrong in.
        """
        hfov, vfov = self._camera_fov()
        altitude = self._camera_altitude()

        half_width = altitude * math.tan(hfov / 2.0)
        half_height = altitude * math.tan(vfov / 2.0)
        return min(half_width, half_height)

    def update_coverage(self):
        """Mark the ground currently under the camera as searched."""
        pos = self._lookup_robot_xy()
        if pos is None:
            return                       # TF not ready yet

        robot_x, robot_y = pos
        radius = self._footprint_radius()

        # Convert the circle's bounding box into grid cell indices.
        min_col = int((robot_x - radius - self.cov_origin_x) / self.coverage_resolution)
        max_col = int((robot_x + radius - self.cov_origin_x) / self.coverage_resolution)
        min_row = int((robot_y - radius - self.cov_origin_y) / self.coverage_resolution)
        max_row = int((robot_y + radius - self.cov_origin_y) / self.coverage_resolution)

        # Clamp to the grid so we never index outside it.
        min_col = max(0, min_col)
        min_row = max(0, min_row)
        max_col = min(self.cov_cols - 1, max_col)
        max_row = min(self.cov_rows - 1, max_row)

        # Mark every cell whose centre falls inside the circle.
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cell_x = self.cov_origin_x + (col + 0.5) * self.coverage_resolution
                cell_y = self.cov_origin_y + (row + 0.5) * self.coverage_resolution
                if math.hypot(cell_x - robot_x, cell_y - robot_y) <= radius:
                    self.coverage[row, col] = 100

        self._publish_coverage()

    def _publish_coverage(self):
        """Publish the coverage grid, and log the percentage now and then."""
        grid = OccupancyGrid()
        grid.header.frame_id = self.map_frame
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = self.coverage_resolution
        grid.info.width = self.cov_cols
        grid.info.height = self.cov_rows
        grid.info.origin.position.x = self.cov_origin_x
        grid.info.origin.position.y = self.cov_origin_y
        grid.info.origin.orientation.w = 1.0
        # OccupancyGrid data is a flat row-major list, same order as the array.
        grid.data = self.coverage.flatten().tolist()
        self.coverage_pub.publish(grid)

        # Measure progress against cells INSIDE the polygon only. The grid is
        # padded wider than the shape, and for a non-rectangular area many
        # cells in the bounding box were never meant to be searched - counting
        # those would mean the percentage could never reach 100%.
        if self.cells_to_search:
            searched = int(np.count_nonzero((self.coverage > 0) & self.inside_mask))
            percent = 100.0 * searched / self.cells_to_search
            if percent - self.last_coverage_log >= 5.0:
                self.last_coverage_log = percent
                self.get_logger().info(
                    f'Search area coverage: {percent:.0f}% '
                    f'({searched}/{self.cells_to_search} cells)'
                )


def main():
    rclpy.init()
    node = SearchNode()
    rclpy.spin(node)

if __name__ == '__main__':
        main() 