#!/usr/bin/env python3


"Fly a lawnmower search patter over a rectangle, using NAV2"

import math
import traceback

import numpy as np
# Connected-component labelling, used by the reachability checks below. Those
# used to be a hand-written breadth-first flood in Python, which took 1.7 s on
# the live 533x533 costmap - far too slow to run continuously. scipy does the
# same 4-connected job in one C call in about 19 ms, and it is that speed-up
# which makes re-deriving the searchable area every couple of seconds
# affordable instead of a once-per-plan luxury.
from scipy import ndimage

import rclpy
from action_msgs.msg import GoalStatus
# Point is a plain (x, y, z) location. Markers are built out of lists of these.
# PolygonStamped carries the search area the operator draws in the ground
# station: a list of corners plus the frame they were measured in.
from geometry_msgs.msg import (Point, PointStamped, PolygonStamped,
                               PoseStamped)
from nav2_msgs.action import NavigateToPose
# OccupancyGrid is the standard ROS "2D grid of values" message. RViz draws it
# as a coloured overlay, so we use it to show which ground has been searched.
# Path is the standard type for a planned route, so anything that wants to
# draw the sweep (the ground station) can read it without knowing about
# RViz markers.
from nav_msgs.msg import OccupancyGrid, Path
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
from std_msgs.msg import Bool, ColorRGBA, Empty, String
# tf2_ros answers "where is the drone right now?" by reading the transform tree.
import tf2_ros
# Marker = one drawing in RViz. MarkerArray = several sent together.
from visualization_msgs.msg import Marker, MarkerArray
# Boxes from yolo_person_detector.py. Only its edge-of-frame topic is read
# here; contacts are the ground station's job.
from vision_msgs.msg import Detection2DArray

# Values published on the search_coverage grid. An OccupancyGrid is int8 and
# RViz reads 0-100 as occupancy, so these double as sensible colours there.
#
# The point of having four rather than two: "I searched it", "I could not
# reach it" and "I never worked out WHETHER I could reach it" are different
# facts, and the operator most needs to know the third. Collapsing all of
# them to -1 drew unknown ground exactly like empty space off the edge of the
# map. Measured on one flight, of 619 unpainted cells inside the area, 478
# (77%) were unknown rather than obstacle - so the biggest category was the
# one being hidden.
COVER_OUTSIDE = -1      # not part of the search area at all; say nothing
COVER_TO_DO = 0         # in the area, reachable, not swept yet
COVER_UNVERIFIED = 25   # in the area, never confirmed reachable: UNKNOWN
COVER_BLOCKED = 50      # in the area, obstacle or walled off with no way in
COVER_SEARCHED = 100    # the camera has been over it

# What the scout says it is doing, published on search_status as
# "STATE|detail". Without this the operator watches the green path vanish and
# the drone fly off out of the box with no way to tell a finished search from
# a look-around trip from a crash - which is exactly how it read.
ST_IDLE = 'IDLE'                  # no search area drawn yet
ST_PLANNING = 'PLANNING'          # waiting on camera or costmap, or planning
ST_SEARCHING = 'SEARCHING'        # flying sweep legs
ST_LOOKING = 'LOOKING'            # flying to a vantage point to get a view
ST_INVESTIGATING = 'INVESTIGATING'  # off the sweep to check a person at the frame edge
ST_RETURNING = 'RETURNING'        # search done, flying back to base
ST_COMPLETE = 'COMPLETE'          # finished and home
ST_HOLD = 'HOLD'                  # paused on a contact
ST_ESTOP = 'ESTOP'                # operator stop
ST_UNSEARCHABLE = 'UNSEARCHABLE'  # nothing in the area can be flown


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


def make_sweep_polygon(polygon, spacing, sweep_angle=None,
                       is_blocked=None, clearance=0.0, min_leg=0.0):
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

    When is_blocked is given, each row's span is additionally cut wherever
    that callback reports an obstacle, so legs stop at the near edge of the
    rock and pick up again on the far side. Nothing plans a waypoint inside an
    obstacle; flying BETWEEN the pieces is left to Nav2, whose global costmap
    now holds the obstacle and will route around it.

    polygon:    list of (x, y) vertices, in order around the shape.
    is_blocked: optional f(x, y) -> True if that world point is obstructed.
    clearance:  metres to keep back from an obstacle edge.
    min_leg:    drop any resulting piece shorter than this - not worth flying.
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

    def clear_pieces(left, right, y):
        """Cut one span into the stretches with no obstacle on them.

        Walks the span in small steps asking is_blocked about each point.
        Points are rotated back to world coordinates first, because the
        scanline works in the rotated frame but the costmap does not.
        """
        if is_blocked is None:
            return [(left, right)]

        probe = max(0.1, step * 0.05)          # sample finer than a row step
        n_probe = max(2, int((right - left) / probe) + 1)
        c, s_ = math.cos(sweep_angle), math.sin(sweep_angle)

        pieces, run_start = [], None
        for k in range(n_probe + 1):
            x = left + (right - left) * k / n_probe
            wx, wy = x * c - y * s_, x * s_ + y * c      # back to world
            if is_blocked(wx, wy):
                if run_start is not None:
                    pieces.append((run_start, x))
                    run_start = None
            elif run_start is None:
                run_start = x
        if run_start is not None:
            pieces.append((run_start, right))

        # Keep back from the rock, then drop anything too short to be worth it.
        out = []
        for a, b in pieces:
            a2 = a + clearance if a > left + 1e-9 else a
            b2 = b - clearance if b < right - 1e-9 else b
            if b2 - a2 >= max(min_leg, 1e-6):
                out.append((a2, b2))
        return out

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

        # Split each span around obstacles, then visit the pieces in the
        # travel direction of this row so the drone still snakes sensibly.
        pieces = []
        for left, right in spans:
            pieces.extend(clear_pieces(left, right, y))
        if flip:
            pieces.reverse()

        for left, right in pieces:
            if flip:
                waypoints.extend([(right, y), (left, y)])
            else:
                waypoints.extend([(left, y), (right, y)])
        flip = not flip

    return rotate_points(waypoints, sweep_angle)


def pixel_to_ground(u, v, width, height, hfov, altitude, x, y, yaw):
    """Ground position seen at pixel (u, v) of the downward camera.

    Same model as pixel_to_ground in UI/ground_station.py, which was checked
    against known positions on the live feed: the camera looks straight down,
    the top of the image is the drone's forward and the right is its right.
    """
    metres_per_px = 2.0 * altitude * math.tan(hfov / 2.0) / width
    forward = (height / 2.0 - v) * metres_per_px
    left = (width / 2.0 - u) * metres_per_px
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + forward * c - left * s, y + forward * s + left * c)


def distance_to_polygon_edge(x, y, polygon):
    """Shortest distance from (x, y) to the polygon's outline, inside or out."""
    best = math.inf
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        t = 0.0 if length_sq == 0 else max(0.0, min(1.0, (
            (x - x1) * dx + (y - y1) * dy) / length_sq))
        best = min(best, math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return best


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
    # Rejections of one waypoint before it is skipped, a second apart.
    MAX_REJECTIONS = 5

    def __init__(self):
        super().__init__('search_node')
        # Declared in groups rather than in one 330-line run here;
        # __init__ was 415 lines and almost all of it was this.
        self._declare_search_parameters()
        self._declare_obstacle_parameters()
        self._declare_reachability_parameters()

        robot_name= self.get_parameter('robot_name').value
        # Goals are stamped in the frame Nav2 plans in. That is now the odom
        # frame, not the SLAM map frame - see global_frame in the nav2 params.
        # Must stay in step with those, or Nav2 will not understand our goals.
        self.map_frame= f'{robot_name}_odom' # e.g. "parrot1_odom", NOT "odom"

        # Build the search shape: an explicit polygon if one was given,
        # otherwise the rectangle from the min/max parameters.
        self.polygon = self._resolve_polygon()
        # Whether there is an area to search yet. While waiting for the
        # operator, self.polygon still holds the launch-parameter shape so the
        # coverage grid has something to size itself to, but nothing flies,
        # draws or measures coverage against it.
        self.area_set = not self.get_parameter('wait_for_area').value
        if not self.area_set:
            self.get_logger().info(
                'Idle: waiting for the operator to draw a search area '
                'in the ground station.')

        # The sweep is deliberately NOT built here. Leg spacing comes from how
        # much ground the camera can see, which depends on the field of view -
        # and that arrives over CameraInfo a moment after startup, not yet.
        # plan_sweep() builds the path from tick() instead, on the first tick
        # where the camera has reported itself. That is also where the
        # return-to-spawn waypoint gets appended (see plan_sweep) - doing it
        # here would just be overwritten once plan_sweep() replaces self.points.
        # The waypoint sequence itself is set up by the one method that owns
        # it, so startup state and reset state cannot drift apart.
        self._start_pass()
        self.spacing = None
        # The two reachability pictures, refreshed by _review_map() and at
        # every plan. 'confirmed' is what the drone can demonstrably get to
        # across ground the lidar has SEEN to be clear; 'possible' also counts
        # unobserved ground as clear. See _confirmed_from / _reachable_from.
        self.confirmed = None
        self.possible = None
        self._shadow_cache = None
        self._shadow_cache_for = None
        # What has EVER been reachable, and ever confirmed reachable, on the
        # coverage grid. Both floods run from the drone's current position,
        # so ground it has already flown over can drop out of them simply
        # because it has moved on - and then the overlay paints searched
        # ground as unverified and the "cannot be searched" figure climbs on
        # its own. Seen in one run going 9% -> 68% while the drone searched.
        # Ground is added as the map improves and only removed when it turns
        # out to be rock, never because the viewpoint changed.
        self.ever_reachable = None
        self.ever_confirmed = None
        # When plan_sweep() started waiting for the lidar to map enough ground
        # to plan from, or None if it is not waiting. See _too_little_mapped.
        self._map_wait_started = None
        # How much of the area was confirmed reachable when the current path
        # was built, in coverage cells. _review_map() compares the live figure
        # against this to spot ground that has opened up or closed off since.
        self.plan_reachable_cells = None
        self.map_replans = 0
        self._replan_is_map_driven = False
        # Vantage points already used. Unlike the no-go list this all
        # replaced, this records something about the DRONE's own history -
        # where it has been and looked from - rather than a guess about the
        # world, so it cannot go stale in the way "that ground is
        # unreachable" did.
        self.looked_from = []
        # Where edge-of-frame sightings have already been flown over and
        # checked, so the same person-shaped rock is not visited every pass.
        self.edge_checked = []
        # Coverage cells written off as impossible to check: unverified, with
        # no vantage point anywhere that can see them. Kept out of the
        # coverage total, or the percentage would stall short of 100% for ever
        # with no way to tell a rock interior from a genuine gap.
        self.unverifiable = None
        # Progress tracking for the look-around trips, see vantage_patience.
        self.unverified_at_vantage = None
        self.fruitless_vantages = 0
        self.replans = 0            # replans used on the current search area
        self.goal_best_dist = None  # least route left on this goal so far
        self.goal_best_crow = float('inf')   # and the best straight-line
        self.goal_progress_at = None  # when it last got closer (seconds, node clock)
        self.goal_sent_at = None    # when the current goal went out
        self.last_recheck = 0.0     # when the goal was last re-validated
        self.goal_leg_length = None  # how far it was when sent
        # Hold = stop sweeping and hover where we are. The ground station sets
        # it while the RGB camera checks a thermal signature and while a
        # contact waits for the operator's ruling, and clears it after.
        self.holding = False
        # E-Stop = the operator's STOP button on the ground station. Same
        # effect as a hold, but kept separate so nothing that releases a hold
        # (a ruling, RESUME on the contacts page, an IR check timing out) can
        # release it - only the E-Stop's own RESUME does.
        self.estopped = False
        self.goal_handle = None
        # Times in a row Nav2 has rejected the current waypoint (see goal_response).
        self.rejections = 0

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
        # Absolute: one E-Stop stops every drone.
        self.create_subscription(Bool, '/aeroaid/estop', self.on_estop, hold_qos)

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

        # COVERAGE MAP - which ground the camera has actually looked at.
        # The footprint is derived live from CameraInfo and the altitude (see
        # _footprint_radius), so nothing here is hardcoded: the parameters
        # below are only FALLBACKS for the first moments before CameraInfo
        # arrives, and they match the current xacro.
        self.declare_parameter('fallback_camera_hfov', 1.0472)   # radians
        self.declare_parameter('fallback_image_width', 720)
        self.declare_parameter('fallback_image_height', 480)
        # Height the drone flies at, in metres above the ground. Together with
        # the camera's field of view this is what sets how wide a strip of ground
        # the camera covers, and therefore how far apart the sweep legs go.
        #
        # It is a parameter rather than a measurement because nothing in flight
        # changes it: Nav2 is a 2D planner and never commands height, so the
        # drone holds whatever altitude takeoff_node.py climbed it to. Keep this
        # in step with the scout's takeoff_altitude in 41068_ignition.launch.py.
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
        # People the detector saw cut off by the frame edge. Its
        # 'edge_detections_topic'; see _on_edge_detections.
        self.create_subscription(
            Detection2DArray, 'camera/edge_detections',
            self._on_edge_detections, 10)

        # Nav2's global costmap, read so the sweep can be planned AROUND
        # obstacles rather than through them. It is a fixed, non-rolling map in
        # the odom frame, so it keeps what the lidar has already seen - without
        # that memory there would be nothing to plan against.
        self.costmap = None
        # SLAM's occupancy map, used ONLY to judge what is walled in.
        self.slam_map = None
        self._slam_walls_cache = None
        self._slam_walls_for = None
        if self.get_parameter('use_slam_walls').value:
            slam_qos = QoSProfile(
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(OccupancyGrid, 'map',
                                     self._slam_map_callback, slam_qos)
        # One-shot guard, see _check_unknown_tracking.
        self._unknown_checked = False
        if self.get_parameter('avoid_obstacles').value:
            costmap_qos = QoSProfile(
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(
                OccupancyGrid, self.get_parameter('costmap_topic').value,
                self._costmap_callback, costmap_qos)

        self.coverage_resolution = float(self.get_parameter('coverage_resolution').value)

        # Sized to the search polygon. Built here for the starting area and
        # rebuilt whenever the operator redraws it (see on_area_request).
        self.last_coverage_log = 0.0
        self._rebuild_coverage_grid()

        self.coverage_pub = self.create_publisher(OccupancyGrid, 'search_coverage', 10)
        # Latched, so the ground station still gets the route if it starts
        # up (or is restarted) after the sweep was planned.
        path_qos = QoSProfile(depth=1,
                              reliability=QoSReliabilityPolicy.RELIABLE,
                              durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(Path, 'search_path', path_qos)
        # Latched: a ground station started or restarted later still learns
        # what the scout is doing instead of showing nothing.
        self.status_pub = self.create_publisher(String, 'search_status', path_qos)
        # Nav2's own route to the current goal. Progress has to be measured
        # along THIS, not as the crow flies - see _check_stalled.
        self.nav_plan_length = None
        self.create_subscription(Path, 'plan', self._on_nav_plan, 1)
        # Where the drone is going to look, on its own topic rather than as
        # the last pose of search_path. It is not a sweep leg and must not be
        # drawn as one, but the ground station still needs it to show the
        # trip. Latched, and only drawn while the status says LOOKING, so a
        # stale value from an earlier plan is never shown.
        self.vantage_pub = self.create_publisher(
            PointStamped, 'search_vantage', path_qos)
        self.status_state = None
        self.status_detail = ''
        self.coverage_timer = self.create_timer(0.5, self.update_coverage)

        # Report the footprint periodically rather than once at startup: at
        # startup neither CameraInfo nor TF has arrived, so the numbers would
        # be the fallbacks rather than the real ones. This also means the log
        # follows the drone if its altitude changes mid-flight.
        self.footprint_timer = self.create_timer(10.0, self._log_footprint)

        # Re-derive what is reachable from the live map and replan when that
        # changes - see _review_map. This is what replaced remembering where
        # a waypoint failed, so that a verdict of "unreachable" can be taken
        # back once the drone has flown somewhere it can see from.
        self.map_timer = self.create_timer(
            max(0.5, float(self.get_parameter('map_review_period').value)),
            self._review_map)

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
        # An area that arrived before the map frame existed (see on_area_request).
        self.pending_area = None
        self.create_subscription(
            PolygonStamped, 'search_area_request', self.on_area_request, area_qos)


    def _declare_search_parameters(self):
        """The search area, the camera, and how the sweep is flown."""

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

        # --- people at the edge of the camera, at the edge of the area ----
        # The detector does not count a person cut off by the frame edge. In
        # the middle of the area that is fine - the next leg overlaps this one
        # and sees them whole - but on a boundary leg there is no next leg, so
        # a person just over the edge would be passed by for good. There the
        # drone flies over them, looks, and goes back to where it left the
        # sweep. See _on_edge_detections.
        self.declare_parameter('investigate_edges', True)
        # "At the edge of the area" means the camera sees past the boundary:
        # the drone is within its footprint radius of it, plus this margin.
        self.declare_parameter('edge_band_margin', 1.0)          # metres
        # Never detour further than this. A person at the frame edge is at
        # most about 7 m off at 10 m altitude, so anything more is a bad
        # projection rather than somebody to go and look at.
        self.declare_parameter('edge_max_detour', 10.0)          # metres
        # How long to hover over them. Long enough for a few YOLO frames (it
        # runs at single-digit FPS on CPU) and for the ground station's
        # thermal check to hold the search if there is body heat.
        self.declare_parameter('edge_dwell_s', 4.0)              # seconds
        # Stop this far short of the sighting rather than right over it. The
        # person only has to be well inside the frame, not under the drone:
        # the thermal camera, the narrower of the two, still sees about 4 m
        # either side of centre at 10 m, so 2.5 m short keeps them in both.
        self.declare_parameter('edge_stop_short', 2.5)           # metres
        # Edge sightings this close to one already checked are the same spot.
        self.declare_parameter('edge_spot_radius', 3.0)          # metres



    def _declare_obstacle_parameters(self):
        """Reading the costmap, and giving up on a waypoint."""
        # --- obstacle avoidance ------------------------------------------
        # Nav2's global costmap, which now keeps what the lidar has seen, is
        # read so the sweep is planned around obstacles instead of through
        # them. Set avoid_obstacles false to go back to pure geometry.
        self.declare_parameter('avoid_obstacles', True)
        self.declare_parameter('costmap_topic', 'global_costmap/costmap')
        # Cost at or above which a cell counts as blocked. This is on the
        # PUBLISHED OccupancyGrid scale, not Nav2's raw 0-254 costs: the
        # costmap publisher translates lethal 254 -> 100, inscribed 253 -> 99
        # and 1..252 -> 1..98. 77 is raw 200 on that scale, which also
        # catches the strongly inflated ring around an obstacle. (The old
        # default of 200 could never be reached, so nothing was ever blocked.)
        self.declare_parameter('obstacle_cost_threshold', 77)
        # Cost at or above which a cell blocks the CAMERA's view along the
        # ground. Deliberately higher than the planning threshold: 77 catches
        # Nav2's inflated ring, which is a margin for the drone's width and
        # has no physical height at all - the camera looks down over it.
        #
        # With 77 used here, flying along the side of the mountain put the
        # drone INSIDE that ring, its own cell read as obstacle, and every
        # sight line failed at once: the whole camera footprint was thrown
        # away and the percentage stopped climbing over ground in plain view.
        # 99 is inscribed-and-above, i.e. where the obstacle really is.
        self.declare_parameter('sight_block_threshold', 99)
        # How far to stay back from an obstacle edge, on top of inflation.
        self.declare_parameter('obstacle_clearance', 1.0)
        # Pieces of a split leg shorter than this are not worth flying.
        self.declare_parameter('min_leg_length', 1.5)
        # Give up on a waypoint only once the drone has stopped GETTING CLOSER
        # to it for this long. An earlier version used a flat time limit from
        # when the goal was sent, which was wrong: a sweep leg can be 30 m and
        # the drone makes about 0.3 m/s, so a perfectly healthy leg took far
        # longer than the limit and got abandoned a third of the way along.
        # Measuring progress instead tells a slow leg apart from a stuck one.
        self.declare_parameter('goal_stall_timeout', 60.0)  # seconds, 0 disables
        # Getting this much closer counts as progress and resets the clock.
        # Deliberately not a token amount: a drone bouncing off a mountain face
        # edges a little nearer now and then, and too small a threshold lets
        # that count as progress forever. Even the slowest honest leg seen here
        # (0.06 m/s) closes 3.6 m in the 60 s window, so 2 m is safe for real
        # flying while ruling out a drone that is only jostling.
        self.declare_parameter('goal_progress_threshold', 2.0)   # metres
        # When a waypoint turns out to be unreachable, replan the rest of the
        # sweep against the costmap as it now stands instead of just skipping
        # it. By the time the drone has been stopped by the side of a mountain
        # it has mapped far more of that mountain than it had when the sweep
        # was first planned, so a second attempt can route around what the
        # first one did not know was there.
        self.declare_parameter('replan_on_obstacle', True)
        # A cap, so a genuinely impossible area cannot replan forever.
        self.declare_parameter('max_replans', 5)
        # Hard backstop on one waypoint, as a multiple of how long the leg
        # should take at this speed. Needed because the progress check above
        # can be defeated by oscillation: a drone bouncing off a mountain face
        # creeps a little closer now and then, which keeps resetting the
        # progress clock while it gets nowhere.
        self.declare_parameter('goal_time_factor', 4.0)
        self.declare_parameter('assumed_speed', 0.3)        # m/s, measured


    def _declare_reachability_parameters(self):
        """Telling "never looked at" from "cannot be reached"."""
        # --- staying honest about what is reachable ----------------------
        # "Unreachable" is not a fact about the world. It is a belief about
        # where the drone has so far been able to LOOK, so it gets re-derived
        # from the live map on this interval rather than remembered.
        #
        # What this replaced: a no-go disc of fixed radius around every
        # waypoint that failed. That was wrong twice over. Wrong size - the
        # rock in this world is 15-49 m across, so a 6 m disc blanked out less
        # than half of one summit and the next sweep leg, 6.2 m away, landed
        # straight back inside it. And permanent - ground written off on the
        # approach stayed written off for the whole flight, even after the
        # drone had flown round and could plainly see it was open.
        self.declare_parameter('map_review_period', 2.0)    # seconds, 0 disables
        # Replan only once the reachable part of the area has changed by at
        # least this fraction of what the operator drew. A threshold rather
        # than any change at all is what stops a replan on every flicker of
        # the costmap, and it guarantees each replan makes real progress.
        self.declare_parameter('map_review_fraction', 0.02)  # 2%
        # Map-driven replans are normal operation rather than a failure, so
        # they get their own generous cap, separate from max_replans.
        self.declare_parameter('max_map_replans', 20)
        # The lidar's 360 rays fan out, so past about 9 m a single scan
        # leaves unobserved wedges BETWEEN them (0.70 m at the 40 m limit,
        # against 0.15 m cells). Gaps narrower than this, with observed-clear
        # ground all round, are closed up before the flood runs - never over
        # a cell the costmap calls blocked, so it cannot invent a way through
        # rock. See _confirmed_from.
        self.declare_parameter('observation_gap_fill', 0.75)   # metres
        # How far the lidar can see, from the sensor's own <max> in the xacro.
        # Used to rule out flying to a vantage point from which the ground in
        # question would still be out of range - the drone would arrive having
        # learned nothing.
        self.declare_parameter('sensor_range', 40.0)        # metres
        # Vantage points this close to one already used count as the same
        # spot, or the drone ping-pongs between two neighbouring frontier
        # cells for ever - a rock's interior is never resolved from outside.
        self.declare_parameter('vantage_min_separation', 3.0)   # metres
        # Give up looking after this many trips in a row that resolve
        # nothing. Needed because unknown ground does not block a sight line,
        # so a rock whose face is only partly mapped always looks as though
        # something might see into it. 5 is measured: working round this
        # world's ridge takes three or four fruitless trips before the one
        # that breaks through, so 3 gave up just short (43.6% of the area
        # searched) where 5 reached 100%.
        self.declare_parameter('vantage_patience', 5)
        # Keep vantage points this far clear of mapped obstacle: a frontier
        # cell is clear when picked but the next scan often fills in rock
        # beside it, and arriving at a dead waypoint costs a failure replan.
        self.declare_parameter('vantage_obstacle_margin', 2.0)   # metres
        # Do not fly anywhere to resolve less unverified ground than this. The
        # camera covers about 98 m2 in one go, so a trip for a square metre
        # costs far more than the ground is worth - and with a patch that
        # small the answer flickers as the costmap updates, which had the
        # drone lapping between "something left to look at" and "no there
        # isn't" for several seconds at the end of a run, commanding nothing.
        self.declare_parameter('min_unverified_area', 2.0)      # m2
        # Use SLAM's occupancy map, as well as the costmap, to decide whether
        # ground is WALLED IN. Not for anything else: waypoints are still
        # placed against the costmap, so the controller and the planner never
        # disagree about where the rock is.
        #
        # Measured on one box drawn inside the mountain. Taking the boundary
        # of the unknown region it sat in: the SLAM map had it 94% walled
        # with 6 m of gap, the costmap 75% walled with 31 m. Inflating the
        # costmap's walls to the drone's width and beyond never sealed it -
        # you cannot bridge a missing 31 m by making walls fatter. The SLAM
        # map sealed at every width from 0.3 m to 1.0 m.
        self.declare_parameter('use_slam_walls', True)
        # Close gaps narrower than twice this in SLAM's walls, so a hole the
        # drone could not fit through is not read as a way in.
        #
        # Deliberately small. The costmap already inflates obstacles by 0.6 m,
        # and 0.75 m here on top of that grew every wall by 1.35 m - enough to
        # pinch shut the corridors around the mountain once its outline
        # closed. The reachable region then fragmented, ground the drone had
        # already flown over fell outside it, and the overlay showed searched
        # ground as unverified. This only has to bridge the gaps BETWEEN
        # laser returns, not stand in for the drone's width.
        self.declare_parameter('slam_wall_inflate', 0.2)        # metres
        # Do not plan until the lidar has confirmed at least this much clear
        # ground around the drone. Planning on the first costmap, while the
        # drone is still lifting off the ute, found no frontier to look from,
        # took that as proof the whole area could not be seen, and wrote it
        # all off: the drone flew one waypoint home and called the search
        # done. At survey height the confirmed region is thousands of m2.
        self.declare_parameter('min_confirmed_area', 50.0)      # m2
        # Stop waiting for that after this long and plan anyway, so a drone
        # that really is boxed in still does something. _finish_plan will not
        # write ground off while the map is this thin, so nothing is lost.
        self.declare_parameter('map_wait_timeout', 30.0)        # seconds
        # How much unverified ground one trip has to resolve to count as
        # progress, as a fraction of the area the operator drew.
        self.declare_parameter('vantage_progress_fraction', 0.02)
        # How often to re-check that the waypoint being flown to is still
        # reachable. The cheap "is it in an obstacle" test runs every tick;
        # the flood fill covers the whole costmap, so it runs on this interval
        # instead. It used to be 3 s because the flood was a 1.7 s Python
        # loop; at 19 ms via scipy it can run every tick.
        self.declare_parameter('goal_recheck_period', 1.0)  # seconds
        # Drop waypoints the drone cannot actually get to, worked out by
        # flooding outwards from its position across free costmap cells.
        self.declare_parameter('drop_unreachable', True)

        # An arbitrary search polygon, as a flat list: [x1, y1, x2, y2, ...].
        # Leave it empty to use the min_x/max_x/min_y/max_y rectangle instead.
        # This is the STARTING area; the operator can redraw it at any time
        # from the ground station (see on_area_request).
        self.declare_parameter('polygon', [0.0])

        # True: hover at spawn and do nothing until the operator draws a search
        # area in the ground station. False: start sweeping the launch-parameter
        # area (polygon or min/max rectangle) straight away, as before.
        self.declare_parameter('wait_for_area', True)

    def on_area_request(self, msg):
        """Operator drew a new search area: apply it, but never die trying.

        The node HAS died here. A smaller polygon than the last one indexed
        an array left over from the old coverage grid out of bounds, the
        IndexError came out of a subscription callback, rclpy stopped the
        executor, and the process exited mid-mission - taking the drone's
        only listener for the E-Stop topic with it.

        That particular cause is fixed (see _forget_the_old_area), but a
        drawn area is operator input arriving from another process, and no
        single piece of it is worth the whole node. A bad one is now
        rejected, loudly, with the drone left holding.

        Not a bare except: the traceback is logged in full, so this hides
        nothing - it only stops a bug in here from becoming a crash.
        """
        try:
            self._apply_area_request(msg)
        except Exception:
            self.get_logger().error(
                'That search area could not be used, so it has been '
                'rejected. The drone is holding where it is. Draw the area '
                'again; if it keeps failing, this traceback is the bug:\n'
                + traceback.format_exc())
            self.area_set = False
            self.planned = False
            self._set_status(
                ST_IDLE, 'that search area could not be used - draw it again')

    def _apply_area_request(self, msg):
        """Take the drawn area into use.

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
        #
        # The map frame only exists once SLAM is running, which is after
        # takeoff and often later still (no lidar returns at survey height).
        # The ground station sends areas drawn before then in odom instead,
        # but if the frame is missing anyway the area is kept and retried
        # from tick() rather than dropped.
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, msg.header.frame_id,
                rclpy.time.Time(), timeout=Duration(seconds=0.5))
        except Exception:
            if self.pending_area is None:
                self.get_logger().info(
                    'Search area received; it will be used once the drone '
                    'is airborne and SLAM is up.')
            self.pending_area = msg
            return
        self.pending_area = None

        dx = tf.transform.translation.x
        dy = tf.transform.translation.y
        self.polygon = [(x + dx, y + dy) for x, y in pts]
        self.area_set = True

        # The coverage grid is sized to the polygon, so it has to be rebuilt -
        # otherwise progress would be measured against the old shape. The
        # rebuild also forgets everything learned about the old area, which
        # is why nothing here has to remember to.
        self._rebuild_coverage_grid()

        # Abandon the current leg. The operator has new information, so
        # finishing a waypoint from the old area is wasted flying.
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self._start_pass()
        # Counters for the old area go too. Kept apart from _start_pass
        # because a replan resets the sequence but must keep these - the
        # coverage built up so far, the vantage points already used.
        self.replans = 0
        self.plan_reachable_cells = None
        self.map_replans = 0
        self.looked_from = []
        self.edge_checked = []
        self.fruitless_vantages = 0

        self.get_logger().info(
            f'Operator set a new {len(self.polygon)}-sided search area; '
            f'replanning on the next tick.')

    def _refresh_searchable(self):
        """Recompute which cells are worth searching, keeping the coverage.

        _rebuild_coverage_grid() would do this too, but it also clears the
        coverage array - which on a replan would throw away the record of
        everything already swept and send the drone round again.
        """
        before = self.cells_to_search
        self.inside_mask = self._build_inside_mask()
        self.cells_to_search = int(np.count_nonzero(self.inside_mask))
        if self.cells_to_search > before:
            self.get_logger().info(
                f'{self.cells_to_search - before} more cells are searchable '
                f'than before - ground that could not be reached has opened '
                f'up; the target is now {self.cells_to_search} cells.')
        elif self.cells_to_search != before:
            dropped = before - self.cells_to_search
            self.get_logger().info(
                f'{dropped} cells that looked searchable turned out to be '
                f'obstacles once mapped; the target is now '
                f'{self.cells_to_search} cells.')
        # Progress is measured against a smaller target now, so let the next
        # percentage print rather than waiting for a further 5% of a number
        # that just changed underneath it.
        self.last_coverage_log = 0.0

    def _forget_the_old_area(self):
        """Drop every per-cell memory, because they are shaped to the OLD grid.

        One place, called by _rebuild_coverage_grid, so a field added here
        later cannot be left behind by a caller that did not know about it.
        That is how the node died: the list used to be split in two, and the
        half that ran AFTER the rebuild left self.unverifiable - sized for
        the previous area - to be indexed by the new grid's rows.

        Two separate reasons to clear, which is why they belong together:

        - shape. unverifiable, ever_reachable, ever_confirmed and
          unverified_at_vantage are arrays on the coverage grid. A new area
          with fewer rows is an IndexError waiting to happen, or a silent
          broadcast error where they are used whole.

        - meaning. confirmed and possible were flooded for the previous
          area, from a different position, against an older map. A new box
          falling outside them came out with no searchable cells at all and
          the node announced "SEARCH AREA NOT SEARCHABLE" - for an area a
          fresh flood reached in full, 960 cells of 960.

        Only assignment, no reads, so this is safe to call from __init__
        before the fields exist.
        """
        self.confirmed = None
        self.possible = None
        self._shadow_cache = None
        self._shadow_cache_for = None
        self.unverifiable = None
        self.ever_reachable = None
        self.ever_confirmed = None
        self.unverified_at_vantage = None

    def _rebuild_coverage_grid(self):
        """Size and clear the coverage grid for the current polygon.

        Pads a little beyond the search area so edge passes still land inside
        the grid, and recomputes which cells are actually inside the shape -
        the percentage is measured against those only, or a non-rectangular
        area could never reach 100%.
        """
        # Before the mask is built, never after: _build_inside_mask reads
        # these and indexes them with the NEW grid's rows.
        self._forget_the_old_area()
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
        if not self.area_set:
            self.get_logger().info(
                'Still idle: draw a search area in the ground station to start.')
            return
        if not self.planned:
            return
        if self.estopped:
            self.get_logger().warn(
                f'STILL E-STOPPED at waypoint {self.index + 1}/{len(self.points)} '
                f'- hovering until the operator presses RESUME.'
            )
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
        # NOTE: no early return for a short path. "No path" and "no ground
        # left to cover" are different things, and an early 0.0 conflated
        # them: on a replan with everything already searched the sweep is
        # legitimately empty, and scoring that 0% is what set the retry loop
        # tightening. The target is worked out first, below, and an empty
        # target answers 1.0; an empty path against a real target still falls
        # out of the loop as 0.0, which is correct.
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

        # Drop what this sweep was never going to cover, so the check marks
        # the plan against its own job rather than against the whole shape.
        #
        # Blocked cells, because the camera cannot see through rock. And
        # ground not CONFIRMED reachable, because legs are only planned over
        # ground the drone can prove a way to - counting the rest made a
        # perfectly sound plan score badly and sent the retry loop below
        # tightening the spacing for nothing: measured going 10.67 m -> 4.0 m,
        # 2.7x as many legs, while the uncovered part was rock either way.
        if bool(self.get_parameter('avoid_obstacles').value) and self.costmap is not None:
            confirmed = self.confirmed
            for r in range(rows):
                for c in range(cols):
                    if not inside[r, c]:
                        continue
                    x, y = gx[r, c], gy[r, c]
                    if self._is_blocked(x, y) or (
                            confirmed is not None
                            and not self._is_reachable(x, y, confirmed)):
                        inside[r, c] = False

        # On a REPLAN the sweep only covers what is left, so ground the
        # camera has already been over must come out of the target too -
        # otherwise the check marks an almost-finished pass against the whole
        # area, scores near zero, and the retry loop tightens the spacing for
        # nothing. Seen in flight at 88% coverage: six "would leave 100.0% of
        # the area unsearched - tightening" lines in a row, spacing dragged
        # from 8.98 m to 3.4 m, and a flat "100.0% will NOT be searched"
        # warning on a search that was nearly done.
        if self.replanning:
            for r in range(rows):
                for c in range(cols):
                    if inside[r, c] and self._already_searched(gx[r, c], gy[r, c]):
                        inside[r, c] = False

        total = int(inside.sum())
        if total == 0:
            # Nothing left to cover is FULLY covered, not uncovered. Returning
            # 0.0 here is what set the retry loop off: it read "no ground
            # covered" when the honest answer is "no ground outstanding".
            return 1.0

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
        strip of ground its own camera covers, so it works the spacing out
        from that, and gaps become impossible by construction rather than
        something we check for afterwards.

        Called from tick() rather than __init__ because it needs live sensor
        data. Returns True once the path has been built, False to be retried
        on the next tick.

        The work is in the four steps below; this is just their order.
        """
        if self.cam_fx is None:
            self.get_logger().info('Waiting for camera info before planning the sweep...')
            return False
        avoiding = bool(self.get_parameter('avoid_obstacles').value)
        if avoiding and self.costmap is None:
            self.get_logger().info(
                'Waiting for the costmap before planning - planning now would '
                'route straight through anything out there...')
            return False

        blocked = self._blocked_test(avoiding)
        if not self._map_ready_to_plan():
            return False
        chosen = self._choose_spacing(blocked)
        if chosen is None:               # no legs: a different question
            return self._nothing_to_sweep()
        predicted, footprint, source = chosen
        self._finish_plan()
        self._log_plan(avoiding, predicted, footprint, source)
        return True

    def _too_little_mapped(self):
        """Has the lidar confirmed too little clear ground to judge anything?

        On the ute, and for the first second or so of the climb, the confirmed
        region is a few square metres round the drone. Nothing can be planned
        or ruled out from that - there is no frontier yet, which is not the
        same as there being nowhere to look from.
        """
        if self.confirmed is None or self.costmap is None:
            return False
        res = self.costmap.info.resolution
        area = int(np.count_nonzero(self.confirmed)) * res * res
        return area < float(self.get_parameter('min_confirmed_area').value)

    def _map_ready_to_plan(self):
        """Wait, up to map_wait_timeout, for enough of a map to plan from."""
        if not self._too_little_mapped():
            self._map_wait_started = None
            return True
        if self._map_wait_started is None:
            self._map_wait_started = self._now()
        waited = self._now() - self._map_wait_started
        limit = float(self.get_parameter('map_wait_timeout').value)
        if waited < limit:
            self.get_logger().info(
                'Waiting for the lidar to map the ground around the drone '
                'before planning - from the pad it has seen too little to '
                'tell open ground from rock...')
            return False
        self.get_logger().warn(
            f'Still almost nothing mapped after {waited:.0f} s - planning '
            f'anyway. Unmapped ground stays in the search rather than being '
            f'written off.')
        self._map_wait_started = None
        return True

    def _blocked_test(self, avoiding):
        """Build the "is this point off limits?" test the sweep is cut by.

        Returns a callable, or None when obstacle avoidance is off. Also
        refreshes self.confirmed and self.possible, which the rest of the
        plan and the overlay both read.
        """
        # Two questions of the same map, and the gap between the answers is
        # the whole point:
        #
        #   confirmed - flood over ground the lidar has SEEN to be clear;
        #               unobserved ground counts as WALL
        #   possible  - the same, but unobserved ground counts as FLOOR
        #
        # in both       -> demonstrably reachable. Plan legs here.
        # possible only -> unverified: could be the far side of a rock or the
        #                  inside of it. Do not fly at it yet.
        # in neither    -> walled in with no way through.
        #
        # Planning over CONFIRMED ground only is what keeps waypoints out of
        # solid rock: a single flood counting unobserved cells as clear let
        # both obstacle tests pass for a point in the middle of a mountain.
        self.confirmed = None
        self.possible = None
        if avoiding and self.get_parameter('drop_unreachable').value:
            here = self._lookup_robot_xy()
            if here is not None:
                self.confirmed = self._confirmed_from(here)
                self.possible = self._reachable_from(here)

        if avoiding and self.confirmed is not None:
            confirmed = self.confirmed
            blocked = lambda x, y: (self._is_blocked(x, y)
                                    or not self._is_reachable(x, y, confirmed))
        elif avoiding:
            blocked = self._is_blocked
        else:
            blocked = None

        return self._skip_searched_ground(blocked)

    def _skip_searched_ground(self, blocked):
        """On a REPLAN, add "the camera has already been here" to the test.

        The new sweep then only covers what is left. Without it the drone
        starts the whole area again from one end every time it meets an
        obstacle, re-flying everything it had already searched.
        """
        if self.replanning:
            geometric = blocked
            if geometric is None:
                return self._already_searched
            return lambda x, y: (geometric(x, y)
                                 or self._already_searched(x, y))
        return blocked

    def _choose_spacing(self, blocked):
        """Set self.spacing and self.points, checking the result covers the area.

        Returns (predicted coverage, footprint width, how the spacing was
        arrived at), or None if the sweep came out empty - which is a
        different situation entirely, handled by _nothing_to_sweep.
        """
        clearance = float(self.get_parameter('obstacle_clearance').value)
        min_leg = float(self.get_parameter('min_leg_length').value)

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
            self.points = make_sweep_polygon(
                self.polygon, self.spacing, is_blocked=blocked,
                clearance=clearance, min_leg=min_leg)
            predicted = self._predicted_coverage(self.points, self._footprint_radius())
        else:
            source = f'{footprint:.1f} m camera footprint x {overlap:.2f} overlap'
            # Start from the footprint-derived spacing, then CHECK it. Spacing
            # on its own does not guarantee coverage on awkward shapes, so if
            # the check predicts a gap we tighten and try again rather than
            # taking off and finding out afterwards.
            self.spacing = footprint * overlap
            radius = self._footprint_radius()
            best_seen = -1.0
            best = (self.spacing, [], 0.0)
            for attempt in range(6):
                self.points = make_sweep_polygon(
                    self.polygon, self.spacing, is_blocked=blocked,
                    clearance=clearance, min_leg=min_leg)
                predicted = self._predicted_coverage(self.points, radius)
                if predicted >= 0.999:
                    break
                # Tightening only helps when the gap is a spacing problem. If
                # the shortfall is ground that cannot be reached at all, every
                # attempt scores the same and the loop just spams the log and
                # triples the number of legs for nothing.
                if attempt and predicted <= best_seen + 0.005:
                    self.spacing, self.points, predicted = best
                    break
                if predicted > best_seen:
                    best_seen = predicted
                    best = (self.spacing, self.points, predicted)
                self.get_logger().warn(
                    f'Spacing {self.spacing:.2f} m would leave '
                    f'{100 * (1 - predicted):.1f}% of the area unsearched - '
                    f'tightening.'
                )
                self.spacing *= 0.85

        # Nothing left after cutting the obstacles out: the whole area is
        # rock. Say so plainly rather than flying a meaningless path.
        if not self.points:
            return None
        # Returned rather than stashed on self: the log line is the only other
        # thing that wants them, and this node already has too much state
        # shared between methods.
        return predicted, footprint, source

    def _nothing_to_sweep(self):
        """No legs. Either the area really is unflyable, or we have not looked
        at it yet, or everything reachable is already done."""
        # Nothing to sweep. That has two very different causes - the area
        # really is all rock, or we have not looked at it yet - and only the
        # first is a reason to give up.
        #
        # An earlier version waited here for a better view when there was
        # unverified ground. That deadlocked: the view only improves if the
        # drone moves, and it only moves if there is a plan, so it sat still
        # for ever. If there is ground we have not judged, fall through to the
        # go-and-look block below instead: flying to one vantage point IS the
        # plan, and it is what produces the better view.
        if not self._worth_looking():
            if self.replanning:
                self.get_logger().info(
                    'Replan found nothing left to fly: everything reachable '
                    'has already been searched. Sweep complete.')
                self.sweep_complete_pub.publish(Empty())
            else:
                self._set_status(
                    ST_UNSEARCHABLE,
                    'nothing in the area can be reached - draw an area that '
                    'includes open ground')
                self.get_logger().error(
                    'SEARCH AREA NOT SEARCHABLE: every part of it is either '
                    'inside an obstacle or walled off with no way in, so there '
                    'is nothing to fly. Draw an area that includes reachable '
                    'open ground.')
            # Report any remainder and stop counting it, so the percentage
            # is honest and nothing comes back to chase it.
            if self._unverified_cells():
                area = self._unverified_cells() * self.coverage_resolution ** 2
                self.get_logger().info(
                    f'{area:.1f} m2 of the area was never verified - too '
                    f'little to be worth flying anywhere to see. Leaving it '
                    f'out of the coverage total.')
                self._write_off_unverified()
            self.planned = True          # stop retrying every tick
            self.replanning = False
            return True

        self._finish_plan()
        self._log_plan(bool(self.get_parameter('avoid_obstacles').value),
                       1.0, self._footprint_radius() * 2.0, 'nothing to sweep')
        return True

    def _finish_plan(self):
        """Tack on the end of the pass: a look-around trip, or the way home."""
        # (0, 0) in the odom frame is the spawn point (see map_frame comment
        # in __init__), so "return to spawn" is just one more waypoint tacked
        # onto the end of the sweep - no separate flight mode needed. Done
        # here, after self.points has its final value, rather than in
        # __init__ - the retry loop above replaces self.points outright, so
        # appending any earlier would just get overwritten.
        # If part of the area can still be judged neither way, end the run by
        # flying somewhere it can be SEEN from instead of going home none the
        # wiser. No new flight mode: it is one more waypoint, and once the
        # drone gets there the lidar has a new view, _review_map() spots the
        # change and replans. Home is appended only once there is nothing left
        # to go and look at. See _frontier_goal.
        vantage = None
        if (self.get_parameter('avoid_obstacles').value
                and self.get_parameter('drop_unreachable').value
                and self._worth_looking()):
            vantage = self._frontier_goal()
        if vantage is not None:
            # Are these trips actually resolving anything? If the last few
            # taught us nothing, the ground in question is not going to be
            # settled by looking at it from yet another angle.
            cells = self._unverified_cells()
            if self.unverified_at_vantage is not None:
                gained = self.unverified_at_vantage - cells
                frac = float(self.get_parameter('vantage_progress_fraction').value)
                if gained / max(1, self.cells_in_shape) < frac:
                    self.fruitless_vantages += 1
                else:
                    self.fruitless_vantages = 0
            if self.fruitless_vantages >= int(self.get_parameter('vantage_patience').value):
                self.get_logger().info(
                    f'The last {self.fruitless_vantages} trips to get a view '
                    f'of the remaining ground resolved almost none of it, so '
                    f'there is no point flying another.')
                vantage = None
            else:
                self.unverified_at_vantage = cells

        if vantage is not None:
            unseen = self._unverified_cells() * self.coverage_resolution ** 2
            self.points.append(vantage)
            self.looked_from.append(vantage)
            self.home_index = None
            self.vantage_index = len(self.points) - 1
            self._publish_vantage(vantage)
            self.get_logger().info(
                f'{unseen:.0f} m2 of the area cannot be judged reachable or '
                f'not from here. Ending this pass at ({vantage[0]:.1f}, '
                f'{vantage[1]:.1f}) to get a view of it, then replanning.')
        else:
            # Say so when part of the area is being left unchecked. Quietly
            # flying home with a gap in the search is the one outcome the
            # operator must never be left to guess at.
            unseen = self._unverified_cells() * self.coverage_resolution ** 2
            if unseen > 0 and self._too_little_mapped():
                # No vantage point because there is no frontier yet, not
                # because every one has been tried. Writing the ground off on
                # that is how a whole area once vanished at take-off. Keep it
                # in the target: _review_map and _after_last_waypoint come
                # back for it once there is a map to look from.
                self.get_logger().warn(
                    f'{unseen:.0f} m2 of the search area cannot be judged yet '
                    f'- too little has been mapped to find anywhere to look '
                    f'from. Keeping it in the search for when there is.')
            elif unseen > 0:
                self.get_logger().warn(
                    f'{unseen:.0f} m2 of the search area could not be checked: '
                    f'nowhere the drone can reach has a view of it, so it is '
                    f'most likely enclosed by rock. Searching everything '
                    f'else, and leaving that out of the coverage total.')
                # Writing it off is also what makes the percentage honest. It
                # is reported separately above rather than silently dropped.
                self._write_off_unverified()
        if vantage is None:
            self.vantage_index = None
        if vantage is None and self.get_parameter('return_to_spawn').value:
            self.points.append((0.0, 0.0))
            self.home_index = len(self.points) - 1

        # What _review_map() will compare against: how much of the area was
        # confirmed reachable at the moment this path was built.
        self.plan_reachable_cells = int(np.count_nonzero(
            self.inside_mask & self._project(self.confirmed)))

        self.planned = True
        self.replanning = False
        self._publish_path()


    def _log_plan(self, avoiding, predicted, footprint, source):
        """Tell the operator what was planned and why."""
        clearance = float(self.get_parameter('obstacle_clearance').value)
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
        # Say plainly how much of the drawn area is out of reach, so a full
        # sweep of what is left is not mistaken for a full sweep of the box.
        in_shape = getattr(self, 'cells_in_shape', 0)
        if avoiding and in_shape:
            flyable = 100.0 * self.cells_to_search / in_shape
            if flyable < 99.0:
                self.get_logger().warn(
                    f'{100 - flyable:.0f}% of the area you drew cannot be '
                    f'searched - it is inside an obstacle or walled off with '
                    f'no way in. Searching the remaining {flyable:.0f}%.')

        if avoiding:
            extra = (' Ground the drone cannot confirm a way to was left out '
                     'of this pass.' if self.confirmed is not None else '')
            self.get_logger().info(
                'Obstacles from the costmap were cut out of the sweep; legs '
                f'stop {clearance:.1f} m clear of them and Nav2 flies around '
                f'between the pieces.{extra}')
        if predicted >= 0.999:
            self.get_logger().info(
                f'Checked before take-off: this path covers '
                f'{100 * predicted:.1f}% of the SEARCHABLE area '
                f'(obstacles excluded).'
            )
        else:
            self.get_logger().warn(
                f'Best achievable with this shape is {100 * predicted:.1f}% '
                f'coverage - {100 * (1 - predicted):.1f}% will NOT be searched.'
            )
        return True

    def tick(self):
        """Called every second; sends the next waypoint to Nav2 once it's ready and we're not already flying to one."""
        if self.pending_area is not None:  # drawn before takeoff, see on_area_request
            self.on_area_request(self.pending_area)
        if not self.area_set:  # nothing to search until the operator draws it
            self._set_status(ST_IDLE, 'draw a search area to start')
            return
        if self.paused():
            if self.estopped:
                self._set_status(ST_ESTOP, 'operator stop - press RESUME')
            else:
                self._set_status(ST_HOLD,
                                 'paused on a contact - confirm or dismiss it')
            return
        if self.busy: #goal is already in flight
            self._check_stalled()
            return
        # Hovering over an edge sighting so the cameras get a proper look.
        # A person there raises a contact and holds the search through the
        # ground station as usual; either way, the next waypoint is the way
        # back to where the drone left the sweep.
        if self._dwell_until is not None:
            if self._now() < self._dwell_until:
                return
            self._dwell_until = None
            back = self.points[self.edge_back_index]
            self.get_logger().info(
                f'Edge check done. Returning to ({back[0]:.1f}, '
                f'{back[1]:.1f}) to carry on the sweep.')
        # Carry out a replan that has been asked for. It has to happen here
        # and not only in the goal-result callback: a goal Nav2 REJECTS
        # returns without ever producing a result, so a replan requested
        # while it was in flight would be dropped and _review_map, which
        # suppresses itself while one is pending, would never ask again.
        if self._replan_wanted:
            self._after_abandoned_goal()
            return
        # Plan on the first tick that has camera data, not in __init__.
        if not self.planned:
            self._set_status(ST_PLANNING, 'working out the search path')
            if not self.plan_sweep():
                return
        if self.index>= len(self.points):
            self._maybe_return_home()
            return
        if not self.nav_client.server_is_ready(): #NAv2 sevrer not up yet
            self.get_logger().info('Waiting for Nav2 to be ready...')
            return
        self.send_goal(self.points[self.index])

    def _start_pass(self, *, replanning=False, returning=False):
        """Reset the waypoint sequence. The ONLY place that may do so.

        Four methods used to set these eleven fields themselves, each in its
        own order, and every deadlock in this node came from one of them
        forgetting one. One owner makes that class of bug impossible rather
        than merely fixed.

        Leaves planned False on purpose: a pass is not planned until
        plan_sweep says so, on tick()'s next turn.
        """
        self.points = []
        self.index = 0
        self.home_index = None
        self.vantage_index = None
        self.busy = False
        self.timed_out = False
        self._replan_wanted = False
        self.replanning = replanning
        self.returning = returning
        self.planned = False
        # An edge check in progress belongs to the old sequence: its two
        # waypoints are gone with it. See _on_edge_detections.
        self.edge_look_index = None
        self.edge_back_index = None
        self._dwell_until = None
        self._goal_superseded = False

    def _heading_home(self):
        """Is the waypoint being flown the trip back to base?

        There are two ways that happens - _fly_home after the search is
        finished, which sets self.returning, and _finish_plan tacking the
        spawn point onto the end of an ordinary plan, which does not. Only
        the first was being recognised, so most trips home reported
        SEARCHING and the operator never saw the run end.
        """
        return bool(self.returning
                    or (self.home_index is not None
                        and self.index == self.home_index))

    def _fly_home(self):
        """Make going home an ordinary one-waypoint pass."""
        self._start_pass(returning=True)
        self.points = [(0.0, 0.0)]
        self.home_index = 0
        self.planned = True
        self._publish_path()
        self._set_status(ST_RETURNING,
                         f'search finished - {self._coverage_summary()}')
        self.get_logger().info(
            f'Search finished ({self._coverage_summary()}). Flying back to base.')

    def _after_last_waypoint(self):
        """The plan is flown out. Decide, in one place, what happens next.

        This decision used to be spread over tick(), the tail of plan_sweep()
        and _maybe_return_home(), each guessing at it - which is how the drone
        came to sit still three separate times today with no error anywhere.
        The order of these branches is the whole logic, so they are listed in
        the order they must be asked.
        """
        # 1. Already on the way home, or home: the run is over.
        if self.returning:
            self._set_status(ST_COMPLETE, self._coverage_summary())
            return

        # 2. Somewhere left worth looking? Fly another pass. This must come
        #    BEFORE the "near base" test below: a pass that happened to end on
        #    a vantage point close to home used to declare the search complete
        #    with ground still unresolved.
        if (self.get_parameter('avoid_obstacles').value
                and self.get_parameter('drop_unreachable').value
                and self._worth_looking()
                and self._frontier_goal() is not None):
            self._refresh_searchable()
            self._start_pass(replanning=True)   # keeps searched ground out
            return

        # 3. Nothing left to resolve. Go home, unless we are there or the
        #    operator has turned that off.
        if not self.get_parameter('return_to_spawn').value:
            self._set_status(ST_COMPLETE, self._coverage_summary())
            return
        here = self._lookup_robot_xy()
        if here is not None and math.hypot(here[0], here[1]) <= 2.0:
            self.returning = True               # nothing to fly
            self._set_status(ST_COMPLETE, self._coverage_summary())
            return
        self._fly_home()

    def _maybe_return_home(self):
        """Kept as the name tick() calls; see _after_last_waypoint."""
        self._after_last_waypoint()

    def _coverage_summary(self):
        """One short phrase for the operator: progress and any gap."""
        if not self.cells_to_search:
            return 'nothing searchable'
        swept = int(np.count_nonzero((self.coverage > 0) & self.inside_mask))
        pct = 100.0 * swept / self.cells_to_search
        gap = 0
        in_shape = getattr(self, 'in_shape_mask', None)
        if in_shape is not None:
            gap = int(np.count_nonzero(in_shape & ~self.inside_mask))
        area = gap * self.coverage_resolution ** 2
        if area >= 0.5:
            return f'{pct:.0f}% of reachable area, {area:.0f} m2 not checked'
        return f'{pct:.0f}% of the area'

    def _now(self):
        """Seconds from the node's own clock, so this follows sim time."""
        return self.get_clock().now().nanoseconds / 1e9

    def _check_stalled(self):
        """Abandon the current waypoint if the drone has stopped approaching it.

        A goal in an unreachable pocket never fails outright - Nav2 just keeps
        replanning longer ways round - so something has to call it. What that
        something must NOT do is measure the whole leg against a fixed time:
        sweep legs here run to 30 m at roughly 0.3 m/s, so a healthy leg takes
        minutes and a stopwatch abandons it halfway. Distance to the goal is
        the honest test - it falls steadily on a slow leg and flattens out on
        a stuck one.
        """
        # paused() covers a contact hold AND the operator E-Stop.
        if self.paused() or self.index >= len(self.points):
            return
        here = self._lookup_robot_xy()
        if here is None:
            return

        gx, gy = self.points[self.index]

        # Read the map rather than infer from behaviour. By the time the drone
        # is hovering at a mountain face, the lidar has already put that face
        # in the costmap - so the node can simply ask whether the waypoint it
        # is flying to is still somewhere it can get to, instead of waiting a
        # minute to work it out from the fact that it is not getting there.
        # The way home is never abandoned for being blocked. The drone took
        # off from the spawn point, so it is reachable by definition - but it
        # sits on the ute, and the ute is an obstacle, so the costmap reads
        # the pad itself at cost 77, exactly the blocked threshold. Every
        # trip home was therefore abandoned the moment it was checked, and
        # the run ended with the drone stopped wherever it happened to be.
        if not self._heading_home() and self._goal_now_impossible(gx, gy):
            self.get_logger().warn(
                f'Waypoint {self.index + 1}/{len(self.points)} is no longer '
                f'reachable - the lidar has since mapped an obstacle there. '
                f'Abandoning it now rather than flying at it.')
            self.timed_out = True
            self._replan_wanted = self._should_replan()
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
            else:
                self._after_abandoned_goal()
            return

        limit = float(self.get_parameter('goal_stall_timeout').value)
        if limit <= 0:
            return
        step = float(self.get_parameter('goal_progress_threshold').value)

        # How far there is still to go. Nav2's own route when we have it,
        # straight-line only as a fallback.
        #
        # This matters most exactly where it used to fail. Routing AROUND an
        # obstacle means flying sideways, and for a long stretch away from the
        # goal, so the straight-line distance stops shrinking - and the drone
        # looked most stuck at the moment it was about to round the corner.
        # Watched in flight: it tracked the bottom edge of the mountain to the
        # eastern end, then gave up and turned back, a few metres short of
        # being able to see round. Along the route there was steady progress
        # the whole way.
        crow = math.hypot(gx - here[0], gy - here[1])
        dist = self.nav_plan_length if self.nav_plan_length is not None else crow

        if self.goal_best_dist is None or dist < self.goal_best_dist - step:
            self.goal_best_dist = dist
            self.goal_progress_at = self._now()
            return
        # A detour that the straight line cannot see: accept either measure.
        if crow < self.goal_best_crow - step:
            self.goal_best_crow = crow
            self.goal_progress_at = self._now()
            return

        stalled = self._now() - self.goal_progress_at > limit

        # Backstop. A drone bouncing off a mountain face creeps a little closer
        # every so often, which resets the progress clock above and lets it
        # oscillate there indefinitely. Cap the whole leg at a generous
        # multiple of how long it should take at this speed.
        overrun = False
        if self.goal_sent_at is not None and self.goal_leg_length:
            speed = max(0.05, float(self.get_parameter('assumed_speed').value))
            # Budget against the route Nav2 is actually flying. A leg round an
            # obstacle is far longer than the straight line, and budgeting on
            # the straight line cut it off part way round.
            leg = max(self.goal_leg_length, self.nav_plan_length or 0.0)
            budget = (leg / speed
                      * float(self.get_parameter('goal_time_factor').value))
            overrun = self._now() - self.goal_sent_at > budget
            if overrun and not stalled:
                self.get_logger().warn(
                    f'Waypoint {self.index + 1}/{len(self.points)} has taken '
                    f'{self._now() - self.goal_sent_at:.0f} s for a {self.goal_leg_length:.0f} m '
                    f'leg (budget {budget:.0f} s) and is still {dist:.1f} m away - '
                    f'it is going back and forth rather than getting there.')

        if stalled or overrun:
            if stalled:
                self.get_logger().warn(
                    f'Waypoint {self.index + 1}/{len(self.points)} has not got any '
                    f'closer for {limit:.0f} s (still {dist:.1f} m away) - giving up '
                    f'on it. It is probably unreachable, inside or behind an '
                    f'obstacle.')
            self.timed_out = True
            self._replan_wanted = self._should_replan()
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
            else:
                self._after_abandoned_goal()

    def _goal_now_impossible(self, gx, gy):
        """Has the map turned the current waypoint into somewhere unreachable?

        Two tests, at different costs:

        - is the waypoint inside a mapped obstacle? One array lookup, so it
          runs every tick. This is the mountain case: as the drone closes in,
          the lidar fills in the rock the waypoint sits in and this fires
          within a tick or two.
        - is it walled off with no way in? That needs a flood fill over the
          whole costmap, so it runs every goal_recheck_period instead.
        """
        if not self.get_parameter('avoid_obstacles').value or self.costmap is None:
            return False

        if self._is_blocked(gx, gy):
            return True

        if not self.get_parameter('drop_unreachable').value:
            return False
        period = float(self.get_parameter('goal_recheck_period').value)
        if period <= 0 or self._now() - self.last_recheck < period:
            return False
        self.last_recheck = self._now()
        here = self._lookup_robot_xy()
        if here is None:
            return False
        reached = self._reachable_from(here)
        return reached is not None and not self._is_reachable(gx, gy, reached)

    def _should_replan(self):
        """Is a replan worth doing after giving up on a waypoint?"""
        if not self.get_parameter('replan_on_obstacle').value:
            return False
        if self.replans >= int(self.get_parameter('max_replans').value):
            self.get_logger().warn(
                f'Already replanned {self.replans} times for this area - '
                f'skipping the rest instead of replanning again.')
            return False
        return self.costmap is not None

    def _after_abandoned_goal(self):
        """Either replan around what we now know, or just skip the waypoint."""
        if not self._replan_wanted:
            self.next_waypoint()
            return
        self._replan_wanted = False
        self.timed_out = False       # leave no stale flag for the next goal
        # A map-driven replan is routine - the drone learned something and is
        # acting on it - so it does not spend one of the failure budget.
        if self._replan_is_map_driven:
            self._replan_is_map_driven = False
        else:
            self.replans += 1
            self.get_logger().warn(
                f'Replanning the rest of the search (attempt {self.replans}) '
                f'against the obstacles mapped so far. Ground already searched '
                f'is kept, so only what is left gets flown.')
        # Keep WHAT HAS BEEN SEARCHED, but refresh WHAT IS WORTH SEARCHING.
        # The mask of searchable cells was built when the operator drew the
        # area, when the mountain had not been mapped - so its cells counted
        # as "still to do". They are rock the camera can never see, so without
        # this the percentage would stall short of 100% forever and there
        # would be no way to tell a mountain from a genuine gap.
        self._refresh_searchable()
        self._start_pass(replanning=True)

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

        # Tell the operator which kind of trip this is. A go-and-look leg
        # takes the drone outside the search area with no sweep pattern to
        # draw, which is indistinguishable from a fault unless it is named.
        if self._heading_home():
            self._set_status(ST_RETURNING, 'search done - flying back to base')
        elif self.edge_look_index is not None and self.index == self.edge_look_index:
            self._set_status(
                ST_INVESTIGATING,
                f'person at the edge of the camera - moving over '
                f'({x:.0f}, {y:.0f}) to check')
        elif self.edge_back_index is not None and self.index == self.edge_back_index:
            self._set_status(
                ST_INVESTIGATING,
                f'checked - back to ({x:.0f}, {y:.0f}) to carry on the sweep')
        elif self.vantage_index is not None and self.index == self.vantage_index:
            self._set_status(
                ST_LOOKING,
                f'no view of part of the area yet - moving to '
                f'({x:.0f}, {y:.0f}) to see it, then replanning')
        else:
            legs = (self.vantage_index if self.vantage_index is not None
                    else len(self.points))
            self._set_status(ST_SEARCHING,
                             f'waypoint {self.index + 1} of {legs}')

        self.busy = True
        # Progress tracking, not a stopwatch on the whole leg.
        self.goal_best_dist = None
        self.goal_best_crow = float('inf')
        self.nav_plan_length = None
        self.goal_progress_at = self._now()
        self.goal_sent_at = self._now()
        self.last_recheck = self._now()
        here = self._lookup_robot_xy()
        self.goal_leg_length = (math.hypot(x - here[0], y - here[1])
                                if here is not None else None)
        self.timed_out = False
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
            # Nav2 rejects goals for a moment after its action server first
            # appears, before bt_navigator is active. That is exactly when the
            # first goal goes out if the area was drawn before LAUNCH, so try
            # the same waypoint again on the next tick before giving up on it.
            self.rejections += 1
            if self.rejections < self.MAX_REJECTIONS:
                self.get_logger().warn(
                    f'Nav2 rejected that goal ({self.rejections}/'
                    f'{self.MAX_REJECTIONS}). Retrying.')
                self.busy = False
                return
            self.get_logger().warn('Nav2 rejected that goal. Skipping it.')
            self.rejections = 0
            self.next_waypoint()
            return

        self.rejections = 0
        self.goal_handle = goal_handle
        if self.paused():
            # Hold or E-Stop arrived while this goal was still being accepted.
            goal_handle.cancel_goal_async()

        #Stage 3: ask to be told when its finished flying.
        result_future= goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_finished)

    def goal_finished(self,future):
        """Stage 3: Nav2 has arrived, failed or given up"""
        self.goal_handle = None
        status = future.result().status
        if self._goal_superseded:
            # Cancelled to start an edge check. It may have arrived in the
            # same instant, but the waypoint now at self.index is the edge
            # check, so advancing would skip it.
            self._goal_superseded = False
            self.busy = False
            return
        if self.timed_out:
            # We abandoned this one as unreachable. Either replan the rest of
            # the sweep against what the costmap now shows, or just skip it.
            self.timed_out = False
            self.busy = False
            self._after_abandoned_goal()
            return
        if self.paused() or status == GoalStatus.STATUS_CANCELED:
            # Cancelled by a hold or E-Stop, or by the operator redrawing the
            # search area. Don't advance: on a hold, resuming re-flies the same
            # waypoint instead of skipping it, and on a redraw the index has
            # already been reset to the start of the new pattern.
            self.busy = False
            return
        if future.result().status ==GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Arrived. ')
        else:
            self.get_logger().warn('Could not reach that one. Skipping it. ')
        self.next_waypoint()

    def paused(self):
        """True while the sweep must not fly: a contact hold or an E-Stop."""
        return self.holding or self.estopped

    def on_estop(self, msg):
        if msg.data == self.estopped:
            return
        self.estopped = msg.data
        if self.estopped:
            self.get_logger().warn(
                f'E-STOP: stopping at waypoint {self.index + 1}/'
                f'{len(self.points)}. Hovering until the operator presses RESUME.'
            )
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
        else:
            self.get_logger().info('E-Stop released.')
            self.tick()        # stays put if a contact hold is still on

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
            # tick() itself stays put if an E-Stop is still on.
            self.tick()

    def _on_edge_detections(self, msg):
        """A person cut off by the frame edge: go and look if nothing else will.

        Only on a boundary leg. In the middle of the area the next leg
        overlaps this one and sees them whole, so a detour is wasted flying;
        at the boundary there is no next leg, and someone standing just past
        it would never be fully in view. Then the drone flies over them, so
        the RGB and thermal cameras (whose narrower view sits inside the RGB
        one) can both see, hovers for edge_dwell_s, and flies back to where
        it left the sweep. Two ordinary waypoints, so holds, E-Stops and
        stall checks all work on them unchanged.
        """
        if not msg.detections or not self._edge_check_allowed():
            return
        best = max(msg.detections,
                   key=lambda d: max((r.hypothesis.score for r in d.results),
                                     default=0.0))
        pose = self._pose_at(msg.header.stamp)
        here = self._lookup_robot_xy()
        if pose is None or here is None:
            return
        target = self._edge_target(best, pose)

        # "At the edge of the area": the camera sees past the boundary.
        band = (self._footprint_radius()
                + float(self.get_parameter('edge_band_margin').value))
        if distance_to_polygon_edge(here[0], here[1], self.polygon) > band:
            return
        radius = float(self.get_parameter('edge_spot_radius').value)
        if any(math.hypot(target[0] - cx, target[1] - cy) < radius
               for cx, cy in self.edge_checked):
            return
        detour = math.hypot(target[0] - here[0], target[1] - here[1])
        if detour > float(self.get_parameter('edge_max_detour').value):
            return
        # Remembered from here on, even if it is then skipped as blocked, so
        # a sighting that cannot be checked is not re-reported every frame.
        self.edge_checked.append(target)
        short = float(self.get_parameter('edge_stop_short').value)
        move = max(0.0, detour - short)
        frac = move / detour if detour > 0 else 0.0
        look = (here[0] + (target[0] - here[0]) * frac,
                here[1] + (target[1] - here[1]) * frac)
        if self.get_parameter('avoid_obstacles').value and self._is_blocked(*look):
            self.get_logger().info(
                f'Person at the edge of the camera near ({target[0]:.1f}, '
                f'{target[1]:.1f}), but the spot to look from is inside a '
                f'mapped obstacle - not flying there.')
            return
        self._start_edge_check(look, here, move)

    def _edge_check_allowed(self):
        """Is the drone flying ordinary sweep legs it can step off for a bit?"""
        if not self.get_parameter('investigate_edges').value:
            return False
        if not self.area_set or not self.planned or self.paused():
            return False
        if self.returning or self.replanning or self._replan_wanted or self.timed_out:
            return False
        if self.edge_look_index is not None or self._dwell_until is not None:
            return False             # one check at a time
        if self.index >= len(self.points):
            return False
        # A goal sent but not yet accepted has no handle to cancel; when it is
        # accepted it would fly the old waypoint and then advance past the
        # edge check. The detector sends another frame in well under a second.
        if self.busy and self.goal_handle is None:
            return False
        return True

    def _pose_at(self, stamp):
        """(x, y, yaw) of the drone when a camera frame was taken, else the
        latest. The detector runs about a second behind, by which time the
        drone has moved on."""
        for when in (rclpy.time.Time.from_msg(stamp), rclpy.time.Time()):
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_frame, when)
            except Exception:
                continue
            q = tf.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
            return tf.transform.translation.x, tf.transform.translation.y, yaw
        return None

    def _edge_target(self, detection, pose):
        """Where on the ground the detection's box centre is."""
        hfov, _ = self._camera_fov()
        width = self.cam_width or float(self.get_parameter('fallback_image_width').value)
        height = self.cam_height or float(self.get_parameter('fallback_image_height').value)
        return pixel_to_ground(
            detection.bbox.center.position.x, detection.bbox.center.position.y,
            width, height, hfov, self._camera_altitude(), *pose)

    def _start_edge_check(self, target, here, detour):
        """Step off the sweep: to look at the sighting, then back to here."""
        at = self.index
        self.points[at:at] = [target, here]
        # Everything after the insert has moved along by two.
        if self.home_index is not None and self.home_index >= at:
            self.home_index += 2
        if self.vantage_index is not None and self.vantage_index >= at:
            self.vantage_index += 2
        self.edge_look_index = at
        self.edge_back_index = at + 1
        self.get_logger().warn(
            f'Possible person at the edge of the camera, at the edge of the '
            f'search area. Flying {detour:.1f} m to ({target[0]:.1f}, '
            f'{target[1]:.1f}) to get them in view, then back to ({here[0]:.1f}, '
            f'{here[1]:.1f}).')
        self._publish_path()
        if self.goal_handle is not None:
            # goal_finished sees this and leaves the index alone, so the next
            # tick flies the edge check.
            self._goal_superseded = True
            self.goal_handle.cancel_goal_async()

    def next_waypoint(self):
        completed_index = self.index
        self.index +=1
        self.busy = False
        if completed_index == self.edge_look_index:
            self._dwell_until = self._now() + float(
                self.get_parameter('edge_dwell_s').value)
            self._set_status(ST_INVESTIGATING,
                             'over the edge sighting - looking for a person')
        elif completed_index == self.edge_back_index:
            self.get_logger().info('Back on the sweep after the edge check.')
            self.edge_look_index = None
            self.edge_back_index = None
        if self.index >=len(self.points):
            self.returning = self.returning or (
                self.home_index is not None
                and completed_index == self.home_index)
            here = self._lookup_robot_xy()
            home = (self.home_index is not None
                    and completed_index == self.home_index
                    and here is not None
                    and math.hypot(here[0], here[1]) <= 3.0)
            if home:
                self.get_logger().info('Sweep complete and returned to spawn.')
            elif self.home_index is not None and completed_index == self.home_index:
                # Said "returned to spawn" while sitting out by the mountain,
                # because the leg home had been abandoned rather than flown.
                self.get_logger().warn(
                    'Sweep complete, but the trip back to base did not finish '
                    '- the drone is still out in the search area.')
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
        if not self.area_set:
            return
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

    def _sealed_mask(self):
        """Cells with no way out through non-wall space: walled in.

        Asked of the map alone, from no particular position, which is what
        lets it override everything else. A region ringed by wall is ringed
        from anywhere; ground merely out of sight from where the drone is
        standing is not, and must not be written off - that distinction is
        the whole reason the reachable set accumulates instead of being taken
        fresh each time.

        Without this the accumulation worked against itself. The inside of a
        mountain is reachable early on, while its outline is still open, so
        it goes into "ever reachable" - and once the outline closes that
        memory keeps it there. Measured on a finished run: 110 m2 of mountain
        interior still drawn as unverified rather than walled in.

        Returns a costmap-shaped bool grid, or None without a costmap.
        """
        got = self._costmap_grid()
        if got is None:
            return None
        grid, info = got
        wall = grid >= self.get_parameter('obstacle_cost_threshold').value
        slam = self._slam_walls()
        if slam is not None:
            wall = wall | slam
        free = ~wall
        if not free.any():
            return None
        labels, count = ndimage.label(free)
        if not count:
            return None
        # Anything connected to the edge of the map has a way out.
        edge = np.concatenate([labels[0, :], labels[-1, :],
                               labels[:, 0], labels[:, -1]])
        outside = np.unique(edge[edge > 0])
        return free & ~np.isin(labels, outside)

    def _reach_picture(self):
        """(ever reachable, ever confirmed) on the coverage grid.

        Falls back to the live floods before anything has accumulated, so the
        very first plan still has an answer. After that the accumulated
        picture is used, which is what stops the searchable area wobbling as
        the drone moves.
        """
        reach, conf = self.ever_reachable, self.ever_confirmed
        if reach is None and self.possible is not None and self.costmap is not None:
            reach = self._project(self.possible)
        if conf is None and self.confirmed is not None and self.costmap is not None:
            conf = self._project(self.confirmed)
        return reach, conf

    def _remember_reachable(self):
        """Fold the current floods into what has EVER been reachable.

        Both floods are computed from where the drone is standing, so taking
        them at face value makes the searchable area wobble as it flies. This
        keeps the union instead: ground is added as the map improves, and is
        only taken away by _build_inside_mask when the costmap says it is
        rock, which is real information rather than a change of viewpoint.
        """
        shape = (self.cov_rows, self.cov_cols)
        for name, flood in (('ever_reachable', self.possible),
                            ('ever_confirmed', self.confirmed)):
            have = getattr(self, name)
            if have is not None and have.shape != shape:
                have = None                    # the area was redrawn
            if flood is None or self.costmap is None:
                setattr(self, name, have)
                continue
            now = self._project(flood)
            setattr(self, name, now.copy() if have is None else (have | now))

    def _build_inside_mask(self):
        """Boolean grid marking which coverage cells are worth searching.

        Inside the polygon, not blocked by an obstacle, and not walled off
        with no way in. Excluding those matters: the camera can never see
        them, so counting them would hold the coverage percentage below 100%
        forever and make a complete search look like a failed one.

        The walled-off test uses the OPTIMISTIC flood (see _reachable_from),
        not the confirmed one. Ground is only dropped from the total once it
        is unreachable however the unseen parts turn out - merely not having
        looked there yet is not a reason to stop counting it.
        """
        avoiding = (bool(self.get_parameter('avoid_obstacles').value)
                    and self.costmap is not None)
        # Reachability accumulates; it does not come and go with the drone.
        self._remember_reachable()
        reach = self._reach_picture()[0] if avoiding else None
        # Being walled in is a fact about the map, not about where the drone
        # is, so it is allowed to override what has ever been reachable.
        sealed = self._project(self._sealed_mask(), default=False) if avoiding \
            else None
        mask = np.zeros((self.cov_rows, self.cov_cols), dtype=bool)
        # Kept alongside the searchable mask so the overlay can say WHY a cell
        # is not being searched instead of just leaving it blank. Accounting
        # and display only - planning reads the costmap directly.
        in_shape_mask = np.zeros_like(mask)
        blocked_mask = np.zeros_like(mask)
        # Cells inside the shape the operator drew, obstacles included. Kept so
        # the operator can be told how much of what they drew is actually
        # flyable - "100% of the searchable area" reads like the whole box was
        # done, when a third of it may have been rock.
        in_shape = 0
        for row in range(self.cov_rows):
            cell_y = self.cov_origin_y + (row + 0.5) * self.coverage_resolution
            for col in range(self.cov_cols):
                cell_x = self.cov_origin_x + (col + 0.5) * self.coverage_resolution
                if not self.point_in_polygon(cell_x, cell_y, self.polygon):
                    continue
                in_shape += 1
                in_shape_mask[row, col] = True
                if avoiding and self._is_blocked(cell_x, cell_y):
                    blocked_mask[row, col] = True
                    continue
                if sealed is not None and sealed[row, col]:
                    blocked_mask[row, col] = True      # ringed by wall
                    continue
                if reach is not None and not reach[row, col]:
                    blocked_mask[row, col] = True      # no way in found yet
                    continue
                if self.unverifiable is not None and self.unverifiable[row, col]:
                    continue
                mask[row, col] = True

        # A coverage cell is 0.5 m but the costmap is 0.15 m, and the cell is
        # judged by its CENTRE - so one inflated costmap cell landing on a
        # centre carves a whole 0.25 m2 square out of the search area. That
        # produced single-cell holes in the overlay and fragmented the
        # percentage. Specks of one or two cells are dropped: at this
        # resolution they are an artefact of sampling, not an obstacle. The
        # planner is unaffected, it reads the costmap itself.
        if blocked_mask.any():
            labels, count = ndimage.label(blocked_mask)
            if count:
                sizes = np.bincount(labels.ravel())
                speck = np.isin(labels, np.nonzero(sizes <= 2)[0]) & blocked_mask
                if speck.any():
                    blocked_mask &= ~speck
                    mask |= speck & in_shape_mask

        self.cells_in_shape = in_shape
        self.in_shape_mask = in_shape_mask
        self.blocked_mask = blocked_mask
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

    def _set_status(self, state, detail=''):
        """Say what the scout is doing, once per change.

        Published rather than logged because the operator is watching the
        map, not the terminal. The whole point is that "the green path
        disappeared and the drone flew out of the box" should read as
        LOOKING - going to get a view of ground it could not judge - and not
        as a crash.
        """
        if state == self.status_state and detail == self.status_detail:
            return
        self.status_state, self.status_detail = state, detail
        msg = String()
        msg.data = f'{state}|{detail}'
        self.status_pub.publish(msg)

    def _on_nav_plan(self, msg):
        """Keep the length of the route Nav2 means to fly."""
        total = 0.0
        for a, b in zip(msg.poses[:-1], msg.poses[1:]):
            total += math.hypot(b.pose.position.x - a.pose.position.x,
                                b.pose.position.y - a.pose.position.y)
        self.nav_plan_length = total if msg.poses else None

    def _slam_map_callback(self, msg):
        """Keep SLAM's map. Only _slam_walls reads it."""
        self.slam_map = msg

    def _slam_walls(self):
        """SLAM's walls, on the costmap's grid, as a bool array.

        Why bother, when we already have a costmap: the costmap is three
        times coarser and loses whole stretches of thin wall, so the mountain
        never closes in it however much the walls are inflated. SLAM's map
        traces the same outline nearly completely. Using it for the walled-in
        test - and nothing else - gets a sealed verdict without putting
        SLAM anywhere near the controller.

        The two maps are in different frames, so this goes through TF. That
        correction drifts by around half a metre, which does not matter here:
        the question is whether a few hundred square metres are ringed by
        metres of rock, not where a wall sits to the centimetre. It would
        matter for placing a waypoint, which is why that still uses the
        costmap.

        Returns None when there is no SLAM map yet.
        """
        if not self.get_parameter('use_slam_walls').value:
            return None
        if self.slam_map is None or self.costmap is None:
            return None
        key = (id(self.slam_map), id(self.costmap))
        if self._slam_walls_for == key:
            return self._slam_walls_cache

        sm, cm = self.slam_map, self.costmap
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, sm.header.frame_id,
                rclpy.time.Time(), timeout=Duration(seconds=0.1))
            dx = tf.transform.translation.x
            dy = tf.transform.translation.y
        except Exception:
            return None                      # no transform yet; skip quietly

        sres = sm.info.resolution
        occupied = (np.asarray(sm.data, dtype=np.int16)
                    .reshape(sm.info.height, sm.info.width) >= 65)
        if occupied.any():
            # A gap the drone cannot fit through is not a way through.
            grow = float(self.get_parameter('slam_wall_inflate').value)
            steps = max(1, int(round(grow / sres)))
            occupied = ndimage.binary_dilation(occupied, iterations=steps)

        # Resample onto the costmap grid. Costmap cell -> planning frame ->
        # SLAM frame -> SLAM cell.
        cres = cm.info.resolution
        cx = (cm.info.origin.position.x
              + (np.arange(cm.info.width) + 0.5) * cres) - dx
        cy = (cm.info.origin.position.y
              + (np.arange(cm.info.height) + 0.5) * cres) - dy
        cols = np.floor((cx - sm.info.origin.position.x) / sres).astype(int)
        rows = np.floor((cy - sm.info.origin.position.y) / sres).astype(int)
        ok_c = (cols >= 0) & (cols < sm.info.width)
        ok_r = (rows >= 0) & (rows < sm.info.height)
        out = np.zeros((cm.info.height, cm.info.width), dtype=bool)
        if ok_c.any() and ok_r.any():
            out[np.ix_(ok_r, ok_c)] = occupied[np.ix_(rows[ok_r], cols[ok_c])]

        self._slam_walls_for = key
        self._slam_walls_cache = out
        return out

    def _costmap_callback(self, msg):
        """Keep the latest costmap so the planner can ask what is blocked."""
        self.costmap = msg
        self._check_unknown_tracking(msg)

    def _check_unknown_tracking(self, msg):
        """Complain once if the costmap cannot express "never observed".

        Not flying at unreachable ground rests entirely on telling "never
        looked at" apart from "looked at and clear". Nav2 only marks the
        first as -1 when track_unknown_space is true on BOTH the global
        costmap and its obstacle layer; with it off both read 0, the two
        floods come out identical and the sweep plants waypoints inside rock.
        Logged as an ERROR because it degrades the whole feature silently.
        """
        if self._unknown_checked:
            return
        data = np.asarray(msg.data, dtype=np.int16)
        if np.any(data < 0):
            self._unknown_checked = True          # configured correctly
            return
        # All zeros is a costmap that has not been filled in yet, not a
        # misconfigured one. Wait until something has actually been mapped.
        if not np.any(data > 0):
            return
        self._unknown_checked = True
        self.get_logger().error(
            'COSTMAP MARKS NOTHING AS UNKNOWN: "never observed" and '
            '"observed and clear" are both 0, so they cannot be told apart. '
            'The inside of an obstacle will read as open ground and the sweep '
            'will plan waypoints in it. Fix: track_unknown_space: true on '
            'BOTH global_costmap and its obstacle_layer in the nav2 params.')

    def _is_blocked(self, x, y):
        """True if (x, y) in the planning frame is obstructed.

        Reads Nav2's costmap directly rather than keeping a second obstacle
        map of our own: it is already in the odom frame we plan in, it already
        inflates obstacles by the drone's size, and it is the same picture the
        controller will use in flight, so the plan and the flying agree.

        Anything outside the costmap is treated as clear - the costmap covers
        the flyable area, and refusing to plan beyond it would be worse than
        letting Nav2 handle whatever is out there.
        """
        m = self.costmap
        if m is None:
            return False
        res = m.info.resolution
        col = int((x - m.info.origin.position.x) / res)
        row = int((y - m.info.origin.position.y) / res)
        if not (0 <= col < m.info.width and 0 <= row < m.info.height):
            return False
        cost = m.data[row * m.info.width + col]
        if cost < 0:                 # -1 = never observed, treat as clear
            return False
        return cost >= self.get_parameter('obstacle_cost_threshold').value

    def _costmap_grid(self):
        """The costmap as a 2D int array, with its geometry. None if we have none."""
        m = self.costmap
        if m is None:
            return None
        grid = np.asarray(m.data, dtype=np.int16).reshape(
            m.info.height, m.info.width)
        return grid, m.info

    def _flood(self, passable, row, col, seed_radius=0):
        """Which passable cells join up with the one the drone is standing on?

        scipy labelling rather than a Python flood: same 4-connected answer,
        19 ms against 1.7 s on a full costmap, which is what lets this run
        every couple of seconds instead of once per plan.

        seed_radius starts from the NEAREST passable cell when the drone's own
        is not passable - routine, since on the pad its cell reads unobserved
        and at altitude it can sit inside an inflation ring. The nearest
        single cell, not a window: a window straddling a wall would seed both
        sides and call a sealed pocket reachable. None if nothing is near
        enough to start from.
        """
        h, w = passable.shape
        if not passable[row, col]:
            if seed_radius <= 0:
                return None
            r0, r1 = max(0, row - seed_radius), min(h, row + seed_radius + 1)
            c0, c1 = max(0, col - seed_radius), min(w, col + seed_radius + 1)
            window = passable[r0:r1, c0:c1]
            if not window.any():
                return None
            wr, wc = np.nonzero(window)
            wr, wc = wr + r0, wc + c0
            nearest = int(np.argmin((wr - row) ** 2 + (wc - col) ** 2))
            row, col = int(wr[nearest]), int(wc[nearest])
        labels, count = ndimage.label(passable)       # 4-connected by default
        if count == 0:
            return None
        return labels == labels[row, col]

    def _reachable_from(self, start_xy):
        """Optimistic flood: where could the drone get to if the unseen parts
        of the world turn out to be clear?

        Answers "could the drone actually get there?", which is a different
        question from "is that cell occupied?" - a waypoint can sit in a
        pocket with no way in without being in the rock itself.

        Unobserved ground counts as FREE here, which makes this the generous
        answer: anything this flood cannot reach is walled in no matter what
        turns out to be hiding in the unseen parts. Use it to rule ground OUT.
        To decide what is safe to fly at, use _confirmed_from instead.

        Returns a numpy bool grid matching the costmap, or None if there is no
        costmap or the drone is off-map or inside a wall.
        """
        got = self._costmap_grid()
        if got is None:
            return None
        grid, info = got
        thr = self.get_parameter('obstacle_cost_threshold').value
        free = (grid < thr) & (grid >= -1)        # -1 = unobserved, count as free
        walls = self._slam_walls()
        if walls is not None:
            free &= ~walls                        # see _slam_walls

        res = info.resolution
        col = int((start_xy[0] - info.origin.position.x) / res)
        row = int((start_xy[1] - info.origin.position.y) / res)
        if not (0 <= col < info.width and 0 <= row < info.height):
            return None                            # drone off-map
        # A drone hovering inside an obstacle's inflation ring reads as being
        # in a wall. Starting from the nearest free cell instead of refusing
        # to answer matters: callers that get None lose the map altogether and
        # stop checking reachability at all.
        return self._flood(free, row, col,
                           seed_radius=max(1, int(2.0 / res)))

    def _confirmed_from(self, start_xy):
        """Pessimistic flood: where can the drone get to across ground the
        lidar has actually seen to be clear?

        The only difference from _reachable_from is one line - unobserved
        cells count as WALL here, not floor - and that one line is what keeps
        waypoints out of the middle of a mountain. The lidar only ever sees a
        rock's near face, so its interior is never observed; counting
        unobserved ground as clear made that interior read as open AND let the
        flood arrive there by running round the ends of the mapped face.

        Returns a bool grid, which is all-False when nothing is confirmed yet
        (the honest answer on the pad, before the lidar has seen anything), or
        None if there is no costmap at all.
        """
        got = self._costmap_grid()
        if got is None:
            return None
        grid, info = got
        thr = self.get_parameter('obstacle_cost_threshold').value
        seen_clear = (grid >= 0) & (grid < thr)   # observed AND clear
        walls = self._slam_walls()
        if walls is not None:
            seen_clear &= ~walls                  # see _slam_walls

        res = info.resolution
        # Close the wedges between diverging lidar rays - see
        # observation_gap_fill. ANDed back against "not blocked" so this can
        # never fill in rock: it only stops a gap between two rays from
        # reading as a wall the flood cannot cross.
        gap = float(self.get_parameter('observation_gap_fill').value)
        if gap > 0:
            # A closing: grow, then shrink back by the same amount, which
            # fills gaps up to 2*iters cells wide and leaves everything else
            # where it was. Done as the two steps rather than binary_closing
            # so each gets the right border: the growth must not invent free
            # space beyond the edge of the map (border_value=0), and the
            # shrink must not eat the map's own edge away (border_value=1).
            iters = max(1, int(math.ceil(gap / 2.0 / res)))
            grown = ndimage.binary_dilation(
                seen_clear, iterations=iters, border_value=0)
            seen_clear = ndimage.binary_erosion(
                grown, iterations=iters, border_value=1) & (grid < thr)
        col = int((start_xy[0] - info.origin.position.x) / res)
        row = int((start_xy[1] - info.origin.position.y) / res)
        if not (0 <= col < info.width and 0 <= row < info.height):
            return None
        # Seeded from the nearest clear cell within 2 m - see _flood.
        reached = self._flood(seen_clear, row, col,
                              seed_radius=max(1, int(2.0 / res)))
        # An empty grid, not None: "nothing is confirmed yet" is a real answer
        # that callers act on by waiting, whereas None means "no map at all".
        return np.zeros_like(seen_clear) if reached is None else reached

    def _project(self, grid, default=True):
        """Resample a costmap-sized bool grid onto the coverage grid.

        The two grids have different resolutions and origins (0.15 m for the
        costmap, 0.5 m for coverage), so comparing them cell for cell needs
        one of them moved. Cells outside the costmap take 'default', which
        matches _is_reachable: beyond the map we do not claim to know, so we
        leave it to Nav2 rather than refusing to search there.
        """
        shape = (self.cov_rows, self.cov_cols)
        m = self.costmap
        if grid is None or m is None:
            return np.full(shape, default, dtype=bool)
        res = m.info.resolution
        xs = self.cov_origin_x + (np.arange(self.cov_cols) + 0.5) * self.coverage_resolution
        ys = self.cov_origin_y + (np.arange(self.cov_rows) + 0.5) * self.coverage_resolution
        cols = np.floor((xs - m.info.origin.position.x) / res).astype(int)
        rows = np.floor((ys - m.info.origin.position.y) / res).astype(int)
        ok_c = (cols >= 0) & (cols < m.info.width)
        ok_r = (rows >= 0) & (rows < m.info.height)
        out = np.full(shape, default, dtype=bool)
        if ok_c.any() and ok_r.any():
            out[np.ix_(ok_r, ok_c)] = grid[np.ix_(rows[ok_r], cols[ok_c])]
        return out

    def _polygon_mask_costmap(self):
        """The search polygon as a mask at costmap resolution.

        Vectorised even-odd crossing test over the polygon's bounding box
        only. point_in_polygon one cell at a time would be 284k Python calls;
        this is two array passes per edge.
        """
        m = self.costmap
        h, w, res = m.info.height, m.info.width, m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        mask = np.zeros((h, w), dtype=bool)
        poly = np.asarray(self.polygon, dtype=float)
        if poly.shape[0] < 3:
            return mask
        c0 = max(0, int((poly[:, 0].min() - ox) / res))
        c1 = min(w, int((poly[:, 0].max() - ox) / res) + 2)
        r0 = max(0, int((poly[:, 1].min() - oy) / res))
        r1 = min(h, int((poly[:, 1].max() - oy) / res) + 2)
        if c1 <= c0 or r1 <= r0:
            return mask
        xs = ox + (np.arange(c0, c1) + 0.5) * res
        ys = oy + (np.arange(r0, r1) + 0.5) * res
        gx, gy = np.meshgrid(xs, ys)
        inside = np.zeros(gx.shape, dtype=bool)
        x1, y1 = poly[-1]
        for x2, y2 in poly:
            if y1 != y2:
                crosses = (y1 > gy) != (y2 > gy)
                xint = (x2 - x1) * (gy - y1) / (y2 - y1) + x1
                inside ^= crosses & (gx < xint)
            x1, y1 = x2, y2
        mask[r0:r1, c0:c1] = inside
        return mask

    def _line_of_sight(self, xs, ys, target, solid, info):
        """Clear straight line from each of (xs, ys) to target? Vectorised.

        Unknown cells do NOT block sight - not having looked somewhere is the
        whole reason to go and look. Only mapped obstacle does.
        """
        tx, ty = target
        res = info.resolution
        far = float(np.max(np.hypot(xs - tx, ys - ty))) if xs.size else 0.0
        n = int(max(16, min(400, far / res)))
        t = np.linspace(0.0, 1.0, n)[None, :]
        px = xs[:, None] + (tx - xs[:, None]) * t
        py = ys[:, None] + (ty - ys[:, None]) * t
        cols = ((px - info.origin.position.x) / res).astype(np.int32)
        rows = ((py - info.origin.position.y) / res).astype(np.int32)
        inside = ((cols >= 0) & (cols < info.width)
                  & (rows >= 0) & (rows < info.height))
        np.clip(cols, 0, info.width - 1, out=cols)
        np.clip(rows, 0, info.height - 1, out=rows)
        return ~(solid[rows, cols] & inside).any(axis=1)

    def _in_obstacle_shadow(self, tx, ty, solid, info):
        """Is this point hidden behind mapped obstacle?

        The straight line from the NEAREST known-free ground to it. Inside a
        rock that line starts hard against the mapped face and crosses it at
        once; for ground merely unvisited it runs through unknown only.

        Used only to ORDER the targets of a look-around trip, never to
        exclude them - seen from one side, the inside of a rock and open
        ground beyond it are the same thing to the map, so this cannot tell
        them apart. Measured: it caught 32% of a rock interior while
        condemning 86% of the open ground past it.
        """
        idx = self._nearest_known()
        if idx is None:
            return False
        res = info.resolution
        col = int((tx - info.origin.position.x) / res)
        row = int((ty - info.origin.position.y) / res)
        if not (0 <= col < info.width and 0 <= row < info.height):
            return False
        nr = int(idx[0][row, col]); nc = int(idx[1][row, col])
        x0 = info.origin.position.x + (nc + 0.5) * res
        y0 = info.origin.position.y + (nr + 0.5) * res
        return not self._lines_clear(np.array([x0]), np.array([y0]),
                                     np.array([tx]), np.array([ty]),
                                     solid, info)[0]

    def _lines_clear(self, x0, y0, x1, y1, solid, info, skip_start=False):
        """Clear straight line from each (x0, y0) to its own (x1, y1)?

        The pairwise form of _line_of_sight, so the whole map can be tested
        in one go rather than a point at a time. Unknown cells do not block:
        only mapped obstacle does.

        skip_start ignores obstacle within the cell the viewer is standing
        in, because a viewer cannot occlude itself. The drone's own cell
        reads as obstacle more often than you would expect - sitting on the
        launch pad, or flying inside the inflated ring beside the mountain -
        and counting it failed EVERY line in the same call, so nothing at
        all was recorded as seen.

        It has to be a distance, not just the first sample: the samples are
        spaced by the LONGEST line in the batch, so on a shorter one several
        of the first can land inside the start cell.
        """
        res = info.resolution
        if x0.size == 0:
            return np.zeros(0, dtype=bool)
        length = np.hypot(x1 - x0, y1 - y0)
        far = float(np.max(length))
        n = int(max(8, min(160, far / res)))
        t = np.linspace(0.0, 1.0, n)[None, :]
        px = x0[:, None] + (x1 - x0)[:, None] * t
        py = y0[:, None] + (y1 - y0)[:, None] * t
        cols = ((px - info.origin.position.x) / res).astype(np.int32)
        rows = ((py - info.origin.position.y) / res).astype(np.int32)
        inside = ((cols >= 0) & (cols < info.width)
                  & (rows >= 0) & (rows < info.height))
        np.clip(cols, 0, info.width - 1, out=cols)
        np.clip(rows, 0, info.height - 1, out=rows)
        hit = solid[rows, cols] & inside
        if skip_start:
            # 1.5 cells clears the diagonal of the one the viewer is in.
            hit = hit & ((length[:, None] * t) > res * 1.5)
        return ~hit.any(axis=1)

    def _nearest_known(self):
        """For every cell, the index of the nearest known-free cell.

        One distance transform, cached per costmap. Two int arrays the size of
        the map, which is a few megabytes - the point being that the shadow
        test can then be answered one point at a time.

        It used to be computed for the WHOLE map at once: an array of
        (unknown cells x samples along each sight line), which on a real
        costmap is about 200000 x 160 = 32 million elements across six
        arrays, roughly 830 MB. That ran the node out of memory and killed it
        mid-flight. The answer is only ever wanted for the dozen targets a
        look-around trip considers, so it is computed for those.
        """
        if self.confirmed is None or not self.confirmed.any():
            return None
        key = id(self.costmap), id(self.confirmed)
        if self._shadow_cache_for == key:
            return self._shadow_cache
        _, idx = ndimage.distance_transform_edt(~self.confirmed,
                                                return_indices=True)
        self._shadow_cache_for = key
        self._shadow_cache = idx
        return idx

    def _frontier_goal(self):
        """Somewhere to fly to in order to SEE ground we cannot yet judge.

        Needed because planning only over confirmed ground is safe but, on its
        own, quietly incomplete. Legs are planned only where the drone can
        prove it can get to, and from the near side of a rock it can never
        prove anything about the far side - so it would search the near half,
        report itself finished, and silently abandon ground that is in fact
        wide open. Measured on this world's rock ridge: 369 m2 of clear ground
        past the ridge, never looked at.

        So when the confirmed sweep runs out and part of the area is still
        unverified, the drone flies to the edge of what it knows, nearest to
        the ground in question, and looks from there. That grows the confirmed
        region, _review_map() notices and replans, and the cycle repeats until
        the ground is either searched or proved to be rock. Standard
        frontier-driven exploration, aimed at the operator's area.

        Returns (x, y) in the planning frame, or None if there is nothing left
        worth going to look at.
        """
        if self.confirmed is None or self.possible is None or self.costmap is None:
            return None
        here = self._lookup_robot_xy()
        if here is None:
            return None
        unverified = self.possible & ~self.confirmed
        if not unverified.any():
            return None

        m = self.costmap
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y

        # Only worth going to look if the ground in question is inside the
        # area the operator drew. Unverified ground elsewhere is not our job.
        wanted = unverified & self._polygon_mask_costmap()
        if not wanted.any():
            return None
        wr, wc = np.nonzero(wanted)
        wx = ox + (wc + 0.5) * res
        wy = oy + (wr + 0.5) * res
        nearest = int(np.argmin(np.hypot(wx - here[0], wy - here[1])))
        target = (wx[nearest], wy[nearest])

        grid_now, info_now = self._costmap_grid()

        # The frontier is where confirmed ground touches unverified ground:
        # the places that, stood on, turn unknown into known. Note this uses
        # ALL the unverified ground, not just the part inside the area - to
        # see round a rock the drone usually has to step outside the box.
        edge = ndimage.binary_dilation(unverified) & self.confirmed
        if not edge.any():
            return None
        er, ec = np.nonzero(edge)
        ex = ox + (ec + 0.5) * res
        ey = oy + (er + 0.5) * res

        # Somewhere we are not already standing, or this would pick the
        # drone's own position and learn nothing, over and over.
        away = np.hypot(ex - here[0], ey - here[1]) > 1.0
        if not away.any():
            return None
        ex, ey = ex[away], ey[away]

        # Nor anywhere too close to mapped obstacle. A frontier cell is clear
        # when it is picked, but it sits by definition at the edge of what has
        # been seen, so the next scan often fills in rock beside it and the
        # inflation ring swallows it. The drone then flies at a waypoint that
        # turns unreachable on arrival, and because that counts as a FAILED
        # waypoint it spends one of the failure-replan budget: three of five
        # went that way in one flight, on vantage points at (3.7, 29.7),
        # (6.7, 30.0) and (9.7, 29.1), all hard against the same rock face.
        solid_now = grid_now >= self.get_parameter('obstacle_cost_threshold').value
        margin = float(self.get_parameter('vantage_obstacle_margin').value)
        if margin > 0 and solid_now.any():
            clear_by = ndimage.distance_transform_edt(~solid_now) * res
            cols_i = np.clip(((ex - ox) / res).astype(int), 0, info_now.width - 1)
            rows_i = np.clip(((ey - oy) / res).astype(int), 0, info_now.height - 1)
            roomy = clear_by[rows_i, cols_i] >= margin
            if roomy.any():                 # if none, take what there is
                ex, ey = ex[roomy], ey[roomy]

        # Nor anywhere already looked from. Some ground can never be settled
        # however long you stare at it - the inside of a rock is unverifiable
        # by definition, since no vantage point outside can see in - and
        # without this the drone ping-pongs between two neighbouring frontier
        # cells until it runs out of replans.
        if self.looked_from:
            sep = float(self.get_parameter('vantage_min_separation').value)
            fresh = np.ones(ex.shape, dtype=bool)
            for lx, ly in self.looked_from:
                fresh &= np.hypot(ex - lx, ey - ly) > sep
            if not fresh.any():
                return None
            ex, ey = ex[fresh], ey[fresh]

        # Now the test that makes this terminate: a vantage point is only
        # worth flying to if the ground in question can actually be SEEN from
        # it. The inside of a rock cannot be seen from anywhere outside it, so
        # without this the drone works its way along the frontier for ever,
        # hop after hop, learning nothing - measured at 14 wasted hops after
        # the search was already complete.
        solid, info = solid_now, info_now
        rng = float(self.get_parameter('sensor_range').value)

        # Try the nearest unverified ground first, then a spread of the rest:
        # one pocket being invisible does not mean they all are.
        order = np.argsort(np.hypot(wx - here[0], wy - here[1]))
        probes = order[np.linspace(0, order.size - 1,
                                   min(order.size, 12)).astype(int)]
        # Try targets NOT behind the mapped face first, then the rest.
        # Ordering, not exclusion: a cell behind the face is as likely to be
        # open ground beyond a rock as the inside of one, and skipping it
        # wrote off ground the drone could have reached by going round.
        # Preferring the others still spends the early trips where they are
        # most likely to pay, which is what cut the hopping along the face.
        lit = np.array([not self._in_obstacle_shadow(wx[i], wy[i],
                                                     solid_now, info_now)
                        for i in probes])
        if lit.size:
            probes = np.concatenate([probes[lit], probes[~lit]])

        for i in probes:
            target = (wx[i], wy[i])
            reach = np.hypot(ex - target[0], ey - target[1])
            near = reach <= rng
            if not near.any():
                continue
            seen = self._line_of_sight(ex[near], ey[near], target, solid, info)
            if not seen.any():
                continue                      # nothing can see this pocket
            cx, cy = ex[near][seen], ey[near][seen]
            best = int(np.argmin(np.hypot(cx - target[0], cy - target[1])))
            return float(cx[best]), float(cy[best])
        return None

    def _write_off_unverified(self):
        """Stop counting ground that has been shown to be uncheckable.

        Only called once every vantage point has been exhausted, so this is a
        conclusion rather than a guess - and it is reported to the operator in
        square metres at the same time, never applied quietly.
        """
        reach, conf = self._reach_picture()
        if reach is None or conf is None:
            return
        gone = self.inside_mask & reach & ~conf
        if self.unverifiable is None:
            self.unverifiable = gone
        else:
            self.unverifiable = self.unverifiable | gone
        self._refresh_searchable()

    def _worth_looking(self):
        """Is the unresolved ground big enough to be worth a trip?

        Below min_unverified_area the honest answer is no: it is reported to
        the operator and written off instead, which is both cheaper and
        steadier than chasing a patch whose size flickers cell by cell.
        """
        cells = self._unverified_cells()
        if not cells:
            return False            # nothing outstanding at all
        area = cells * self.coverage_resolution ** 2
        return area >= float(self.get_parameter('min_unverified_area').value)

    def _unverified_cells(self):
        """Coverage cells inside the area that cannot yet be called either way.

        Reachable if the unseen ground turns out to be clear, but not
        confirmed reachable across ground the lidar has actually looked at.
        This is what tells "the area is rock" apart from "we have not looked
        at the area yet", which are very different things to report.
        """
        reach, conf = self._reach_picture()
        if reach is None or conf is None:
            return 0
        return int(np.count_nonzero(self.inside_mask & reach & ~conf))

    def _review_map(self):
        """Re-derive what is reachable, and replan when the answer changes.

        This is what keeps the verdict honest, and it is the whole reason the
        no-go list could be deleted. Whether ground is reachable is a belief
        about where the drone has been able to look, so it is recomputed from
        the live map instead of remembered: ground written off on the approach
        is picked up again the moment the drone can see round the obstacle,
        and ground that turns out to be rock drops out of the search.

        It also makes the first plan harmless. The path is built about a
        second after the operator draws the area, while the drone is still on
        the pad with its lidar at 0.2 m - a map that has little to do with
        what it will see at survey height. That plan no longer has to be
        right, because this corrects it as soon as there is a real map.
        """
        if not self.area_set or not self.planned or self.replanning:
            return
        # A replan already asked for, waiting on a goal to cancel. Without
        # this the review fires again on the next period, reports the very
        # same change, and spends another of the replan budget doing it -
        # observed running 2/20, 3/20, 4/20 on a map that had not altered at
        # all, because self.replanning is not set until the cancel lands.
        if self._replan_wanted:
            return
        if self.paused():
            return
        if float(self.get_parameter('map_review_period').value) <= 0:
            return
        if not self.get_parameter('avoid_obstacles').value:
            return
        if not self.get_parameter('drop_unreachable').value or self.costmap is None:
            return
        here = self._lookup_robot_xy()
        if here is None:
            return
        confirmed = self._confirmed_from(here)
        if confirmed is None:
            return
        self.confirmed = confirmed
        self.possible = self._reachable_from(here)

        # Measured over the area the operator drew, NOT over what is left to
        # search: using the outstanding cells would fall as the drone flew and
        # trigger a replan on ordinary progress.
        reachable = int(np.count_nonzero(
            self.inside_mask & self._project(confirmed)))
        if self.plan_reachable_cells is None:
            self.plan_reachable_cells = reachable
            return

        change = reachable - self.plan_reachable_cells
        denom = max(1, self.cells_in_shape)
        if abs(change) / denom < float(self.get_parameter('map_review_fraction').value):
            return
        cap = int(self.get_parameter('max_map_replans').value)
        if self.map_replans >= cap:
            self.plan_reachable_cells = reachable   # stop re-reporting it
            self.get_logger().warn(
                f'What is reachable has changed again, but this area has '
                f'already been replanned from the map {cap} times - flying '
                f'the current path out instead.')
            return

        self.map_replans += 1
        # Take the new figure as the baseline NOW, not when the replan
        # eventually runs: the replan may be several seconds away and the
        # review must not keep reporting a change it has already acted on.
        self.plan_reachable_cells = reachable
        area = abs(change) * self.coverage_resolution ** 2
        opened = 'opened up' if change > 0 else 'turned out to be blocked'
        self.get_logger().info(
            f'The map has changed what is reachable: {area:.0f} m2 of the '
            f'search area has {opened} since this path was planned. '
            f'Replanning over what is reachable now (map replan '
            f'{self.map_replans}/{cap}).')

        # Reuses the same route as an abandoned waypoint: cancel what is in
        # flight, then rebuild the remaining sweep keeping searched ground.
        self._replan_is_map_driven = True
        self._replan_wanted = True
        if self.goal_handle is not None:
            self.timed_out = True
            self.goal_handle.cancel_goal_async()
        elif self.busy:
            self.timed_out = True        # dealt with when the goal resolves
        else:
            self._after_abandoned_goal()

    def _is_reachable(self, x, y, reached):
        """Look one point up in the grid from _reachable_from."""
        if reached is None:
            return True
        m = self.costmap
        res = m.info.resolution
        col = int((x - m.info.origin.position.x) / res)
        row = int((y - m.info.origin.position.y) / res)
        if not (0 <= col < m.info.width and 0 <= row < m.info.height):
            return True                            # outside the map: let Nav2 try
        return bool(reached[row, col])

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
        from TF: the EKF does not fuse z, so odom -> base_link stays at
        z = 0.0 however high the drone flies, and the only z in the
        odom -> camera_link chain is the camera's own 0.2 m mounting offset.
        Reading that gave a 0.2 m "altitude", a camera footprint smaller than
        one grid cell, and a coverage map that never marked anything at all.

        A parameter is the right answer rather than a stopgap: Nav2 is a 2D
        planner and never commands height, so the drone holds its takeoff
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
        if not self.area_set:
            return
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

        # Mark every cell whose centre falls inside the circle - EXCEPT the
        # ones the camera cannot have seen.
        #
        # It used to mark the lot. Flying past a rock, the footprint circle
        # laps over it and the ground inside got recorded as searched: 18.5 m2
        # of the inside of a mountain marked as covered, in one measured case.
        # That inflates the percentage and, worse, makes _already_searched
        # skip that ground on a replan - so the drone would decline to search
        # ground it had never actually seen.
        #
        # Two things disqualify a cell: it is mapped obstacle (the camera
        # surveys ground, not rock), or mapped obstacle stands between the
        # drone and it. The second is what rules out the inside of a rock the
        # drone is flying alongside.
        candidates = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cell_x = self.cov_origin_x + (col + 0.5) * self.coverage_resolution
                cell_y = self.cov_origin_y + (row + 0.5) * self.coverage_resolution
                if math.hypot(cell_x - robot_x, cell_y - robot_y) > radius:
                    continue
                if self._is_blocked(cell_x, cell_y):
                    continue
                candidates.append((row, col, cell_x, cell_y))

        if candidates and self.costmap is not None:
            grid, info = self._costmap_grid()
            # sight_block_threshold, not obstacle_cost_threshold: see the
            # parameter's comment. What stops the drone flying somewhere is
            # not what stops the camera seeing it.
            solid = grid >= self.get_parameter('sight_block_threshold').value
            xs = np.array([c[2] for c in candidates])
            ys = np.array([c[3] for c in candidates])
            seen = self._lines_clear(np.full(xs.shape, robot_x),
                                     np.full(ys.shape, robot_y),
                                     xs, ys, solid, info, skip_start=True)
        else:
            seen = np.ones(len(candidates), dtype=bool)

        for (row, col, _, _), ok in zip(candidates, seen):
            if ok:
                self.coverage[row, col] = 100

        self._publish_coverage()

    def _already_searched(self, x, y):
        """True if the camera has already covered this point.

        Used only when replanning. The coverage grid is deliberately NOT reset
        on a replan, so this is the record of everything swept before the
        obstacle stopped us.
        """
        col = int((x - self.cov_origin_x) / self.coverage_resolution)
        row = int((y - self.cov_origin_y) / self.coverage_resolution)
        if not (0 <= col < self.cov_cols and 0 <= row < self.cov_rows):
            return False
        return bool(self.coverage[row, col] > 0)

    def _publish_vantage(self, point):
        """Tell the ground station where the look-around trip is headed."""
        msg = PointStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y = float(point[0]), float(point[1])
        self.vantage_pub.publish(msg)

    def _publish_path(self):
        """Publish the planned SWEEP so the ground station can draw it.

        Sent once per plan rather than on a timer: the route only changes when
        the sweep is replanned, and the publisher is latched so a ground
        station starting later still receives it.

        The sweep legs only. A trailing go-and-look trip or the leg home is
        not part of the search pattern and must not be drawn as one: the
        vantage point sits outside the area by design - 33 m out, in one case
        - and drawing it in the same green dashes as the lawnmower made it
        look like a waypoint had been planned miles off target. The trip is
        reported on search_status while it is flown, and the ground station
        draws it as its own amber leg.
        """
        legs = self.points
        cut = self.vantage_index if self.vantage_index is not None else self.home_index
        if cut is not None:
            legs = self.points[:cut]

        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in legs:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

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
        # Cells that are not part of the search go out as -1, OccupancyGrid's
        # "unknown". The array itself only holds 0 and 100, which leaves a
        # subscriber unable to tell "inside the area, not swept yet" from
        # "outside the area, never meant to be swept" - and the grid is padded
        # 5 m beyond the shape, so that difference is most of the edge. Anyone
        # measuring progress from this message would read far too low a figure.
        #
        # See the COVER_* constants at the top of the file. Precedence
        # matters: ground the camera HAS been over stays marked searched even
        # if the costmap later calls it blocked, because we did look at it.
        published = np.full((self.cov_rows, self.cov_cols), COVER_OUTSIDE,
                            dtype=np.int8)
        in_shape = getattr(self, 'in_shape_mask', None)
        if in_shape is None:
            in_shape = self.inside_mask
        blocked = getattr(self, 'blocked_mask', None)
        if blocked is None:
            blocked = np.zeros_like(in_shape)

        # Inside the area and not confirmed reachable: the honest "unknown".
        unverified = in_shape & ~blocked & ~self.inside_mask
        _, conf = self._reach_picture()
        if conf is not None:
            unverified |= in_shape & ~blocked & ~conf

        # Unknown ground stays amber even when it sits behind the mapped
        # face, because "behind the face" does not mean "cannot check": from
        # one side, the inside of a rock and open ground beyond it are the
        # same thing to the map. Colouring it grey claimed otherwise and
        # measured 86% of the open ground north of the ridge condemned.
        published[in_shape] = COVER_UNVERIFIED
        published[in_shape & ~unverified] = COVER_TO_DO
        published[blocked] = COVER_BLOCKED
        published[in_shape & (self.coverage > 0)] = COVER_SEARCHED
        published = published.astype(np.int8)
        # OccupancyGrid data is a flat row-major list, same order as the array.
        grid.data = published.flatten().tolist()
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
        main()
