#!/usr/bin/env python3
"""AeroAid ground station  AMPCD-style operator interface.

    ros2 run 41068_ignition_bringup ground_station.py --ros-args \
        -r __ns:=/parrot1 -p robot_name:=parrot1

Layout follows multifunction-display convention: one large display area with
option select buttons around the bezel. The buttons are generic; the labels
change to suit whichever page is up, and the label tells you what the button
currently does.

Slot allocation is deliberate and fixed:
    left column    page selection  never changes, so it can be hit blind
    right column   actions for the current page
    bottom row     contact ruling  always available, because a decision on a
                   possible person should never be more than one press away
    E-STOP         bottom right, on every page. Stops every drone where it is;
                   only its own RESUME restarts them

Keyboard equivalents: 1-5 left, F1-F7 right, Q/W/R bottom, Esc cancels an
area drag, Space E-Stops (stop only - resuming takes a click on RESUME, so a
stray key can never set the drones moving). E raises a fake contact for
testing the interface.

The drones wait on their pads on the ute until LAUNCH is pressed. A search
area can be drawn on the aerial photo before then; the scout flies it once
it is airborne.

Qt owns the event loop; rclpy is pumped from a QTimer (see main()). Never
block inside a callback or the window freezes.
"""

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.executors import (ShutdownException, SingleThreadedExecutor,
                             TimeoutException)
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy, qos_profile_sensor_data)

from action_msgs.srv import CancelGoal
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import (Point32, PointStamped, PolygonStamped,
                               Twist)
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from ros_gz_interfaces.msg import Light
from vision_msgs.msg import Detection2DArray
import tf2_ros

from cv_bridge import CvBridge
import cv2
import yaml

from PyQt5.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer
from PyQt5.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter,
                         QPainterPath, QPen,
                         QPixmap)
from PyQt5.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout,
                             QLabel, QListWidget, QListWidgetItem,
                             QMainWindow, QPushButton, QShortcut, QSizePolicy,
                             QStackedWidget, QVBoxLayout, QWidget)

# ---------------------------------------------------------------------------
# Visual language
#
# Slate base rather than black: the operator is in a command vehicle in
# daylight and pure black panels glare against a bright cabin. Amber is the
# loudest colour in the palette and is reserved for one thing only  a contact
# nobody has ruled on yet. Nothing else is allowed to compete with it.
# ---------------------------------------------------------------------------
BASE  = "#2B001E"   # display background, behind the map
PANEL = "#160010"   # status strip, selected rows — recessed below the display
BEZEL = "#220F1C"   # window surround and button faces
LINE  = "#4A2A3C"   # borders and dividers, plum-tinted
TEXT  = "#EDE6EA"   # primary text, warmed a touch so it sits on plum
MUTED = "#9B8793"   # labels, secondary text, disabled

ALERT = "#D6FF00"   # unruled contact — the one thing demanding a decision
CONFIRM = "#5AE1E6"   # confirmed, committed area, LINK OK
RED   = "#FF4136"   # dismissed
DRONE  = "#FFFFFF"   # drone marker and flight trail

MAP_UNKNOWN = (62, 38, 54)      # unscanned, plum-dark
MAP_FREE    = (188, 188, 188)   # swept and clear — #BCBCBC
MAP_WALL    = (18, 18, 18)      # obstacle — #121212
# Ground the scout's camera has already swept. Deliberately low-contrast:
# it is reassurance, not a thing demanding a decision, and must not compete
# with a contact marker.
COVERED     = (90, 225, 230, 70)   # RGBA, drawn over the map
# Ground inside the area the scout has NOT settled. Three outcomes, and they
# are not interchangeable: "still to do", "I could not reach it" and "I never
# worked out WHETHER I could reach it". The third is the one that needs a
# decision from the operator, so it is the one drawn boldly. Collapsing all
# three into "no colour" drew unknown ground exactly like empty space off the
# edge of the map - on one flight that hid 478 of 619 unpainted cells.
TO_DO       = (90, 225, 230, 22)   # reachable, not swept yet — barely there
UNVERIFIED  = (235, 165, 40, 90)   # never confirmed reachable — AMBER, loud
BLOCKED_GND = (120, 120, 120, 70)  # obstacle or walled off — grey, inert
PLANNED     = "#00FF88"            # planned sweep route — bright green.
                                   # Kept clear of ALERT's yellow-green so
                                   # an unruled contact still reads as the
                                   # one thing wanting a decision.
COURIER     = "#FFD23F"            # courier and its route to the contact —
                                   # warm yellow, kept off ALERT's
                                   # yellow-green so it never reads as a
                                   # contact wanting a decision.

PAGE_MAP, PAGE_VIS, PAGE_IR, PAGE_CONTACTS, PAGE_SETTINGS = range(5)

# The scout's flashlight: always off, always on, or on while the search holds
# (THERMAL - heat holds the search, so the light comes on for the RGB check).
FLASHLIGHT_MODES = ("OFF", "ON", "THERMAL")
HEAT_STEP_K = 0.5    # one press of HEAT+ / HEAT- on the SETUP page

# Six option select buttons down the right bezel. The map page fills all six;
# other pages leave the spare slots blank rather than reflowing, so a given
# function stays in the same physical place.
RIGHT_SLOTS = 7


@dataclass
class Contact:
    """One candidate person detection.

    Placeholder until the team's Detection.msg exists. Keep the field names
    identical to the message fields and the swap is a one-line change.
    """
    id: int
    x: float
    y: float
    rgb_confidence: float
    thermal_confidence: float
    status: str = "UNCONFIRMED"
    # Where the detected object itself is on the ground (projected from the
    # YOLO box), as opposed to x, y, the drone's position when raised.
    obj_x: Optional[float] = None
    obj_y: Optional[float] = None
    # The GroundTracker track of the human that raised it. Dismissing the
    # contact dismisses the track, so the same person can't raise it again.
    track_id: Optional[int] = None


def match_one_to_one(points, anchors, gate: float) -> dict:
    """Pair points with anchors, closest pairs first, each used at most once,
    pairs gate metres or further apart never. Returns {point index: anchor
    index}. One-to-one is what lets the gate be wide without swallowing a
    second person: with two people beside one anchor, only one can take it.
    None points (not projectable) are never matched."""
    pairs = sorted(
        (np.hypot(p[0] - a[0], p[1] - a[1]), i, j)
        for i, p in enumerate(points) if p is not None
        for j, a in enumerate(anchors))
    matched, used = {}, set()
    for dist, i, j in pairs:
        if dist >= gate:
            break
        if i not in matched and j not in used:
            matched[i] = j
            used.add(j)
    return matched


EDGE_MARGIN = 0.05   # fraction of the frame; detector's edge_margin_frac


def box_region(mask: np.ndarray, bbox, width: int, height: int,
               pad: float = 0.25) -> Tuple[slice, slice]:
    """Rows and columns of the thermal mask under a YOLO box (RGB pixels,
    image width x height), padded by `pad` of the box size."""
    sx, sy = mask.shape[1] / width, mask.shape[0] / height
    cx, cy = bbox.center.position.x, bbox.center.position.y
    hw, hh = bbox.size_x * (0.5 + pad), bbox.size_y * (0.5 + pad)
    return (slice(int(max(cy - hh, 0) * sy), int((cy + hh) * sy) + 1),
            slice(int(max(cx - hw, 0) * sx), int((cx + hw) * sx) + 1))


def box_on_heat(mask: Optional[np.ndarray], bbox, width: int, height: int,
                min_px: int = 20, pad: float = 0.25) -> bool:
    """True if a YOLO box (RGB pixels, image width x height) covers body heat
    in the thermal mask, so a contact's heat and human are the same object.
    The box is padded by `pad` of its size for the small time offset between
    the two cameras' frames.

    shortcut: assumes the thermal and RGB cameras share mount and FOV
    (parrot.gazebo.xacro), so pixels line up after scaling; if they ever
    differ, compare ground positions instead.
    """
    if mask is None:
        return False
    return np.count_nonzero(mask[box_region(mask, bbox, width, height, pad)]) >= min_px


@dataclass
class Track:
    id: int
    x: float
    y: float
    last_seen: float
    dismissed: bool = False


class GroundTracker:
    """Follows each YOLO human from frame to frame by ground position.

    Projecting a box to the ground is only good to ~1-1.5 m while the drone
    moves (measured on the live sim), too loose to recognise a dismissed
    person by a fixed radius. Frame to frame, though, the same person moves
    little, so each detection joins the nearest track from the frames before
    (within gate, one-to-one) and the track follows it. A dismissed track
    keeps following its person however far the projection drifts, and is
    never dropped, so the person is still recognised on a later sweep row.
    Tracks not dismissed are dropped after timeout_s unseen.
    """

    def __init__(self, gate: float, timeout_s: float):
        self.gate = gate
        self.timeout_s = timeout_s
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, points, now: float) -> List[Optional[Track]]:
        """One frame's projected detections in, the track of each out (None
        for a point that could not be projected)."""
        self.tracks = [t for t in self.tracks
                       if t.dismissed or now - t.last_seen < self.timeout_s]
        matched = match_one_to_one(
            points, [(t.x, t.y) for t in self.tracks], self.gate)
        out: List[Optional[Track]] = []
        for i, p in enumerate(points):
            if p is None:
                out.append(None)
                continue
            if i in matched:
                track = self.tracks[matched[i]]
            else:
                track = Track(self._next_id, p[0], p[1], now)
                self._next_id += 1
                self.tracks.append(track)
            track.x, track.y, track.last_seen = p[0], p[1], now
            out.append(track)
        return out

    def get(self, track_id: Optional[int]) -> Optional[Track]:
        return next((t for t in self.tracks if t.id == track_id), None)

    def dismiss(self, track_id: Optional[int],
                fallback_xy: Optional[Tuple[float, float]] = None,
                now: float = 0.0) -> Optional[Track]:
        """Mark a track dismissed. If it is gone (or there never was one) but
        the object's position is known, start a dismissed track there."""
        track = self.get(track_id)
        if track is None and fallback_xy is not None:
            track = Track(self._next_id, fallback_xy[0], fallback_xy[1], now)
            self._next_id += 1
            self.tracks.append(track)
        if track is not None:
            track.dismissed = True
        return track

    def dismissed(self) -> List[Track]:
        return [t for t in self.tracks if t.dismissed]


def pixel_to_ground(u: float, v: float, width: int, height: int, hfov: float,
                    altitude: float, x: float, y: float,
                    yaw: float) -> Tuple[float, float]:
    """Ground position seen at pixel (u, v) of a camera looking straight
    down from (x, y) at `altitude` above flat ground, heading `yaw`.

    The cameras are pitched 90 deg down (camera_joint in parrot.urdf.xacro),
    so the top of the image is the drone's forward and the right of the
    image is its right. Checked on the live thermal feed against the known
    positions of person1, the kangaroo and the wombat: within ~0.3 m.
    """
    metres_per_px = 2.0 * altitude * np.tan(hfov / 2.0) / width
    forward = (height / 2.0 - v) * metres_per_px
    left = (width / 2.0 - u) * metres_per_px
    c, s = np.cos(yaw), np.sin(yaw)
    return (float(x + forward * c - left * s),
            float(y + forward * s + left * c))


# ---------------------------------------------------------------------------
# ROS
# ---------------------------------------------------------------------------
class GroundStationNode(Node):
    """Caches the latest data. No Qt, no drawing  the GUI reads whatever is
    current when it repaints, which decouples frame rate from message rate."""

    def __init__(self):
        super().__init__("ground_station")

        self.declare_parameter("robot_name", "parrot1")
        self.robot_name = str(self.get_parameter("robot_name").value)

        # 16-bit pixels are temperature / resolution. The <resolution> in
        # parrot.gazebo.xacro is not reaching the sensor, so it runs at the
        # Gazebo default of 0.01 K (checked on the live topic: 28800 = 288 K).
        # The display range is fixed to the sensor's min/max so the colours
        # mean the same thing on every frame (per-frame min-max would stretch
        # ambient noise to full scale).
        self.declare_parameter("thermal_resolution", 0.01)
        self.declare_parameter("thermal_min_k", 275.0)
        self.declare_parameter("thermal_max_k", 320.0)
        self.thermal_resolution = float(self.get_parameter("thermal_resolution").value)
        self.thermal_min_k = float(self.get_parameter("thermal_min_k").value)
        self.thermal_max_k = float(self.get_parameter("thermal_max_k").value)

        # Thermal body-heat check. Scenery sits at ambient (~288 K) and the
        # person at body heat (310 K), so a plain threshold is enough in sim.
        # Heat on its own never raises a contact (animals are warm too): it
        # holds the search so the RGB detector can check - see
        # MainWindow._check_detections. min_pixels is the hot area needed. From
        # the ~10 m survey altitude a whole person is roughly 1000 px, so 600
        # needs most of the body in frame instead of a sliver at the edge.
        self.declare_parameter("person_min_k", 305.0)
        self.declare_parameter("person_min_pixels", 600)
        # After a dismiss, let the drone fly on this long before checking
        # again, so it can move off the heat source it was holding over.
        self.declare_parameter("dismiss_cooldown_s", 2.0)
        # Every YOLO human is tracked frame to frame by its position on the
        # ground (GroundTracker), and a dismissed contact dismisses its track,
        # so the same person can't raise it again: not while the drone holds
        # over them, flies on, or comes back on a later row. track_gate is how
        # far a detection may be from a track's last position and still join
        # it (projection drifts up to ~1.6 m while moving, measured live).
        # Projection needs the cameras' fields of view (parrot.gazebo.xacro)
        # and height above ground, from odometry z once it arrives.
        self.declare_parameter("track_gate", 2.0)
        self.declare_parameter("track_timeout_s", 10.0)
        self.declare_parameter("rgb_hfov", 1.4)
        self.declare_parameter("thermal_hfov", 1.4)
        self.declare_parameter("flight_altitude", 10.0)
        self.tracker = GroundTracker(
            float(self.get_parameter("track_gate").value),
            float(self.get_parameter("track_timeout_s").value))
        self.rgb_hfov = float(self.get_parameter("rgb_hfov").value)
        self.thermal_hfov = float(self.get_parameter("thermal_hfov").value)
        self.drone_alt = float(self.get_parameter("flight_altitude").value)
        self.person_min_k = float(self.get_parameter("person_min_k").value)
        self.person_min_pixels = int(self.get_parameter("person_min_pixels").value)
        self.dismiss_cooldown_s = float(self.get_parameter("dismiss_cooldown_s").value)

        # A contact needs heat AND an RGB (YOLO) human. When heat is seen, the
        # search holds for up to rgb_check_s so the detector, single-digit
        # FPS on CPU, gets several frames of the spot. One RGB sighting counts
        # for rgb_fresh_s, since detector frames arrive slower than the GUI
        # checks. detector_stale_s without any detections message (empty or
        # not) means the detector is not running.
        self.declare_parameter("rgb_check_s", 6.0)
        self.declare_parameter("rgb_fresh_s", 1.5)
        self.declare_parameter("detector_stale_s", 5.0)
        self.rgb_check_s = float(self.get_parameter("rgb_check_s").value)
        self.rgb_fresh_s = float(self.get_parameter("rgb_fresh_s").value)
        self.detector_stale_s = float(self.get_parameter("detector_stale_s").value)
        # (best confidence, monotonic time) of the latest frame with a human.
        self.rgb_hit: Optional[Tuple[float, float]] = None
        # Ground position and track of the human in rgb_hit, if projected.
        self.rgb_hit_xy: Optional[Tuple[float, float]] = None
        self.rgb_hit_track: Optional[int] = None
        # (class name, monotonic time) of the latest frame with an ignored
        # class (kangaroo, wombat) in it. Never raises a contact; only lets
        # an animal's heat signature skip or end the RGB check early.
        self.animal_hit: Optional[Tuple[str, float]] = None
        self.animal_boxes: list = []     # bboxes from the frame of animal_hit
        self._detections_time = 0.0
        # (peak kelvin, hot pixel count) for the latest frame, None if nothing.
        self.thermal_hit: Optional[Tuple[float, int]] = None
        # Body-heat pixels of the latest thermal frame (dismissed heat removed).
        self.thermal_mask: Optional[np.ndarray] = None

        self.bridge = CvBridge()
        self.rgb_frame: Optional[np.ndarray] = None
        self.thermal_frame: Optional[np.ndarray] = None

        # AI (YOLO) detections overlay from yolo_person_detector.py, boxes
        # already drawn on by that node. display_frame is what the VIS page
        # and map PIP actually show: the AI overlay while that node is alive
        # and publishing, falling back to the plain RGB feed otherwise (e.g.
        # the detector node isn't running) so the camera view never just
        # goes blank because of a node that's merely optional.
        self.ai_frame: Optional[np.ndarray] = None
        self.display_frame: Optional[np.ndarray] = None
        self._ai_frame_time = 0.0
        self._ai_stale_after_s = 2.0

        self.map_image: Optional[np.ndarray] = None
        # Same grid as RGBA: unknown fully transparent, so an aerial photo
        # underneath shows through wherever SLAM has not looked yet.
        self.map_overlay: Optional[np.ndarray] = None
        self.map_resolution = 0.0
        self.map_origin = (0.0, 0.0)
        # SLAM's map frame (slam_params_<robot>.yaml). Known up front so a
        # search area drawn before SLAM starts can still be stamped in it;
        # the map header confirms it once SLAM is running.
        self.map_frame = f"{self.robot_name}_map"
        self.odom_frame = f"{self.robot_name}_odom"

        # Optional aerial photo of the world to draw under the map, written by
        # scripts/capture_aerial_map.py. Empty = no photo, plain map as before.
        # The photo is in Gazebo world coordinates while the map frame starts
        # where the scout spawned, so the spawn point is needed to line them
        # up. Keep it in step with parrot1's x/y in 41068_ignition.launch.py.
        self.declare_parameter("aerial_map", "")
        self.declare_parameter("scout_spawn_x", 2.0)
        self.declare_parameter("scout_spawn_y", 0.0)
        self.spawn_xy = (float(self.get_parameter("scout_spawn_x").value),
                         float(self.get_parameter("scout_spawn_y").value))
        self.aerial_image: Optional[np.ndarray] = None   # RGB
        self.aerial_bounds: Optional[Tuple[float, float, float, float]] = None
        self._load_aerial(str(self.get_parameter("aerial_map").value))

        # Coverage the scout has achieved, from its search_coverage grid.
        # Pre-rendered to an RGBA image in the callback rather than in
        # paintEvent, the same way the SLAM map is, so repaints stay cheap.
        self.coverage_image: Optional[np.ndarray] = None      # RGBA
        self.coverage_bounds: Optional[Tuple[float, float, float, float]] = None
        self.coverage_percent: Optional[float] = None
        # Progress reported three ways, because one number cannot say it.
        # "100% searched" with a third of the box unreachable is true and
        # badly misleading: a reader takes it to mean the box was checked.
        self.coverage_assigned: Optional[float] = None   # % of the box drawn
        self.coverage_gap_m2 = 0.0                       # not checked, m2
        self.coverage_gap_reason = ""                    # and why
        # Planned sweep route, in the map frame once transformed.
        self.search_path: List[Tuple[float, float]] = []
        # Kept in the frame they arrived in and converted when drawn - see
        # _on_path. Both are sent once per plan, so a transform that is not
        # ready at that moment must not lose them.
        self.search_path_raw: List[Tuple[float, float]] = []
        self.search_path_frame = ""
        self.vantage_raw: Optional[Tuple[float, float]] = None
        self.vantage_frame = ""
        # The courier's route to a confirmed contact, as Nav2 planned it, and
        # where the courier is now - both in this map's frame. Nav2 replans
        # about once a second while it is flying and stops when it is not
        # (arrived, dropped, E-Stopped), so a route older than
        # courier_plan_timeout_s is no longer being flown and is not drawn.
        self.declare_parameter("courier_name", "parrot2")
        self.declare_parameter("courier_plan_timeout_s", 3.0)
        self.courier_name = str(self.get_parameter("courier_name").value)
        self.courier_plan_timeout_s = float(
            self.get_parameter("courier_plan_timeout_s").value)
        self.courier_path: List[Tuple[float, float]] = []
        self.courier_path_time = 0.0
        self.courier_xy: Optional[Tuple[float, float]] = None
        self.courier_yaw = 0.0
        # RETURNING, HOME or RETURN_FAILED once the package is dropped.
        self.courier_state: Optional[str] = None
        # What the scout says it is doing. Without it the operator watches the
        # green path vanish and the drone fly out of the box with no way to
        # tell a finished search from a look-around trip from a crash.
        self.search_state: Optional[str] = None
        # Where a go-and-look trip is headed. Its own value, not the last pose
        # of search_path: it is not a sweep leg, and drawing it as one made a
        # waypoint 33 m outside the area look like a planning fault.
        self.vantage_xy: Optional[Tuple[float, float]] = None
        self.search_detail = ""

        self.drone_xy: Optional[Tuple[float, float]] = None
        self.drone_yaw = 0.0
        self.trail = deque(maxlen=3000)

        self.base_frame = f"{self.robot_name}_base_link"
        self.tf_buffer = tf2_ros.Buffer()
        # On its own thread. /tf is by far the busiest topic here, and pumped
        # through the Qt timer with everything else it queued up and arrived
        # seconds late - so the drone marker trailed the real drone, and a
        # contact raised at the marker landed where the drone used to be.
        # The buffer is thread-safe, and nothing else runs on that thread.
        # It also means lookups never need a timeout: the latest transform is
        # already in the buffer or it is not, so the GUI never waits on one.
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True)

        # SLAM publishes the map transient-local. A default subscription
        # receives nothing at all, with no error  this QoS is required.
        map_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(Image, "camera/image",
                                 self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, "camera/thermal",
                                 self._on_thermal, qos_profile_sensor_data)
        # Published by yolo_person_detector.py (scripts/yolo_person_detector.py) -
        # optional node, not started by this one. Topic name has to match its
        # 'annotated_topic' parameter (default below is that default).
        self.create_subscription(Image, "camera/person_detections_image",
                                 self._on_ai, qos_profile_sensor_data)
        # Same node's structured output (its 'detections_topic'). This is
        # what raises contacts; the image above is only for display.
        self.create_subscription(Detection2DArray, "camera/person_detections",
                                 self._on_detections, 10)
        # Same node's ignored classes (its 'ignored_detections_topic'):
        # animals the model recognises. Display-only on the image above;
        # here they only stop an animal's body heat holding the search.
        self.create_subscription(Detection2DArray, "camera/animal_detections",
                                 self._on_animal_detections, 10)
        # Only for height above ground: TF z is always 0 (the EKF is 2D),
        # odometry z is the real height (ground in large_demo is at z ~ 0).
        self.create_subscription(Odometry, "odometry", self._on_odometry, 10)
        self.create_subscription(OccupancyGrid, "map", self._on_map, map_qos)
        # The scout publishes this with default (volatile) QoS, so a plain
        # subscription matches it.
        self.create_subscription(OccupancyGrid, "search_coverage",
                                 self._on_coverage, 10)
        # Latched by the scout, so the route arrives even if this starts later.
        path_qos = QoSProfile(depth=1,
                              reliability=QoSReliabilityPolicy.RELIABLE,
                              durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, "search_path", self._on_path, path_qos)
        # The courier's Nav2 planner output and its position. Absolute, since
        # this node runs in the scout's namespace.
        self.create_subscription(Path, f"/{self.courier_name}/plan",
                                 self._on_courier_plan, 10)
        self.create_subscription(Odometry, f"/{self.courier_name}/odometry",
                                 self._on_courier_odometry, 10)
        self.create_subscription(String, "/aeroaid/courier_state",
                                 self._on_courier_state, path_qos)
        self.create_subscription(String, "search_status", self._on_status, path_qos)
        self.create_subscription(PointStamped, "search_vantage",
                                 self._on_vantage, path_qos)

        self.decision_pub = self.create_publisher(String, "operator/decision", 10)

        # Tells search_node to stop and hover. Transient local so a search
        # node started later still picks up the current state.
        hold_qos = QoSProfile(depth=1,
                              reliability=QoSReliabilityPolicy.RELIABLE,
                              durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.hold_pub = self.create_publisher(Bool, "search/hold", hold_qos)
        self.holding = False
        self.hold_pub.publish(Bool(data=False))

        # Night mode (NIGHT=true in run_blue_mountains.sh): the world is dark,
        # so the scout's floodlight follows the search hold. Heat holds the
        # search, the light comes on, and the RGB detector can then see what
        # the thermal camera found; it goes off when the search resumes.
        self.declare_parameter("night_mode", False)
        self.night_mode = bool(self.get_parameter("night_mode").value)
        # The operator can change all of the settings below on the SETUP
        # page; these parameters only pick where they start.
        #
        # flashlight_mode: OFF, ON, or THERMAL (on exactly while the search
        # holds, which is what night mode does). Empty picks THERMAL at night
        # and OFF by day.
        # Any type: the command line reads on / off / ON / OFF as YAML
        # booleans, which a plain string parameter refuses with a crash.
        self.declare_parameter("flashlight_mode", "",
                               ParameterDescriptor(dynamic_typing=True))
        mode = self.get_parameter("flashlight_mode").value
        if isinstance(mode, bool):
            mode = "ON" if mode else "OFF"
        mode = str(mode).upper()
        if mode not in FLASHLIGHT_MODES:
            if mode:
                self.get_logger().warn(
                    f"flashlight_mode '{mode}' is not one of "
                    f"{', '.join(FLASHLIGHT_MODES)}; using the default.")
            mode = "THERMAL" if self.night_mode else "OFF"
        self.flashlight_mode = mode
        # Whether the light needs re-sending each second. Off until it is
        # used, so a day run that never touches the light publishes nothing
        # to it, as before this setting existed.
        self._light_active = mode != "OFF"
        self.light_pub = self.create_publisher(Light, "flashlight", 10)
        # Re-sent each second: nothing latches on the Gazebo side, so a
        # message sent before the bridge connects would be lost.
        self.create_timer(1.0, self._resend_light)

        # Which detectors may raise a contact. Both on is the normal rule:
        # body heat AND a YOLO human. Thermal off (e.g. a hot day, when the
        # ground is as warm as a person) means a YOLO human alone raises a
        # contact; YOLO off means body heat alone does. At least one stays on.
        self.declare_parameter("thermal_detection", True)
        self.declare_parameter("rgb_detection", True)
        self.thermal_enabled = bool(self.get_parameter("thermal_detection").value)
        self.rgb_enabled = bool(self.get_parameter("rgb_detection").value)
        if not (self.thermal_enabled or self.rgb_enabled):
            self.get_logger().warn(
                "thermal_detection and rgb_detection are both off, so nothing "
                "could raise a contact. Turning thermal detection back on.")
            self.thermal_enabled = True
        # Ground position of the hot area in thermal_hit, for a contact
        # raised on heat alone (YOLO off).
        self.thermal_hit_xy: Optional[Tuple[float, float]] = None

        # Tells the courier drone where to fly. Transient local so a courier
        # started after the confirm still gets the last confirmed position.
        self.confirmed_point_pub = self.create_publisher(
            PointStamped, "operator/confirmed_point", hold_qos)

        # Operator-drawn search area. PolygonStamped rather than a custom
        # message so nothing needs building and search_node can subscribe
        # without waiting on an interface build; same transient-local
        # reasoning as the two publishers above.
        self.area_pub = self.create_publisher(
            PolygonStamped, "search_area_request", hold_qos)

        # Lifts both drones off the ute (scripts/takeoff_node.py). Absolute,
        # since the courier listens too, and transient local so a takeoff
        # node that is slow to start still gets the press.
        self.launch_pub = self.create_publisher(Bool, "/aeroaid/launch", hold_qos)
        self.launched = False

        # E-Stop for every drone (search_node, courier_node, takeoff_node).
        # Absolute and transient local, like the launch topic, so a node that
        # starts while it is on still stops. Separate from search/hold, which
        # the contact logic releases on its own.
        self.estop_pub = self.create_publisher(Bool, "/aeroaid/estop", hold_qos)
        self.estopped = False
        self.estop_pub.publish(Bool(data=False))

        # The button does not trust the drones' own nodes.
        #
        # It used to do one thing: publish /aeroaid/estop and leave each node
        # to cancel its own Nav2 goal. That works only while those nodes are
        # healthy, and the one time it was really needed they were not - the
        # search node had been killed by a MemoryError and the scout flew on
        # with nobody left to hear the button. A stop that needs the thing it
        # is stopping to still be working is not a stop.
        #
        # So the ground station now also stops the drones itself, by two
        # routes that between them cover both ways a drone can still be
        # moving with its node gone:
        #
        #   cancel_goal - cancels whatever Nav2 is executing, whoever asked
        #   for it. In the action protocol a CancelGoal request with a zero
        #   goal id and a zero stamp means every goal, so a default-built
        #   request stops a goal this process never sent.
        #
        #   cmd_vel held at zero - for the climb, which takeoff_node drives
        #   as raw velocity with Nav2 not involved at all, and for a last
        #   command left standing after Nav2 itself has gone. Held, not sent
        #   once, because the drone keeps the last velocity it was given.
        self.declare_parameter("estop_cancel_nav", True)
        self.declare_parameter("estop_zero_velocity", True)
        self.declare_parameter("estop_zero_period_s", 0.1)
        estop_drones = [self.robot_name, self.courier_name]
        self.estop_cancel = {
            name: self.create_client(
                CancelGoal, f"/{name}/navigate_to_pose/_action/cancel_goal")
            for name in estop_drones}
        self.estop_cmd = {
            name: self.create_publisher(Twist, f"/{name}/cmd_vel", 10)
            for name in estop_drones}
        self.estop_zero_timer = None

        self.create_timer(0.1, self._poll_pose)

        self.get_logger().info(f"Ground station up for {self.robot_name}")

    def _load_aerial(self, path: str):
        """Read the aerial photo and work out where it sits in the map frame.

        Any problem is a warning, not a crash: the photo is a convenience and
        the ground station is fully usable without it.
        """
        if not path:
            return
        try:
            with open(path) as f:
                meta = yaml.safe_load(f)
            img_path = os.path.join(os.path.dirname(path), meta["image"])
            bgr = cv2.imread(img_path)
            if bgr is None:
                raise IOError(f"cannot read {img_path}")
        except Exception as exc:
            self.get_logger().warn(f"No aerial map ({path}): {exc}")
            return

        sx, sy = self.spawn_xy
        self.aerial_image = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        # World -> map frame is a pure shift by the spawn point: SLAM starts
        # its map where the scout starts, with no rotation.
        self.aerial_bounds = (float(meta["x_min"]) - sx, float(meta["y_min"]) - sy,
                              float(meta["x_max"]) - sx, float(meta["y_max"]) - sy)
        self.get_logger().info(f"Aerial map loaded: {img_path}")

    def _on_rgb(self, msg: Image):
        try:
            self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Reclaim the display feed from a dead/never-started AI node.
            # Runs on every RGB frame (always arriving, detector or not),
            # so it's the natural heartbeat to check AI staleness against -
            # no separate timer needed.
            ai_fresh = (time.monotonic() - self._ai_frame_time) < self._ai_stale_after_s
            if not ai_fresh:
                self.display_frame = self.rgb_frame
        except Exception as exc:
            self.get_logger().warn(f"RGB decode failed: {exc}", once=True)

    def _on_ai(self, msg: Image):
        try:
            self.ai_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self._ai_frame_time = time.monotonic()
            self.display_frame = self.ai_frame
        except Exception as exc:
            self.get_logger().warn(f"AI detections decode failed: {exc}", once=True)

    def _on_odometry(self, msg: Odometry):
        self.drone_alt = msg.pose.pose.position.z

    def _pose_at(self, stamp) -> Optional[Tuple[float, float, float]]:
        """(x, y, yaw) of the drone when an image was taken, from TF history.
        Detector frames arrive up to a second late and the drone may be
        flying, so the pose at processing time would put the object in the
        wrong place. Falls back to the latest pose."""
        for frame in (self.map_frame, self.odom_frame):
            try:
                tf = self.tf_buffer.lookup_transform(
                    frame, self.base_frame, rclpy.time.Time.from_msg(stamp))
            except Exception:
                continue
            q = tf.transform.rotation
            yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
            return tf.transform.translation.x, tf.transform.translation.y, yaw
        if self.drone_xy is None:
            return None
        return self.drone_xy[0], self.drone_xy[1], self.drone_yaw

    def ground_point(self, u: float, v: float, width: int, height: int,
                     hfov: float, stamp) -> Optional[Tuple[float, float]]:
        pose = self._pose_at(stamp)
        if pose is None:
            return None
        return pixel_to_ground(u, v, width, height, hfov, self.drone_alt, *pose)

    def _on_detections(self, msg: Detection2DArray):
        now = time.monotonic()
        self._detections_time = now
        if not self.rgb_enabled:
            return            # still alive (above), but its boxes count for nothing
        if self.rgb_frame is not None:
            height, width = self.rgb_frame.shape[:2]
        else:
            width, height = 720, 480      # parrot.gazebo.xacro rgbd_camera
        points = [self.ground_point(d.bbox.center.position.x,
                                    d.bbox.center.position.y,
                                    width, height, self.rgb_hfov,
                                    msg.header.stamp)
                  for d in msg.detections]
        tracks = self.tracker.update(points, now)
        best = None
        for d, xy, track in zip(msg.detections, points, tracks):
            if track is not None and track.dismissed:
                continue      # the operator already ruled this one out
            if (self.thermal_enabled
                    and not box_on_heat(self.thermal_mask, d.bbox, width, height)):
                continue      # heat elsewhere in frame (e.g. a wombat) doesn't back this box
            for r in d.results:
                if best is None or r.hypothesis.score > best[0]:
                    best = (r.hypothesis.score, xy,
                            track.id if track is not None else None)
        if best is not None:
            self.rgb_hit = (best[0], now)
            self.rgb_hit_xy = best[1]
            self.rgb_hit_track = best[2]

    def fresh_rgb_hit(self) -> Optional[float]:
        """Confidence of a human seen in the last rgb_fresh_s, else None."""
        if self.rgb_hit is None or not self.rgb_enabled:
            return None
        conf, seen = self.rgb_hit
        return conf if time.monotonic() - seen < self.rgb_fresh_s else None

    def _on_animal_detections(self, msg: Detection2DArray):
        if not self.rgb_enabled:
            return
        best = max(((r.hypothesis.score, r.hypothesis.class_id)
                    for d in msg.detections for r in d.results), default=None)
        if best is not None:
            self.animal_hit = (best[1], time.monotonic())
            self.animal_boxes = [d.bbox for d in msg.detections]

    def fresh_animal_only(self) -> Optional[str]:
        """Class name of an animal seen in the last rgb_fresh_s with no human
        seen in that time, else None. A human always wins: an animal next to
        a person must never stop the person being checked.

        Also None while there is body heat outside the animals' boxes: that
        heat is something else (a person the detector hasn't seen yet, e.g.
        beside the wombat), so the check must carry on."""
        if self.animal_hit is None or self.fresh_rgb_hit() is not None:
            return None
        name, seen = self.animal_hit
        if time.monotonic() - seen >= self.rgb_fresh_s:
            return None
        if self.thermal_mask is not None:
            if self.rgb_frame is not None:
                height, width = self.rgb_frame.shape[:2]
            else:
                width, height = 720, 480      # parrot.gazebo.xacro rgbd_camera
            rest = self.thermal_mask.copy()
            for b in self.animal_boxes:
                rest[box_region(rest, b, width, height)] = False
            if np.count_nonzero(rest) >= self.person_min_pixels // 4:
                return None
        return name

    def detector_alive(self) -> bool:
        return time.monotonic() - self._detections_time < self.detector_stale_s

    def _on_thermal(self, msg: Image):
        """Thermal may arrive 16-bit, 8-bit or colour depending on the xacro.
        Handle all three, and false-colour mono so heat reads instantly rather
        than as grey mush."""
        try:
            if msg.encoding in ("mono16", "16UC1"):
                # Gazebo's default (L16). Letting cv_bridge convert to bgr8
                # scales by 255/65535, which leaves the whole image near black.
                raw = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                kelvin = raw.astype(np.float32) * self.thermal_resolution
                span = self.thermal_max_k - self.thermal_min_k
                scaled = (kelvin - self.thermal_min_k) * (255.0 / span)
                mono = np.clip(scaled, 0, 255).astype(np.uint8)
                frame = cv2.applyColorMap(mono, cv2.COLORMAP_INFERNO)
                if self.thermal_enabled:
                    self.thermal_hit = self._detect_person(kelvin, frame,
                                                           msg.header.stamp)
                else:
                    # Still shown on the IR page, but never holds or backs
                    # anything.
                    self.thermal_hit = self.thermal_mask = None
                    self.thermal_hit_xy = None
                self.thermal_frame = frame
            elif msg.encoding in ("mono8", "8UC1"):
                mono = self.bridge.imgmsg_to_cv2(msg, "mono8")
                self.thermal_frame = cv2.applyColorMap(mono, cv2.COLORMAP_INFERNO)
            else:
                self.thermal_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Thermal decode failed: {exc}", once=True)

    def _detect_person(self, kelvin: np.ndarray, frame: np.ndarray,
                       stamp=None) -> Optional[Tuple[float, int]]:
        """Threshold for body heat. Boxes the hot region on the frame so a
        frozen frame shows what triggered the check.

        Each dismissed track claims the one hot blob nearest it (within the
        track gate) and that blob is left out, so the heat of a person the
        operator already ruled out doesn't hold the search again, while
        anyone else's heat still counts.

        A blob touching the frame edge is left out too: it is a body only
        partly in view, and holding there leaves the person at the edge of
        the RGB frame, where the detector rightly won't count them. The
        search flies on until the whole body is in view."""
        hot = kelvin >= self.person_min_k
        if hot.any():
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(
                hot.astype(np.uint8))
            h, w = hot.shape
            # Same margin as the detector's edge_margin_frac default.
            mx, my = EDGE_MARGIN * w, EDGE_MARGIN * h
            drop = {i for i in range(1, n)
                    if stats[i][0] < mx or stats[i][1] < my
                    or stats[i][0] + stats[i][2] > w - mx
                    or stats[i][1] + stats[i][3] > h - my}
            dismissed = self.tracker.dismissed()
            if dismissed and stamp is not None:
                blobs = [self.ground_point(centroids[i][0], centroids[i][1],
                                           w, h, self.thermal_hfov, stamp)
                         for i in range(1, n)]
                drop |= {i + 1 for i in match_one_to_one(
                    blobs, [(t.x, t.y) for t in dismissed], self.tracker.gate)}
            for i in drop:
                hot[labels == i] = False
        self.thermal_mask = hot
        self.thermal_hit_xy = None
        count = int(np.count_nonzero(hot))
        if count < self.person_min_pixels:
            return None
        if stamp is not None:
            # Where the heat is on the ground, for a contact raised on heat
            # alone when YOLO is switched off.
            rows, cols = np.nonzero(hot)
            self.thermal_hit_xy = self.ground_point(
                float(cols.mean()), float(rows.mean()), hot.shape[1],
                hot.shape[0], self.thermal_hfov, stamp)
        x, y, w, h = cv2.boundingRect(hot.astype(np.uint8))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 214), 2)  # ALERT
        return float(kelvin[hot].max()), count

    def _on_map(self, msg: OccupancyGrid):
        w, h = msg.info.width, msg.info.height
        if w == 0 or h == 0:
            return
        grid = np.asarray(msg.data, dtype=np.int8).reshape(h, w)

        img = np.empty((h, w, 3), dtype=np.uint8)
        img[:] = MAP_UNKNOWN
        img[grid == 0] = MAP_FREE
        img[grid > 50] = MAP_WALL

        # Row 0 of an OccupancyGrid is the bottom of the world; row 0 of an
        # image is the top. Flip once here so everything downstream can treat
        # the map like an ordinary picture.
        self.map_image = np.flipud(img).copy()

        # Over a photo: scanned ground gets a light wash so "where have we
        # looked" still reads, walls stay dark, unknown lets the photo through.
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[grid == 0] = (*MAP_FREE, 70)
        overlay[grid > 50] = (*MAP_WALL, 200)
        self.map_overlay = np.flipud(overlay).copy()

        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_frame = msg.header.frame_id

    def _on_coverage(self, msg: OccupancyGrid):
        """Turn the scout's coverage grid into a translucent overlay.

        The grid is published in the scout's ODOM frame while this map is
        drawn in its MAP frame. Those two differ by SLAM's running correction
        - around a metre, and it moves during a flight - so the offset is
        looked up from TF rather than assumed to be zero. If the lookup is not
        available yet the overlay is simply held back; drawing it in the wrong
        place would be worse than not drawing it.
        """
        w, h = msg.info.width, msg.info.height
        if w == 0 or h == 0:
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, msg.header.frame_id,
                rclpy.time.Time())
        except Exception:
            return

        grid = np.asarray(msg.data, dtype=np.int8).reshape(h, w)

        # Must match the COVER_* constants in scripts/search_node.py.
        SEARCHED, BLOCKED_V, UNVERIFIED_V, TO_DO_V = 100, 50, 25, 0

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[grid == TO_DO_V] = TO_DO
        rgba[grid == BLOCKED_V] = BLOCKED_GND
        rgba[grid == UNVERIFIED_V] = UNVERIFIED
        rgba[grid == SEARCHED] = COVERED    # outside the area stays clear
        # Row 0 of an OccupancyGrid is the bottom of the world, row 0 of an
        # image is the top - same flip the SLAM map does.
        self.coverage_image = np.flipud(rgba).copy()

        res = msg.info.resolution
        ox = msg.info.origin.position.x + tf.transform.translation.x
        oy = msg.info.origin.position.y + tf.transform.translation.y
        self.coverage_bounds = (ox, oy, ox + w * res, oy + h * res)

        # Three figures, from the one grid. Everything with a value at all
        # is inside the area the operator drew; -1 is outside it and says
        # nothing, which is why the 5 m pad around the shape cannot skew this.
        cell = res * res
        searched = int(np.count_nonzero(grid == SEARCHED))
        to_do = int(np.count_nonzero(grid == TO_DO_V))
        unverified = int(np.count_nonzero(grid == UNVERIFIED_V))
        blocked = int(np.count_nonzero(grid == BLOCKED_V))
        assigned = searched + to_do + unverified + blocked

        # Progress against what can actually be flown - the "am I done?"
        # number, which does reach 100%.
        reachable = searched + to_do
        self.coverage_percent = (100.0 * searched / reachable) if reachable else None
        # Progress against what was ASSIGNED - the honest one, which does not
        # reach 100% when part of the box is rock.
        self.coverage_assigned = (100.0 * searched / assigned) if assigned else None
        self.coverage_gap_m2 = (unverified + blocked) * cell
        if unverified and blocked:
            self.coverage_gap_reason = "unknown + terrain"
        elif unverified:
            self.coverage_gap_reason = "never verified"
        elif blocked:
            self.coverage_gap_reason = "enclosed by terrain"
        else:
            self.coverage_gap_reason = ""

    def _on_status(self, msg) -> None:
        """Store the scout's own account of what it is doing.

        Sent as "STATE|detail" so the state can drive the colour while the
        detail carries the specifics (which waypoint, how far off, why).
        """
        state, _, detail = msg.data.partition("|")
        self.search_state = state.strip() or None
        self.search_detail = detail.strip()

    def _on_vantage(self, msg) -> None:
        """Store where the scout is going to look, in its own frame.

        Converted when drawn, for the same reason as the sweep: it is sent
        once per plan, so a transform that is not ready yet must not lose it.
        """
        self.vantage_raw = (msg.point.x, msg.point.y)
        self.vantage_frame = msg.header.frame_id

    def _on_path(self, msg: Path):
        """Store the scout's planned sweep, in the frame it arrived in.

        Kept raw and converted when it is drawn, not here. The sweep is
        published ONCE per plan, so a transform that is not ready at that
        instant used to lose the route for good - which is what happened on
        a restart, where the ground station comes up a few seconds before the
        scout and SLAM. Converting at draw time also keeps the route in step
        with the search box as SLAM's correction drifts, instead of freezing
        whatever offset happened to apply when the message landed.
        """
        self.search_path_raw = [(q.pose.position.x, q.pose.position.y)
                                for q in msg.poses]
        self.search_path_frame = msg.header.frame_id

    def _odom_to_map_shift(self) -> Tuple[float, float]:
        """Offset from the scout's odom frame to this map's frame.

        Zero before SLAM starts: until then the view is drawn in odom, as
        _poll_pose explains.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.odom_frame, rclpy.time.Time())
        except Exception:
            return 0.0, 0.0
        return tf.transform.translation.x, tf.transform.translation.y

    def _on_courier_plan(self, msg: Path):
        """Store the courier's planned route in this map's frame.

        The courier's odom frame and the scout's are both the Gazebo world
        frame (see courier_node.py), so the courier's plan is already in the
        scout's odom frame and needs only the same shift as search_path.
        """
        dx, dy = self._odom_to_map_shift()
        self.courier_path = [(p.pose.position.x + dx, p.pose.position.y + dy)
                             for p in msg.poses]
        self.courier_path_time = time.monotonic()

    def _on_courier_odometry(self, msg: Odometry):
        # Raw Gazebo odometry is world pose, i.e. the shared odom frame.
        dx, dy = self._odom_to_map_shift()
        self.courier_xy = (msg.pose.pose.position.x + dx,
                           msg.pose.pose.position.y + dy)
        # The shift is translation only, so the heading carries straight over.
        q = msg.pose.pose.orientation
        self.courier_yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                                      1.0 - 2.0 * (q.y ** 2 + q.z ** 2))

    def _on_courier_state(self, msg: String):
        self.courier_state = msg.data

    def courier_used(self) -> bool:
        """It carries one package and only flies once, see courier_node.py."""
        return self.courier_state in ("HOME", "RETURN_FAILED")

    def courier_route(self) -> List[Tuple[float, float]]:
        """What is left of the courier's route, from where it is now.

        Empty when no route is being flown. The plan is up to a second old
        and starts where the courier was then, so it is cut at the point
        nearest the courier and joined to its live position.
        """
        if (len(self.courier_path) < 2 or time.monotonic()
                - self.courier_path_time > self.courier_plan_timeout_s):
            return []
        if self.courier_xy is None:
            return list(self.courier_path)
        cx, cy = self.courier_xy
        nearest = min(range(len(self.courier_path)),
                      key=lambda i: (self.courier_path[i][0] - cx) ** 2
                      + (self.courier_path[i][1] - cy) ** 2)
        return [self.courier_xy] + self.courier_path[nearest + 1:]

    def _poll_pose(self):
        """Ask TF where the drone is, in the map's own frame.

        Using TF rather than /odometry matters: the search node plans in the
        odom frame while the map lives in the map frame, and SLAM corrections
        pull those apart. Looking up into whatever frame the map header
        declares keeps the drone marker on the map over a long mission.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            x = tf.transform.translation.x
            y = tf.transform.translation.y
        except Exception:
            # No map frame until SLAM starts, which is after takeoff. Until
            # then use odom, which is what the view is already drawn in: the
            # aerial photo was placed by subtracting the spawn point, so odom
            # coordinates land in the right place as they are. Subtracting the
            # spawn again would move the drone marker 2 m from where it
            # actually is - odom's origin IS the spawn point, so a position
            # looked up in odom is already relative to it.
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.odom_frame, self.base_frame, rclpy.time.Time())
            except Exception:
                return  # normal at startup, before transforms flow
            x = tf.transform.translation.x
            y = tf.transform.translation.y
        self.drone_xy = (x, y)
        self._refresh_overlays()

        q = tf.transform.rotation
        self.drone_yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y ** 2 + q.z ** 2))

        if not self.trail or (abs(self.trail[-1][0] - x) > 0.05 or
                              abs(self.trail[-1][1] - y) > 0.05):
            self.trail.append((x, y))

    def _refresh_overlays(self) -> None:
        """Put the sweep and the vantage point into this map's frame.

        Done every time the pose is polled rather than once on arrival, so a
        transform that was not ready when the message landed is picked up as
        soon as it is, and so the route follows SLAM's correction instead of
        being frozen at whatever it was.

        No timeout on the lookup. tf2 busy-waits for one, and this runs in a
        ROS callback pumped from the Qt event loop, so asking for 50 ms
        stalled the window by up to that much on every poll - in exactly the
        case the retry exists for, the first seconds when the map frame does
        not exist yet. Without it a miss costs one 100 ms frame instead, and
        the next poll tries again.
        """
        for raw, frame, attr in ((self.search_path_raw, self.search_path_frame,
                                  'search_path'),
                                 (self.vantage_raw, self.vantage_frame,
                                  'vantage_xy')):
            if not raw:
                # An EMPTY sweep is a message, not a missing one. The scout
                # publishes one whenever a plan has no lawnmower in it - the
                # trip home from _fly_home, or a vantage-only plan - and
                # skipping it left the PREVIOUS sweep drawn in dashed green
                # for the rest of the mission, a route nothing was flying.
                if attr == 'search_path':
                    self.search_path = []
                else:
                    self.vantage_xy = None
                continue
            dx = dy = 0.0
            if frame and frame != self.map_frame:
                try:
                    tf = self.tf_buffer.lookup_transform(
                        self.map_frame, frame, rclpy.time.Time())
                    dx = tf.transform.translation.x
                    dy = tf.transform.translation.y
                except Exception:
                    continue           # try again on the next poll
            if attr == 'search_path':
                self.search_path = [(x + dx, y + dy) for x, y in raw]
            else:
                self.vantage_xy = (raw[0] + dx, raw[1] + dy)

    def set_hold(self, hold: bool):
        if hold == self.holding:
            return
        self.holding = hold
        self.hold_pub.publish(Bool(data=hold))
        self.get_logger().info("Search HOLD" if hold else "Search resumed")
        if self.flashlight_mode == "THERMAL":
            self._send_light(hold)

    # -- operator settings (SETUP page) ------------------------------------
    def light_on(self) -> bool:
        """Whether the flashlight should be lit right now."""
        return (self.flashlight_mode == "ON"
                or (self.flashlight_mode == "THERMAL" and self.holding))

    def _resend_light(self):
        if self._light_active:
            self._send_light(self.light_on())

    def set_flashlight_mode(self, mode: str):
        if mode not in FLASHLIGHT_MODES or mode == self.flashlight_mode:
            return
        self.flashlight_mode = mode
        self._light_active = True
        self._send_light(self.light_on())
        self.get_logger().info(f"Setting: flashlight {mode}")

    def set_person_min_k(self, kelvin: float):
        """Body-heat threshold, kept inside the thermal display range so the
        IR page can still show what the threshold picks out."""
        kelvin = round(min(max(kelvin, self.thermal_min_k), self.thermal_max_k), 1)
        if kelvin == self.person_min_k:
            return
        self.person_min_k = kelvin
        self.get_logger().info(
            f"Setting: minimum body heat {kelvin:.1f} K ({kelvin - 273.15:.1f} C)")

    def set_thermal_detection(self, on: bool) -> bool:
        """False, and no change, if this would leave no detector on."""
        if not on and not self.rgb_enabled:
            self.get_logger().warn(
                "Thermal detection stays on: YOLO is off, and with both off "
                "nothing could raise a contact.")
            return False
        if on != self.thermal_enabled:
            self.thermal_enabled = on
            # Heat seen before the switch must not hold or back anything.
            self.thermal_hit = self.thermal_mask = self.thermal_hit_xy = None
            self.get_logger().info(
                f"Setting: thermal detection {'ON' if on else 'OFF'}")
        return True

    def set_rgb_detection(self, on: bool) -> bool:
        """False, and no change, if this would leave no detector on."""
        if not on and not self.thermal_enabled:
            self.get_logger().warn(
                "YOLO detection stays on: thermal is off, and with both off "
                "nothing could raise a contact.")
            return False
        if on != self.rgb_enabled:
            self.rgb_enabled = on
            self.rgb_hit = self.rgb_hit_xy = self.rgb_hit_track = None
            self.animal_hit = None
            self.animal_boxes = []
            self.get_logger().info(
                f"Setting: YOLO detection {'ON' if on else 'OFF'}")
        return True

    def _send_light(self, on: bool):
        """Switch the scout's floodlight. Gazebo replaces the whole light
        with this message, so every field must match the <light> in
        parrot.gazebo.xacro; only the brightness changes."""
        msg = Light()
        msg.name = "flashlight"
        msg.type = Light.SPOT
        msg.pose.position.z = -0.15
        msg.direction.z = -1.0
        level = 1.0 if on else 0.0
        msg.diffuse.r = msg.diffuse.g = msg.diffuse.b = level
        msg.diffuse.a = 1.0
        msg.specular.r = msg.specular.g = msg.specular.b = 0.3 * level
        msg.specular.a = 1.0
        msg.range = 40.0
        msg.attenuation_constant = 0.2
        msg.attenuation_linear = 0.01
        msg.attenuation_quadratic = 0.0
        msg.spot_inner_angle = 1.3
        msg.spot_outer_angle = 1.6
        msg.spot_falloff = 1.0
        msg.intensity = 3.0 * level
        self.light_pub.publish(msg)

    def set_estop(self, stop: bool):
        if stop == self.estopped:
            return
        self.estopped = stop
        self.estop_pub.publish(Bool(data=stop))
        if stop:
            self.get_logger().warn("E-STOP: all drones stopping")
            self.stop_the_drones()
        else:
            self.get_logger().info("E-Stop released: drones resuming")
            self.let_the_drones_fly()

    def stop_the_drones(self):
        """Stop both drones without needing their nodes to cooperate.

        Belt and braces on purpose. Cancelling the Nav2 goal is the clean
        stop and leaves Nav2 able to fly again on RESUME; zeroing cmd_vel is
        what actually stops a drone whose node has died mid-climb, when
        there is no goal to cancel. Neither depends on search_node or
        courier_node being alive to hear the topic.
        """
        for name, client in self.estop_cancel.items():
            if not self.get_parameter("estop_cancel_nav").value:
                break
            if not client.service_is_ready():
                # Not a failure of the E-Stop: either Nav2 is not up, so
                # nothing is flying under it, or it is gone, which is exactly
                # the case cmd_vel covers.
                self.get_logger().warn(
                    f"E-STOP: no cancel service for {name} - Nav2 is not "
                    f"running. Holding its velocity at zero instead.")
                continue
            client.call_async(CancelGoal.Request())
            self.get_logger().warn(f"E-STOP: cancelled Nav2 goals for {name}")

        if self.get_parameter("estop_zero_velocity").value:
            self.publish_zero_velocity()
            if self.estop_zero_timer is None:
                self.estop_zero_timer = self.create_timer(
                    float(self.get_parameter("estop_zero_period_s").value),
                    self.publish_zero_velocity)

    def let_the_drones_fly(self):
        """RESUME: stop holding the velocity down, or nothing can move."""
        if self.estop_zero_timer is not None:
            self.estop_zero_timer.cancel()
            self.destroy_timer(self.estop_zero_timer)
            self.estop_zero_timer = None

    def publish_zero_velocity(self):
        """Hover. Gravity is off, so zero velocity is a stop, not a drop."""
        for pub in self.estop_cmd.values():
            pub.publish(Twist())

    def launch(self):
        if self.launched:
            return
        self.launched = True
        self.launch_pub.publish(Bool(data=True))
        self.get_logger().info("LAUNCH: drones lifting off the ute")

    def send_decision(self, contact_id: int, decision: str):
        self.decision_pub.publish(String(data=f"{decision}:{contact_id}"))
        self.get_logger().info(f"Operator {decision} contact {contact_id}")

    def publish_confirmed_point(self, x: float, y: float):
        """Send the courier drone to a confirmed contact's position.

        Stamped in this robot's own map frame (self.map_frame, e.g.
        "parrot1_map") since that is where the contact's (x, y) were
        recorded. The courier reads this robot's TF to convert the point
        into its own frame (see courier_node.py).
        """
        if self.courier_used():
            self.get_logger().warn(
                f"Contact confirmed at ({x:.1f}, {y:.1f}), but the courier has "
                f"already used its package and will not fly.")
        msg = PointStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = x
        msg.point.y = y
        self.confirmed_point_pub.publish(msg)

    def publish_search_area(self, min_x: float, min_y: float,
                            max_x: float, max_y: float):
        """Send the operator-drawn box as a four-corner polygon.

        Stamped in this robot's map frame, same as the confirmed point, since
        that is the frame the operator drew it in (or in odom before there is
        a map, see below). search_node plans in the
        odom frame, so it has to convert these corners through TF rather than
        use them raw.
        """
        msg = PolygonStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        if self.map_image is None:
            # Drawn on the aerial photo before SLAM has a map - usually while
            # the drones are still on the ute, and SLAM gets no lidar returns
            # at survey height, so the map frame may not exist for a while.
            # Send it in odom, the only frame that definitely exists yet.
            #
            # The corners go as they are. _load_aerial already subtracted the
            # spawn point when it placed the photo, so the view - and anything
            # dragged on it - is ALREADY in this frame. Shifting by the spawn
            # again would convert them back to Gazebo world coordinates while
            # still labelling them odom, putting the search area 2 m from
            # where the operator drew it. odom's origin is the drone's start
            # point, not the world origin; those are only the same thing when
            # a robot happens to spawn at (0, 0).
            msg.header.frame_id = self.odom_frame
        else:
            msg.header.frame_id = self.map_frame
        for x, y in corners:
            msg.polygon.points.append(Point32(x=float(x), y=float(y), z=0.0))
        self.area_pub.publish(msg)
        self.get_logger().info(
            f"Search area set: {min_x:.1f},{min_y:.1f} to {max_x:.1f},{max_y:.1f}"
            f" ({abs(max_x - min_x):.0f} x {abs(max_y - min_y):.0f} m)")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def numpy_to_pixmap(bgr: np.ndarray) -> QPixmap:
    """cv2 BGR array -> QPixmap. The .copy() is not optional: QImage does not
    own the buffer, and without it the array is freed underneath you and the
    panel goes blank or garbled."""
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w,
                                    QImage.Format_RGB888).copy())


class BezelButton(QPushButton):
    """One option select button.

    Generic hardware, page-specific label  the whole point of the layout.
    A boxed border means the option is currently selected, which is the
    convention on real multifunction displays and reads faster than colour.
    Sized for a gloved finger on a touchscreen: nothing under 48 px.
    """

    def __init__(self, slot: str):
        super().__init__("")
        self.slot = slot
        self._boxed = False
        self._tone = MUTED
        self.setFixedSize(86, 58)
        self.setCursor(Qt.PointingHandCursor)
        # No keyboard focus: a focused button takes Space as a click, which
        # would turn the Space E-Stop into a press of RESUME (or whatever
        # was clicked last). Every button already has its own key.
        self.setFocusPolicy(Qt.NoFocus)
        font = QFont("DejaVu Sans Mono", 9)
        font.setBold(True)
        self.setFont(font)
        self._restyle()

    def configure(self, label: str, tone: str = TEXT, boxed: bool = False,
                  enabled: bool = True):
        changed = (label != self.text() or tone != self._tone
                   or boxed != self._boxed)
        self.setText(label)
        self.setEnabled(enabled and bool(label))
        if changed:
            self._tone, self._boxed = tone, boxed
            self._restyle()

    def _restyle(self):
        border = self._tone if self._boxed else LINE
        width = 2 if self._boxed else 1
        self.setStyleSheet(f"""
            QPushButton {{
                background:{BEZEL}; color:{self._tone};
                border:{width}px solid {border}; border-radius:2px;
            }}
            QPushButton:hover {{ background:{PANEL}; }}
            QPushButton:pressed {{ background:{self._tone}; color:{BEZEL}; }}
            QPushButton:disabled {{ color:#3A444E; border-color:#262E36; }}
        """)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
class MapPage(QWidget):
    """Occupancy grid with flight trail, drone and contacts drawn over it.

    The only page that answers "where have we looked, and where haven't we",
    which is why it is the default.
    """

    # "display" rather than "rgb" - see GroundStationNode.display_frame: the
    # AI (YOLO) overlay when that node is alive, plain RGB otherwise.
    PIP_MODES = [
        ("OFF",  []),
        ("IR",   ["thermal"]),
        ("VIS",  ["display"]),
        ("BOTH", ["thermal", "display"]),
    ]

    # Below this, a drag is almost certainly a stray click rather than an
    # intended area. Sending it would task the drone with a patch of ground
    # smaller than its own turning circle.
    MIN_AREA_SIDE_M = 1.0

    def __init__(self, node: GroundStationNode):
        super().__init__()
        self.pip_mode = 1        # IR by default
        self.node = node
        self.contacts: List[Contact] = []
        self.show_trail = True
        self.show_coverage = True
        self.show_path = True
        self.show_contacts = True
        self.zoom = 1.0
        self.show_aerial = node.aerial_image is not None
        self._aerial_pixmap: Optional[QPixmap] = None
        self.setStyleSheet(f"background:{BASE};")
        # Pixels per metre, the widget position of the view's top-left, and
        # the map-frame coordinates (min x, max y) that sit at that corner.
        self._scale, self._ox, self._oy = 1.0, 0.0, 0.0
        self._vx0, self._vy1 = 0.0, 0.0

        # Operator-drawn search area.
        self.draw_mode = False
        self.area: Optional[Tuple[float, float, float, float]] = None
        self._drag_a = None      # widget coords, live during a drag
        self._drag_b = None
        self.on_area_set: Optional[Callable] = None

    def cycle_pip(self):
        self.pip_mode = (self.pip_mode + 1) % len(self.PIP_MODES)

    # -- coordinate transforms --------------------------------------------
    # Everything is placed in map-frame metres rather than SLAM grid pixels,
    # so the aerial photo and the SLAM grid - two images with different
    # extents and resolutions - can share one view.
    def _world_to_widget(self, x: float, y: float) -> QPointF:
        return QPointF(self._ox + (x - self._vx0) * self._scale,
                       self._oy + (self._vy1 - y) * self._scale)

    def _widget_to_world(self, pt) -> Optional[Tuple[float, float]]:
        """Inverse of the above. Returns None before there is anything to
        show (no map and no aerial photo), since the view is not laid out
        until then."""
        if not self._has_view() or self._scale == 0:
            return None
        return (self._vx0 + (pt.x() - self._ox) / self._scale,
                self._vy1 - (pt.y() - self._oy) / self._scale)

    def _has_view(self) -> bool:
        """Something to draw: the SLAM map, or the aerial photo on its own
        while the drones are still on the ute and SLAM has not started."""
        return (self.node.map_image is not None or
                (self.show_aerial and self.node.aerial_image is not None))

    def _slam_bounds(self) -> Tuple[float, float, float, float]:
        h, w, _ = self.node.map_image.shape
        res = self.node.map_resolution
        ox, oy = self.node.map_origin
        return ox, oy, ox + w * res, oy + h * res

    def _view_bounds(self) -> Tuple[float, float, float, float]:
        """The area to fit on screen: the SLAM map, grown to take in the whole
        aerial photo when it is shown, so unexplored ground can be tasked."""
        if self.node.map_image is None:
            return self.node.aerial_bounds
        x0, y0, x1, y1 = self._slam_bounds()
        if self.show_aerial and self.node.aerial_bounds:
            ax0, ay0, ax1, ay1 = self.node.aerial_bounds
            x0, y0, x1, y1 = min(x0, ax0), min(y0, ay0), max(x1, ax1), max(y1, ay1)
        return x0, y0, x1, y1

    def _rect_for(self, x0: float, y0: float, x1: float, y1: float) -> QRectF:
        """Widget rectangle covering a map-frame rectangle."""
        return QRectF(self._world_to_widget(x0, y1), self._world_to_widget(x1, y0))

    # -- area drawing ------------------------------------------------------
    def begin_draw(self):
        self.draw_mode = True
        self.setCursor(Qt.CrossCursor)

    def clear_area(self):
        self.area = None
        self.draw_mode = False
        self._drag_a = self._drag_b = None
        self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, ev):
        if self.draw_mode and ev.button() == Qt.LeftButton:
            self._drag_a = ev.pos()
            self._drag_b = ev.pos()

    def mouseMoveEvent(self, ev):
        if self._drag_a is not None:
            self._drag_b = ev.pos()

    def mouseReleaseEvent(self, ev):
        if self._drag_a is None:
            return
        a = self._widget_to_world(self._drag_a)
        b = self._widget_to_world(ev.pos())
        self._drag_a = self._drag_b = None

        if a is None or b is None:
            return
        if (abs(a[0] - b[0]) < self.MIN_AREA_SIDE_M or
                abs(a[1] - b[1]) < self.MIN_AREA_SIDE_M):
            return   # stray click, stay armed so the operator can try again

        self.area = (min(a[0], b[0]), min(a[1], b[1]),
                     max(a[0], b[0]), max(a[1], b[1]))
        self.draw_mode = False
        self.setCursor(Qt.ArrowCursor)
        if self.on_area_set:
            self.on_area_set(*self.area)

    # -- painting ----------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BASE))

        if not self._has_view():
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "No map yet. SLAM starts once the drones are launched."
                       if not self.node.launched else
                       "No map. Start the simulation with nav2:=true")
            return

        # Fit to the widget, then apply zoom, keeping the drone centred when
        # zoomed in  panning a map by hand during a search is a distraction.
        # Drone-follow is suspended while drawing: a map that moves under the
        # cursor makes it impossible to place a corner.
        vx0, vy0, vx1, vy1 = self._view_bounds()
        span_x, span_y = max(vx1 - vx0, 1e-6), max(vy1 - vy0, 1e-6)
        self._vx0, self._vy1 = vx0, vy1
        fit = min(self.width() / span_x, self.height() / span_y)
        self._scale = fit * self.zoom
        if self.zoom > 1.0 and self.node.drone_xy and not self.draw_mode:
            dx, dy = self.node.drone_xy
            self._ox = self.width() / 2 - (dx - vx0) * self._scale
            self._oy = self.height() / 2 - (vy1 - dy) * self._scale
        else:
            self._ox = (self.width() - span_x * self._scale) / 2
            self._oy = (self.height() - span_y * self._scale) / 2

        have_map = self.node.map_image is not None
        if self.show_aerial and self.node.aerial_image is not None:
            if self._aerial_pixmap is None:
                a = self.node.aerial_image
                ah, aw, _ = a.shape
                self._aerial_pixmap = QPixmap.fromImage(
                    QImage(a.data, aw, ah, 3 * aw, QImage.Format_RGB888).copy())
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawPixmap(self._rect_for(*self.node.aerial_bounds), self._aerial_pixmap,
                         QRectF(self._aerial_pixmap.rect()))
            p.setRenderHint(QPainter.SmoothPixmapTransform, False)
            if have_map:
                img = self.node.map_overlay
                h, w, _ = img.shape
                qimg = QImage(img.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        elif have_map:
            img = self.node.map_image
            h, w, _ = img.shape
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()

        if have_map:
            p.drawPixmap(self._rect_for(*self._slam_bounds()),
                         QPixmap.fromImage(qimg), QRectF(0, 0, w, h))

        # Searched ground, under everything else. Drawn straight from the
        # pre-rendered RGBA image; the transparent cells let the map and the
        # aerial photo show through where nothing has been swept yet.
        if (self.show_coverage and self.node.coverage_image is not None
                and self.node.coverage_bounds is not None):
            cov = self.node.coverage_image
            ch, cw, _ = cov.shape
            cimg = QImage(cov.data, cw, ch, 4 * cw, QImage.Format_RGBA8888).copy()
            p.drawPixmap(self._rect_for(*self.node.coverage_bounds),
                         QPixmap.fromImage(cimg), QRectF(0, 0, cw, ch))

        # The planned sweep, under the trail so the line the drone actually
        # flew reads on top of the one it meant to. Where the route jumps
        # around an obstacle the gap is visible, which is the point.
        # A go-and-look trip has no sweep pattern to draw - often a single
        # waypoint, outside the search area - so the green path legitimately
        # disappears. Draw the leg from the drone to it in amber instead, or
        # it reads as the drone flying off for no reason.
        if (self.show_path and self.node.search_state == "LOOKING"
                and self.node.vantage_xy is not None
                and self.node.drone_xy is not None):
            a = self._world_to_widget(*self.node.drone_xy)
            b = self._world_to_widget(*self.node.vantage_xy)
            p.setPen(QPen(QColor(ALERT), 2, Qt.DotLine))
            p.drawLine(a, b)
            p.setBrush(QColor(ALERT))
            p.drawEllipse(b, 4, 4)

        if self.show_path and len(self.node.search_path) > 1:
            pts = [self._world_to_widget(x, y) for x, y in self.node.search_path]
            p.setPen(QPen(QColor(PLANNED), 2, Qt.DashLine))
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
            p.setPen(QPen(QColor(PLANNED), 1))
            p.setBrush(QColor(PLANNED))
            for q in pts:
                p.drawEllipse(q, 3, 3)

        # Under the trail and contacts, so it reads as ground marking rather
        # than another overlay competing with the things that need a decision.
        self._draw_area(p)

        if self.show_trail and len(self.node.trail) > 1:
            p.setPen(QPen(QColor(DRONE), 2))
            pts = [self._world_to_widget(x, y) for x, y in self.node.trail]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)

        # The courier's route to the contact, the part still ahead of it.
        # Dash-dot so it is not read as the scout's dashed sweep, and over
        # the trail so it is not lost under it. The contact marker drawn next
        # sits at its far end.
        route = self.node.courier_route() if self.show_path else []
        if route:
            pts = [self._world_to_widget(x, y) for x, y in route]
            p.save()
            p.setPen(QPen(QColor(COURIER), 2, Qt.DashDotLine))
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(COURIER))
            p.drawEllipse(pts[-1], 4, 4)
            p.restore()

        if self.show_contacts:
            for c in self.contacts:
                if c.status == "DISMISSED":
                    continue
                pos = self._world_to_widget(c.x, c.y)
                colour = QColor(CONFIRM if c.status == "CONFIRMED" else ALERT)
                p.setPen(QPen(colour, 2))
                p.setBrush(QColor(colour.red(), colour.green(), colour.blue(), 60))
                p.drawEllipse(pos, 11, 11)
                p.setPen(QPen(colour, 1))
                p.drawText(QPointF(pos.x() + 15, pos.y() + 4), f"{c.id:02d}")

        # Both drones as the same arrow, told apart by colour: the courier
        # first, so the scout's arrow is on top where they overlap on the ute.
        if self.node.courier_xy is not None:
            self._draw_drone(p, self.node.courier_xy, self.node.courier_yaw,
                             COURIER, BASE)
        if self.node.drone_xy:
            self._draw_drone(p, self.node.drone_xy, self.node.drone_yaw,
                             DRONE, TEXT)

        # How much of the searchable ground has been swept. The one number
        # that answers "is this area done?", so it stays on screen rather than
        # living only in the search node's terminal.
        # What the scout is doing, in its own words. The single most useful
        # line on the screen: it is what tells "finished" apart from "gone to
        # get a view" apart from "something has broken".
        STATE_COLOUR = {
            "SEARCHING": CONFIRM, "LOOKING": ALERT, "INVESTIGATING": ALERT,
            "RETURNING": TEXT,
            "COMPLETE": CONFIRM, "HOLD": ALERT, "ESTOP": ALERT,
            "UNSEARCHABLE": ALERT, "PLANNING": MUTED, "IDLE": MUTED,
        }
        if self.node.search_state and not self.draw_mode:
            p.setFont(QFont("DejaVu Sans Mono", 11, QFont.Bold))
            p.setPen(QColor(STATE_COLOUR.get(self.node.search_state, TEXT)))
            p.drawText(18, 28, self.node.search_state)
            if self.node.search_detail:
                p.setFont(QFont("DejaVu Sans Mono", 9))
                p.setPen(QColor(MUTED))
                p.drawText(18, 44, self.node.search_detail[:68])
            if self.node.courier_state:
                p.setFont(QFont("DejaVu Sans Mono", 9, QFont.Bold))
                p.setPen(QColor(COURIER))
                note = self.node.courier_state.replace("_", " ")
                if self.node.courier_used():
                    note += " - PACKAGE USED"
                p.drawText(18, 60, "COURIER " + note)

        if self.show_coverage and self.node.coverage_percent is not None:
            p.setFont(QFont("DejaVu Sans Mono", 10, QFont.Bold))
            done = self.node.coverage_percent >= 99.5
            y = self.height() - 18
            gap = self.node.coverage_gap_m2
            # Bottom line last: the gap is the thing a reader must not be
            # able to miss, so it sits under the percentages in amber rather
            # than being implied by the difference between them.
            if gap >= 0.5:
                p.setPen(QColor(ALERT))
                p.drawText(18, y, f"NOT CHECKED {gap:6.0f} m\u00b2  "
                                  f"{self.node.coverage_gap_reason}")
                y -= 16
            if self.node.coverage_assigned is not None:
                p.setPen(QColor(TEXT))
                p.drawText(18, y,
                           f"OF AREA ASSIGNED {self.node.coverage_assigned:5.1f}%")
                y -= 16
            p.setPen(QColor(CONFIRM if done else TEXT))
            p.drawText(18, y, f"SEARCHED  {self.node.coverage_percent:5.1f}%"
                              f"{'  of reachable' if gap >= 0.5 else ''}")

        if self.draw_mode and self._drag_a is None:
            p.setPen(QColor(ALERT))
            p.setFont(QFont("DejaVu Sans Mono", 10, QFont.Bold))
            p.drawText(18, 28, "DRAG TO SET SEARCH AREA   (ESC TO CANCEL)")

        if self.PIP_MODES[self.pip_mode][1]:
            self._draw_pips(p)

    def _draw_drone(self, p: QPainter, xy: Tuple[float, float], yaw: float,
                    fill: str, outline: str):
        """An arrow at xy pointing along yaw. save/restore so its brush does
        not leak into whatever is drawn after it."""
        p.save()
        p.translate(self._world_to_widget(*xy))
        p.rotate(-np.degrees(yaw))
        p.setBrush(QColor(fill))
        p.setPen(QPen(QColor(outline), 1))
        p.drawPolygon(QPointF(12, 0), QPointF(-7, 7), QPointF(-7, -7))
        p.restore()

    def _draw_area(self, p: QPainter):
        """Dashed amber while dragging, solid green once committed and sent.
        The live dimensions matter: an operator sizing a search box needs to
        know whether they have drawn 20 m or 200 m of ground."""
        if self._drag_a is not None and self._drag_b is not None:
            rect = QRectF(self._drag_a, self._drag_b).normalized()
            p.setPen(QPen(QColor(ALERT), 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect)

            a = self._widget_to_world(self._drag_a)
            b = self._widget_to_world(self._drag_b)
            if a and b:
                p.setPen(QColor(ALERT))
                p.setFont(QFont("DejaVu Sans Mono", 9))
                p.drawText(rect.topLeft() + QPointF(6, -6),
                           f"{abs(a[0] - b[0]):.0f} x {abs(a[1] - b[1]):.0f} m")
            return

        if self.area:
            x0, y0, x1, y1 = self.area
            rect = QRectF(self._world_to_widget(x0, y1),
                          self._world_to_widget(x1, y0))
            p.setPen(QPen(QColor(CONFIRM), 2))
            p.setBrush(QColor(76, 175, 109, 25))
            p.drawRect(rect)
            p.setPen(QColor(CONFIRM))
            p.setFont(QFont("DejaVu Sans Mono", 9))
            p.drawText(rect.topLeft() + QPointF(6, -6),
                       f"SEARCH AREA  {abs(x1 - x0):.0f} x {abs(y1 - y0):.0f} m")

    def _draw_pips(self, p: QPainter):
        """Insets stack upward from the bottom-right corner. Two at once eats
        real map area, which is why OFF and single-feed modes exist  the
        operator decides how much map they are willing to trade."""
        pw, ph = 200, 150
        x = self.width() - pw - 14
        y = self.height() - ph - 14

        for source in self.PIP_MODES[self.pip_mode][1]:
            frame = getattr(self.node, f"{source}_frame")
            if frame is None:
                continue
            pix = numpy_to_pixmap(frame).scaled(
                pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap(x, y, pix)
            p.setPen(QPen(QColor(LINE), 1))
            # A border only: drawRect fills with the current brush, and a
            # brush left set by anything drawn earlier would paint over the
            # picture.
            p.setBrush(Qt.NoBrush)
            p.drawRect(x, y, pix.width(), pix.height())
            p.setPen(QColor(MUTED))
            p.setFont(QFont("DejaVu Sans Mono", 8))
            p.drawText(x + 5, y + 14, "IR" if source == "thermal" else "VIS")
            y -= pix.height() + 8


class VideoPage(QWidget):
    """Full-bleed camera feed. Shows a held frame instead while a contact is
    awaiting a ruling, so the evidence for it stays up (see MainWindow)."""

    def __init__(self, node: GroundStationNode, source: str, empty_text: str):
        super().__init__()
        self.node = node
        self.source = source            # "display" (RGB or AI overlay), "thermal"
        self.empty_text = empty_text
        self.frozen: Optional[np.ndarray] = None
        self.setStyleSheet(f"background:{BASE};")

    def current(self) -> Optional[np.ndarray]:
        if self.frozen is not None:
            return self.frozen
        return getattr(self.node, f"{self.source}_frame")

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(BASE))
        frame = self.current()
        if frame is None:
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, self.empty_text)
            return

        pix = numpy_to_pixmap(frame).scaled(self.size(), Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation)
        p.drawPixmap((self.width() - pix.width()) // 2,
                     (self.height() - pix.height()) // 2, pix)

        if self.frozen is not None:
            p.setPen(QColor(ALERT))
            p.setFont(QFont("DejaVu Sans Mono", 11, QFont.Bold))
            p.drawText(18, 30, "FROZEN")


class ContactsPage(QWidget):
    """Every contact and its ruling. Its own page rather than a side rail, so
    the display area stays undivided."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background:{BASE};")
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)

        self.list = QListWidget()
        self.list.setStyleSheet(f"""
            QListWidget {{ background:{BASE}; color:{TEXT}; border:none;
                           font-family:'DejaVu Sans Mono'; font-size:11pt; }}
            QListWidget::item {{ padding:12px 16px;
                                 border-bottom:1px solid {LINE}; }}
            QListWidget::item:selected {{ background:{PANEL}; }}
        """)
        box.addWidget(self.list)

        self.empty = QLabel("No contacts. The drone is still searching.")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setStyleSheet(f"color:{MUTED};")
        box.addWidget(self.empty)

    def refresh(self, contacts: List[Contact]):
        self.empty.setVisible(not contacts)
        self.list.setVisible(bool(contacts))

        keep = self.list.currentRow()
        self.list.clear()
        for c in contacts:
            item = QListWidgetItem(
                f"{c.id:02d}   {c.x:+7.1f} {c.y:+7.1f} m     "
                f"VIS {c.rgb_confidence:>4.0%}   IR {c.thermal_confidence:>4.0%}"
                f"     {c.status}")
            item.setForeground(QColor({"UNCONFIRMED": ALERT,
                                       "CONFIRMED": CONFIRM,
                                       "DISMISSED": MUTED}[c.status]))
            self.list.addItem(item)
        if 0 <= keep < self.list.count():
            self.list.setCurrentRow(keep)
        elif self.list.count():
            self.list.setCurrentRow(0)

    def selected_index(self) -> int:
        return self.list.currentRow()


class SettingsPage(QWidget):
    """Detection and flashlight settings. A readout only: the values are
    changed with the right column, like every other page's options, so the
    page works the same with a gloved finger or the F keys."""

    LIGHT_TEXT = {
        "OFF": "never on",
        "ON": "on all the time",
        "THERMAL": "on while the search holds to check body heat",
    }

    def __init__(self, node: GroundStationNode):
        super().__init__()
        self.node = node
        self.setStyleSheet(f"background:{BASE};")
        box = QVBoxLayout(self)
        box.setContentsMargins(40, 30, 40, 30)
        self.text = QLabel("")
        self.text.setTextFormat(Qt.RichText)
        self.text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.text.setWordWrap(True)
        self.text.setStyleSheet(f"color:{TEXT}; border:none;"
                                f"font-family:'DejaVu Sans Mono'; font-size:11pt;")
        box.addWidget(self.text)
        box.addStretch(1)
        self.refresh()

    def describe(self) -> str:
        n = self.node
        on = f'<span style="color:{CONFIRM}"><b>ON</b></span>'
        off = f'<span style="color:{RED}"><b>OFF</b></span>'

        def row(key, name, value, note):
            return (f'<tr><td style="color:{MUTED}; padding:10px 18px 10px 0">'
                    f'{key}</td><td style="padding:10px 18px 10px 0"><b>{name}'
                    f'</b></td><td style="padding:10px 18px 10px 0">{value}</td>'
                    f'<td style="color:{MUTED}; padding:10px 0">{note}</td></tr>')

        lit = "lit now" if n.light_on() else "dark now"
        if n.thermal_enabled and n.rgb_enabled:
            rule = ("A contact needs body heat <b>and</b> the AI seeing a "
                    "person on it.")
        elif n.rgb_enabled:
            rule = ("Thermal off: the AI seeing a person raises a contact "
                    "on its own.")
        else:
            rule = ("AI off: body heat raises a contact on its own. Animals "
                    "are warm too - dismiss them.")
        notes = []
        if n.flashlight_mode == "THERMAL" and not n.thermal_enabled:
            notes.append("With thermal off, THERMAL only lights the "
                         "flashlight while a contact holds the search.")
        if n.person_min_k > 310.0:
            notes.append("The threshold is above body heat (310 K): people "
                         "may not register at all.")
        return (
            f'<p style="color:{MUTED}">SETUP — press the buttons on the right '
            f'(F1–F5). Changes apply at once.</p>'
            '<table cellspacing="0">'
            + row("F1", "FLASHLIGHT", n.flashlight_mode,
                  f"{self.LIGHT_TEXT[n.flashlight_mode]} ({lit})")
            + row("F2/F3", "MIN BODY HEAT",
                  f"{n.person_min_k:.1f} K ({n.person_min_k - 273.15:.1f} °C)",
                  "thermal pixels this warm count as body heat")
            + row("F4", "THERMAL DETECT", on if n.thermal_enabled else off,
                  "body heat holds the search for a check")
            + row("F5", "AI (YOLO) DETECT", on if n.rgb_enabled else off,
                  "the RGB camera's person detector")
            + '</table>'
            f'<p>{rule}</p>'
            # Not ALERT: that colour is kept for an unruled contact.
            + "".join(f'<p style="color:{RED}">{t}</p>' for t in notes)
            + f'<p style="color:{MUTED}">At least one detector stays on.</p>')

    def refresh(self):
        text = self.describe()
        if text != self.text.text():     # rich text relayout is not free
            self.text.setText(text)


# ---------------------------------------------------------------------------
# Tutorial
# ---------------------------------------------------------------------------
@dataclass
class TutorialStep:
    title: str
    body: str
    # Returns the widgets to highlight; empty means a centred box with no
    # highlight (welcome and summary steps).
    targets: Callable[[], List[QWidget]] = lambda: []
    # Page to show while this step is up, so page-specific labels (the right
    # column) say what the text describes.
    page: Optional[int] = None


def place_callout(target: Optional[QRect], box: QSize, bounds: QRect,
                  gap: int = 16) -> QPoint:
    """Top-left for a callout box of size `box` next to `target`, inside
    `bounds`. Tries right, left, below, then above the target and takes the
    first side with room; centres it if there is no target or no room."""
    if target is None:
        return QPoint(bounds.center().x() - box.width() // 2,
                      bounds.center().y() - box.height() // 2)
    cy = target.center().y() - box.height() // 2
    cx = target.center().x() - box.width() // 2
    candidates = [
        QPoint(target.right() + gap, cy),
        QPoint(target.left() - gap - box.width(), cy),
        QPoint(cx, target.bottom() + gap),
        QPoint(cx, target.top() - gap - box.height()),
    ]
    for p in candidates:
        r = QRect(p, box)
        # Sideways only needs horizontal room (the clamp below fixes y);
        # above/below only needs vertical room.
        if bounds.left() <= r.left() and r.right() <= bounds.right() and \
                bounds.top() <= r.top() + box.height() // 2 and \
                r.bottom() - box.height() // 2 <= bounds.bottom():
            break
    else:
        p = QPoint(bounds.center().x() - box.width() // 2,
                   bounds.center().y() - box.height() // 2)
    x = min(max(p.x(), bounds.left()), bounds.right() - box.width())
    y = min(max(p.y(), bounds.top()), bounds.bottom() - box.height())
    return QPoint(x, y)


class TutorialOverlay(QWidget):
    """Guided tour: dims the window, outlines one control at a time and pops
    a small box beside it saying what it does.

    Covers the whole window, so clicks can't reach the bezels underneath
    while the operator is reading. Highlight is CONFIRM cyan, never ALERT:
    the yellow-green stays reserved for an unruled contact.
    """

    def __init__(self, parent: QWidget, steps: List[TutorialStep],
                 on_page: Callable[[int], None], on_close: Callable[[], None]):
        super().__init__(parent)
        self.steps = steps
        self.index = 0
        self._on_page = on_page
        self._on_close = on_close
        self.setFocusPolicy(Qt.StrongFocus)

        self.box = QFrame(self)
        self.box.setFixedWidth(360)
        self.box.setStyleSheet(
            f"QFrame {{ background:{PANEL}; border:2px solid {CONFIRM}; "
            f"border-radius:4px; }}"
            f"QLabel {{ border:none; background:transparent; }}")
        lay = QVBoxLayout(self.box)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(8)

        self.step_label = QLabel()
        self.step_label.setFont(QFont("DejaVu Sans Mono", 8))
        self.step_label.setStyleSheet(f"color:{MUTED};")
        lay.addWidget(self.step_label)

        self.title_label = QLabel()
        f = QFont("DejaVu Sans Mono", 12)
        f.setBold(True)
        self.title_label.setFont(f)
        self.title_label.setStyleSheet(f"color:{CONFIRM};")
        self.title_label.setWordWrap(True)
        lay.addWidget(self.title_label)

        self.body_label = QLabel()
        self.body_label.setFont(QFont("DejaVu Sans", 10))
        self.body_label.setStyleSheet(f"color:{TEXT};")
        self.body_label.setWordWrap(True)
        self.body_label.setTextFormat(Qt.RichText)
        lay.addWidget(self.body_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.close_btn = self._button("CLOSE", MUTED, self.finish)
        self.back_btn = self._button("◀ BACK", TEXT, self.back)
        self.next_btn = self._button("NEXT ▶", CONFIRM, self.next)
        buttons.addWidget(self.close_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.back_btn)
        buttons.addWidget(self.next_btn)
        lay.addLayout(buttons)

        self.hint = QLabel("← → to move, Esc to close. Space still E-Stops.")
        self.hint.setFont(QFont("DejaVu Sans Mono", 8))
        self.hint.setStyleSheet(f"color:{MUTED};")
        lay.addWidget(self.hint)

        self._hole: Optional[QRect] = None

    def _button(self, text: str, tone: str, handler) -> QPushButton:
        b = QPushButton(text)
        b.setCursor(Qt.PointingHandCursor)
        b.setFocusPolicy(Qt.NoFocus)
        b.setFont(QFont("DejaVu Sans Mono", 9, QFont.Bold))
        b.setMinimumHeight(32)
        b.setStyleSheet(f"""
            QPushButton {{ background:{BEZEL}; color:{tone};
                           border:1px solid {LINE}; border-radius:2px;
                           padding:2px 12px; }}
            QPushButton:hover {{ border-color:{tone}; }}
        """)
        b.clicked.connect(handler)
        return b

    # -- navigation --------------------------------------------------------
    def start(self):
        self.index = 0
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self.setFocus()
        self._show_step()

    def next(self):
        if self.index >= len(self.steps) - 1:
            self.finish()
        else:
            self.index += 1
            self._show_step()

    def back(self):
        if self.index > 0:
            self.index -= 1
            self._show_step()

    def finish(self):
        self.hide()
        self._on_close()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Right, Qt.Key_Return, Qt.Key_Enter):
            self.next()
        elif key == Qt.Key_Left:
            self.back()
        elif key == Qt.Key_Escape:
            self.finish()
        else:
            super().keyPressEvent(event)

    # -- drawing -----------------------------------------------------------
    def _target_rect(self, widgets: List[QWidget]) -> Optional[QRect]:
        rect = None
        for w in widgets:
            if w is None or not w.isVisible():
                continue
            r = QRect(w.mapTo(self.parentWidget(), QPoint(0, 0)), w.size())
            rect = r if rect is None else rect.united(r)
        return rect.adjusted(-6, -6, 6, 6) if rect is not None else None

    def _show_step(self):
        step = self.steps[self.index]
        if step.page is not None:
            self._on_page(step.page)
        last = self.index == len(self.steps) - 1
        self.step_label.setText(f"TUTORIAL  {self.index + 1} / {len(self.steps)}")
        self.title_label.setText(step.title)
        self.body_label.setText(step.body)
        self.back_btn.setEnabled(self.index > 0)
        self.next_btn.setText("DONE ✓" if last else "NEXT ▶")
        self.reposition()

    def reposition(self):
        """Lay out for the current step. Also called on window resize."""
        if not self.isVisible():
            return
        self.setGeometry(self.parentWidget().rect())
        self._hole = self._target_rect(self.steps[self.index].targets())
        self.box.adjustSize()
        bounds = self.rect().adjusted(12, 12, -12, -12)
        self.box.move(place_callout(self._hole, self.box.size(), bounds))
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        shade = QPainterPath()
        shade.addRect(QRectF(self.rect()))
        if self._hole is not None:
            hole = QPainterPath()
            hole.addRoundedRect(QRectF(self._hole), 6, 6)
            shade = shade.subtracted(hole)
        p.fillPath(shade, QColor(0, 0, 0, 165))
        if self._hole is not None:
            p.setPen(QPen(QColor(CONFIRM), 3))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(self._hole), 6, 6)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    CONTACT_RADIUS = 5.0  # metres; closer than a confirmed contact = same person

    def __init__(self, node: GroundStationNode):
        super().__init__()
        self.node = node
        self.contacts: List[Contact] = []
        self._next_id = 1
        self._blink = False
        self._select_newest = False
        self._no_check_until = 0.0
        # Thermal check in progress (see _check_detections): when it gives
        # up, where it started, the best heat seen and the IR frame at start.
        self._check_until: Optional[float] = None
        self._check_xy = (0.0, 0.0)
        self._check_heat_conf = 0.0
        self._check_ir_frame: Optional[np.ndarray] = None
        # Heat the RGB check did not confirm as human; not checked again.
        self.rejected_spots: List[Tuple[float, float]] = []
        self.page = PAGE_MAP
        # A confirmed person keeps the drone holding over them until the
        # operator explicitly resumes the search.
        self.hold_for_confirmed = False
        self._mission_over = False
        # An area was drawn after the latest confirm: when the courier gets
        # back the scout searches that, so heat checks must stay on.
        self._area_since_confirm = False

        self.setWindowTitle("AeroAid  ground station")
        self.resize(1440, 900)
        self.setStyleSheet(f"QMainWindow {{ background:{BEZEL}; }}"
                           f"QLabel {{ color:{TEXT}; }}")

        self.map_page = MapPage(node)
        # "display": AI (YOLO) detection overlay while yolo_person_detector.py
        # is alive and publishing, falling back to plain RGB otherwise - see
        # GroundStationNode.display_frame.
        self.vis_page = VideoPage(node, "display", "No camera feed")
        self.ir_page = VideoPage(node, "thermal", "No thermal feed")
        self.contacts_page = ContactsPage()
        self.settings_page = SettingsPage(node)

        self.map_page.on_area_set = self._set_area

        self.stack = QStackedWidget()
        for page in (self.map_page, self.vis_page, self.ir_page,
                     self.contacts_page, self.settings_page):
            self.stack.addWidget(page)
        self.stack.setStyleSheet(f"border:1px solid {LINE};")

        self._build_layout()
        self._build_shortcuts()
        self.select_page(PAGE_MAP)

        self._page_before_tutorial = PAGE_MAP
        self.tutorial = TutorialOverlay(self, self._tutorial_steps(),
                                        on_page=self._tutorial_page,
                                        on_close=self._close_tutorial)
        self.tutorial.hide()

        # 20 Hz is plenty for a human. Sensor data arrives faster; there is no
        # reason for the interface to chase it.
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.refresh)
        self.ui_timer.start(50)

        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._toggle_blink)
        self.blink_timer.start(600)

    # -- construction ------------------------------------------------------
    def _build_layout(self):
        root = QWidget()
        self.setCentralWidget(root)
        grid = QGridLayout(root)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(8)

        grid.addWidget(self._status_strip(), 0, 0, 1, 3)

        self.left_buttons = [BezelButton(f"L{i+1}") for i in range(5)]
        left = QVBoxLayout()
        left.setSpacing(8)
        left.addStretch(1)
        for i, b in enumerate(self.left_buttons):
            b.clicked.connect(lambda _, n=i: self.select_page(n))
            left.addWidget(b)
        left.addStretch(1)
        grid.addLayout(left, 1, 0)

        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.stack, 1, 1)

        self.right_buttons = [BezelButton(f"R{i+1}") for i in range(RIGHT_SLOTS)]
        right = QVBoxLayout()
        right.setSpacing(8)
        right.addStretch(1)
        for i, b in enumerate(self.right_buttons):
            b.clicked.connect(lambda _, n=i: self._press_right(n))
            right.addWidget(b)
        right.addStretch(1)
        grid.addLayout(right, 1, 2)

        self.bottom_buttons = [BezelButton(f"B{i+1}") for i in range(3)]
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addStretch(1)
        for b in self.bottom_buttons:
            bottom.addWidget(b)
        bottom.addStretch(1)
        self.bottom_buttons[0].clicked.connect(lambda: self._rule("CONFIRMED"))
        self.bottom_buttons[1].clicked.connect(lambda: self._rule("DISMISSED"))
        self.bottom_buttons[2].clicked.connect(self.node.launch)
        bottom.addSpacing(40)
        # Wider than the bezels and set apart from them, so it is the easiest
        # thing on the screen to hit and hard to hit by mistake for another.
        self.estop_button = BezelButton("ESTOP")
        self.estop_button.setFixedSize(150, 58)
        self.estop_button.clicked.connect(self._press_estop)
        bottom.addWidget(self.estop_button)
        grid.addLayout(bottom, 2, 1)

        grid.setColumnStretch(1, 1)
        grid.setRowStretch(1, 1)

    def _status_strip(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("status_strip")     # the tutorial highlights it
        bar.setStyleSheet(f"background:{PANEL}; border:1px solid {LINE};")
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 8, 14, 8)

        title = QLabel("AEROAID")
        f = QFont("DejaVu Sans Mono", 12)
        f.setBold(True)
        title.setFont(f)
        title.setStyleSheet(f"color:{TEXT}; border:none;")
        row.addWidget(title)

        self.page_label = QLabel("")
        self.page_label.setStyleSheet(f"color:{DRONE}; border:none; padding-left:20px;")
        row.addWidget(self.page_label)

        row.addStretch(1)

        # Master caution. A single-display layout can hide a new contact
        # behind whichever page is up, so one indicator stays visible on every
        # page and is the only thing in the interface that blinks.
        self.caution = QLabel("")
        self.caution.setFont(QFont("DejaVu Sans Mono", 11, QFont.Bold))
        self.caution.setStyleSheet("border:none;")
        row.addWidget(self.caution)

        self.hold_label = QLabel("")
        self.hold_label.setFont(QFont("DejaVu Sans Mono", 10, QFont.Bold))
        self.hold_label.setStyleSheet(f"color:{RED}; border:none; padding-left:24px;")
        row.addWidget(self.hold_label)

        # RGB person detector (yolo_person_detector.py) running or not.
        self.ai_label = QLabel("AI ---")
        self.ai_label.setFont(QFont("DejaVu Sans Mono", 10))
        self.ai_label.setStyleSheet(f"color:{RED}; border:none; padding-left:24px;")
        row.addWidget(self.ai_label)

        # Empty unless thermal detection is switched off (SETUP page).
        self.ir_label = QLabel("")
        self.ir_label.setFont(QFont("DejaVu Sans Mono", 10))
        self.ir_label.setStyleSheet(f"color:{MUTED}; border:none; padding-left:24px;")
        row.addWidget(self.ir_label)

        self.link_label = QLabel("LINK ---")
        self.link_label.setFont(QFont("DejaVu Sans Mono", 10))
        self.link_label.setStyleSheet(f"color:{MUTED}; border:none; padding-left:24px;")
        row.addWidget(self.link_label)

        # Up here rather than in a bezel slot: it is on every page and
        # doesn't take a function slot away from the display.
        row.addSpacing(24)
        self.help_button = QPushButton("? HELP")
        self.help_button.setCursor(Qt.PointingHandCursor)
        self.help_button.setFocusPolicy(Qt.NoFocus)   # see BezelButton
        self.help_button.setFont(QFont("DejaVu Sans Mono", 9, QFont.Bold))
        self.help_button.setStyleSheet(f"""
            QPushButton {{ background:{BEZEL}; color:{CONFIRM};
                           border:1px solid {CONFIRM}; border-radius:2px;
                           padding:3px 12px; }}
            QPushButton:hover {{ background:{CONFIRM}; color:{BEZEL}; }}
        """)
        self.help_button.clicked.connect(self.open_tutorial)
        row.addWidget(self.help_button)
        return bar

    def _build_shortcuts(self):
        # Every shortcut but the E-Stop goes in this list, so the tutorial can
        # switch them off while it is open: a stray R while reading must not
        # launch the drones. Space stays live everywhere.
        self._shortcuts: List[QShortcut] = []

        def key(seq, handler):
            self._shortcuts.append(QShortcut(QKeySequence(seq), self, handler))

        for i, k in enumerate(["1", "2", "3", "4", "5"]):
            key(k, lambda n=i: self.select_page(n))
        for i in range(RIGHT_SLOTS):
            key(f"F{i+1}", lambda n=i: self._press_right(n))
        key("Q", lambda: self._rule("CONFIRMED"))
        key("W", lambda: self._rule("DISMISSED"))
        key("R", self.node.launch)
        QShortcut(QKeySequence("Space"), self, lambda: self.node.set_estop(True))
        # Fake contact for testing the interface; keyboard only.
        key("E", self._simulate_contact)
        # Escape stands down an armed area draw without setting one.
        key("Escape", self._cancel_draw)
        key("H", self.open_tutorial)

    # -- tutorial ----------------------------------------------------------
    def _tutorial_steps(self) -> List[TutorialStep]:
        L, R, B = self.left_buttons, self.right_buttons, self.bottom_buttons
        k = '<span style="color:%s"><b>%%s</b></span>' % CONFIRM   # key name
        return [
            TutorialStep(
                "Welcome to AeroAid",
                "This station runs a two-drone search and rescue mission. "
                "The <b>scout</b> flies a lawnmower search over an area you "
                "draw, looking for people with a thermal camera and an AI "
                "(YOLO) camera. You rule on each possible person it finds. "
                "When you confirm one, the <b>courier</b> flies a care "
                "package to them.<br><br>This tour shows what every button "
                "does. Use NEXT and BACK, or the arrow keys."),
            TutorialStep(
                "Status strip",
                "Always visible, whatever page is up.<br>"
                "• <b>CONTACT n</b> blinks when a possible person needs "
                "your decision. It is the only thing that blinks.<br>"
                "• <b>SEARCH HOLD / IR CHECK / E-STOP</b>: why the scout "
                "is stopped, if it is.<br>"
                "• <b>AI OK</b>: the person detector is running.<br>"
                "• <b>LINK OK</b>: data is arriving from the scout.",
                lambda: [self.findChild(QFrame, "status_strip")]),
            TutorialStep(
                "Main display",
                "Shows the page picked on the left. On the map: the white "
                "arrow is the scout and its trail, the green line is the "
                "planned search route, cyan shading is ground already "
                "searched, and markers are contacts.",
                lambda: [self.stack], PAGE_MAP),
            TutorialStep(
                f"MAP  ({k % '1'})",
                "The overview: map, scout position, search area and "
                "contacts. The left column picks the page and never changes, "
                "so you can hit it without looking.",
                lambda: [L[0]], PAGE_MAP),
            TutorialStep(
                f"VIS  ({k % '2'})",
                "The scout's camera with the AI's boxes drawn on:<br>"
                "• <span style='color:#00FF00'><b>green</b></span>: a "
                "person, fully in view. Only these can raise a contact.<br>"
                "• <span style='color:#FFFF00'><b>yellow (entering)</b></span>: "
                "a person still at the edge of the frame.<br>"
                "• <span style='color:#FFBF00'><b>amber (ignored)</b></span>: "
                "a kangaroo or wombat. Shown, but never a contact.",
                lambda: [L[1]]),
            TutorialStep(
                f"IR  ({k % '3'})",
                "The thermal camera. Bright means warm. Body heat makes the "
                "scout stop and look with the AI camera; a contact needs "
                "<b>both</b> heat and the AI seeing a person.",
                lambda: [L[2]]),
            TutorialStep(
                f"CNTCT  ({k % '4'})",
                "Every contact and its ruling. The number is how many there "
                "are; the label lights up when one is waiting for you. "
                "This page also has <b>RESUME</b>, which restarts the search "
                "after you confirm someone.",
                lambda: [L[3]]),
            TutorialStep(
                f"SETUP  ({k % '5'})",
                "Detection settings, changed with the right column:<br>"
                "• <b>LIGHT</b>: the scout's flashlight OFF, ON, or THERMAL "
                "(on while it stops to check body heat).<br>"
                "• <b>HEAT+ / HEAT-</b>: how warm counts as body heat.<br>"
                "• <b>IR DET / AI DET</b>: switch a detector off, e.g. "
                "thermal on a hot day, and the other raises contacts on its "
                "own. One always stays on.",
                lambda: [L[4]]),
            TutorialStep(
                f"Map tools  ({k % 'F1'}–{k % 'F7'})",
                "The right column changes with the page; these are the map's.<br>"
                "• <b>TRAIL</b>: show or hide the scout's flight path.<br>"
                "• <b>CNTCT</b>: show or hide contact markers.<br>"
                "• <b>PIP</b>: a small camera picture over the map.<br>"
                "• <b>ZOOM+ / ZOOM-</b>: zoom the map.<br>"
                "• <b>AERIAL</b>: the aerial photo under the map.<br>"
                "A boxed label means that option is on.<br>"
                "The white arrow is the scout, the yellow arrow the "
                "<b>courier</b>. While the courier flies to a confirmed "
                "person, its route ahead is the yellow dash-dot line.",
                lambda: [b for b in R if b.text()], PAGE_MAP),
            TutorialStep(
                f"SET AREA  ({k % 'F6'})",
                "Step one of every mission. Press it, then drag a box on the "
                "map over the ground to search. You can do this before "
                "launch. Press again (<b>CLR AREA</b>) to clear it, or "
                f"{k % 'Esc'} to cancel while drawing.",
                lambda: [R[5]], PAGE_MAP),
            TutorialStep(
                f"LAUNCH  ({k % 'R'})",
                "Lifts both drones off the ute. The scout climbs and flies "
                "the search area. It changes to <b>AIRBORNE</b> and can't be "
                "undone: the drones don't land back on the ute.",
                lambda: [B[2]]),
            TutorialStep(
                f"CONFIRM  ({k % 'Q'})",
                "When CONTACT blinks, the scout holds still and the camera "
                "pages freeze on what it saw. If it is a person, CONFIRM: "
                "the courier flies the care package to them. The scout keeps "
                "holding until you press RESUME on the CNTCT page.",
                lambda: [B[0]]),
            TutorialStep(
                f"DISMISS  ({k % 'W'})",
                "Not a person? DISMISS. The scout carries on, and that same "
                "object won't be raised again, even when the scout passes "
                "over it on a later row.",
                lambda: [B[1]]),
            TutorialStep(
                f"E-STOP  ({k % 'Space'})",
                "Stops <b>every</b> drone where it is, at any time. Space "
                "works on every page, even with this tour open. Only a "
                "click on <b>RESUME</b> (the same button) restarts them, so "
                "a stray key can never set them moving.",
                lambda: [self.estop_button]),
            TutorialStep(
                "Running a mission",
                "<b>1.</b> SET AREA and drag a box over the search ground.<br>"
                "<b>2.</b> LAUNCH.<br>"
                "<b>3.</b> Watch the map. When CONTACT blinks, check VIS "
                "and IR.<br>"
                "<b>4.</b> CONFIRM a person, or DISMISS anything else.<br>"
                "<b>5.</b> After a confirm, open CNTCT and press RESUME to "
                "search on.<br><br>"
                f"Press ? HELP or {k % 'H'} to see this tour again.",
                page=PAGE_MAP),
        ]

    def open_tutorial(self):
        if self.tutorial.isVisible():
            return
        self._page_before_tutorial = self.page
        for s in self._shortcuts:
            s.setEnabled(False)
        self.tutorial.start()

    def _tutorial_page(self, page: int):
        self.select_page(page)
        # Relabel now, not on the next refresh tick, so the step highlights
        # this page's buttons rather than the last page's.
        self._sync_bezels()

    def _close_tutorial(self):
        for s in self._shortcuts:
            s.setEnabled(True)
        self.select_page(self._page_before_tutorial)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "tutorial"):
            self.tutorial.reposition()

    # -- page plumbing -----------------------------------------------------
    def select_page(self, index: int):
        self.page = index
        self.stack.setCurrentIndex(index)

    def _right_actions(self) -> List[Tuple[str, Callable, bool]]:
        """Label, handler and boxed-state for the right column, for whichever
        page is current. Rebuilding it each refresh is what lets a toggle show
        its own state in the label box."""
        blank = ("", lambda: None, False)

        if self.page == PAGE_MAP:
            m = self.map_page
            return [
                ("TRAIL", lambda: setattr(m, "show_trail", not m.show_trail), m.show_trail),
                ("CNTCT", lambda: setattr(m, "show_contacts", not m.show_contacts), m.show_contacts),
                (f"PIP\n{MapPage.PIP_MODES[m.pip_mode][0]}", m.cycle_pip, m.pip_mode != 0),
                (f"ZOOM+\n{m.zoom:.1f}x", lambda: setattr(m, "zoom", min(m.zoom * 1.5, 8.0)), m.zoom > 1.0),
                ("ZOOM-", lambda: setattr(m, "zoom", max(m.zoom / 1.5, 1.0)), False),
                # One slot, three states: no area, armed, area set. The label
                # names the action the next press performs, not the state.
                ("CLR\nAREA" if m.area else "SET\nAREA",
                 self._toggle_area_draw, m.draw_mode or bool(m.area)),
                # Only offered when a photo was loaded (see aerial_map).
                ("AERIAL", lambda: setattr(m, "show_aerial", not m.show_aerial),
                 m.show_aerial) if self.node.aerial_image is not None else blank,
            ]
        if self.page == PAGE_CONTACTS:
            return [("RESUME", self._resume_search, False)] + [blank] * (RIGHT_SLOTS - 1)
        if self.page == PAGE_SETTINGS:
            n = self.node
            return [
                (f"LIGHT\n{n.flashlight_mode}", self._cycle_flashlight,
                 n.flashlight_mode != "OFF"),
                (f"HEAT+\n{n.person_min_k:.1f}K",
                 lambda: n.set_person_min_k(n.person_min_k + HEAT_STEP_K), False),
                ("HEAT-", lambda: n.set_person_min_k(n.person_min_k - HEAT_STEP_K),
                 False),
                ("IR DET", lambda: n.set_thermal_detection(not n.thermal_enabled),
                 n.thermal_enabled),
                ("AI DET", lambda: n.set_rgb_detection(not n.rgb_enabled),
                 n.rgb_enabled),
            ] + [blank] * (RIGHT_SLOTS - 5)
        return [blank] * RIGHT_SLOTS

    def _cycle_flashlight(self):
        modes = FLASHLIGHT_MODES
        i = modes.index(self.node.flashlight_mode)
        self.node.set_flashlight_mode(modes[(i + 1) % len(modes)])

    def _press_right(self, slot: int):
        actions = self._right_actions()
        if slot < len(actions) and actions[slot][0]:
            actions[slot][1]()

    def _toggle_area_draw(self):
        m = self.map_page
        if m.area or m.draw_mode:
            m.clear_area()      # clear a set area, or stand down an armed one
        else:
            m.begin_draw()

    def _cancel_draw(self):
        if self.map_page.draw_mode:
            self.map_page.clear_area()

    def _sync_bezels(self):
        pending = sum(1 for c in self.contacts if c.status == "UNCONFIRMED")
        labels = [("MAP", TEXT), ("VIS", TEXT), ("IR", TEXT),
                  (f"CNTCT\n{len(self.contacts):02d}",
                   ALERT if pending else TEXT),
                  ("SETUP", TEXT)]
        for i, (btn, (label, tone)) in enumerate(zip(self.left_buttons, labels)):
            btn.configure(label, tone, boxed=(i == self.page))

        for btn, (label, _, boxed) in zip(self.right_buttons, self._right_actions()):
            btn.configure(label, TEXT, boxed=boxed)

        sel = self._selected()
        can_rule = sel is not None and sel.status == "UNCONFIRMED"
        self.bottom_buttons[0].configure("CONFIRM", CONFIRM, enabled=can_rule)
        self.bottom_buttons[1].configure("DISMISS", RED, enabled=can_rule)
        # Lit until pressed, then stays down: the drones cannot go back
        # onto the ute from here.
        if self.node.launched:
            self.bottom_buttons[2].configure("AIRBORNE", MUTED, enabled=False)
        else:
            self.bottom_buttons[2].configure("LAUNCH", CONFIRM, boxed=True)

        # Names the action the next press performs, like every other slot.
        if self.node.estopped:
            self.estop_button.configure("RESUME", CONFIRM, boxed=True)
        else:
            self.estop_button.configure("E-STOP", RED, boxed=True)

    def _press_estop(self):
        self.node.set_estop(not self.node.estopped)

    # -- behaviour ---------------------------------------------------------
    def _selected(self) -> Optional[Contact]:
        i = self.contacts_page.selected_index()
        return self.contacts[i] if 0 <= i < len(self.contacts) else None

    def _rule(self, decision: str):
        c = self._selected()
        if c is None or c.status != "UNCONFIRMED":
            return
        c.status = decision
        self.node.send_decision(c.id, decision)
        # The evidence frames were held for this ruling; back to live video.
        self.vis_page.frozen = None
        self.ir_page.frozen = None
        if decision == "CONFIRMED":
            self._area_since_confirm = False
            self.node.publish_confirmed_point(c.x, c.y)
            self.hold_for_confirmed = True
        else:
            self._no_check_until = time.monotonic() + self.node.dismiss_cooldown_s
            # Remember the person, not just the moment: their track stays
            # ruled out however the drone comes back over them.
            fallback = (c.obj_x, c.obj_y) if c.obj_x is not None else None
            track = self.node.tracker.dismiss(c.track_id, fallback,
                                              time.monotonic())
            if track is not None:
                self.node.get_logger().info(
                    f"Contact {c.id} dismissed: ignoring track {track.id} "
                    f"(now at {track.x:.1f}, {track.y:.1f}) from now on.")
        self._update_hold()

    def _update_hold(self):
        """Hold while any contact is unruled, while the RGB camera checks a
        thermal signature, or after a confirm until the operator presses
        RESUME. A dismiss on its own lets the search go on."""
        pending = any(c.status == "UNCONFIRMED" for c in self.contacts)
        checking = self._check_until is not None
        self.node.set_hold(pending or checking or self.hold_for_confirmed)

    def _resume_search(self):
        self.hold_for_confirmed = False
        self._update_hold()

    def _raise_contact(self, x: float, y: float, rgb_conf: float,
                       thermal_conf: float, from_rgb: bool = True):
        """from_rgb False: raised on heat alone (YOLO off), so the object is
        the hot area and there is no YOLO track to dismiss with it."""
        if from_rgb:
            obj = self.node.rgb_hit_xy or (None, None)
            track = self.node.rgb_hit_track
        else:
            obj = self.node.thermal_hit_xy or (None, None)
            track = None
        self.contacts.insert(0, Contact(
            id=self._next_id, x=x, y=y,
            rgb_confidence=rgb_conf, thermal_confidence=thermal_conf,
            obj_x=obj[0], obj_y=obj[1], track_id=track))
        self._next_id += 1
        self._select_newest = True
        self._update_hold()

    def _check_detections(self):
        """Raise a contact only when there is body heat AND the RGB detector
        sees a human.

        Thermal is a checkpoint, not a detection. Body heat in view holds
        the search (up to rgb_check_s) and lets the RGB detector look; a
        human seen in that time raises the contact. Nothing seen: the heat
        was something else (e.g. an animal), so resume and don't hold for
        heat within CONTACT_RADIUS of that spot again. An RGB sighting with
        no heat (no check running, nothing hot in view) does nothing.

        Animals (kangaroo, wombat) are as warm as a person. When the detector
        names one and sees no human, the heat is treated as already checked:
        no hold if it is seen before the check starts, and a running check
        ends at once instead of waiting out rgb_check_s. They never raise a
        contact themselves.

        A rejected spot only stops the drone holding there again. Heat and a
        human seen together there (e.g. on a later sweep row) still raise a
        contact, so a person the detector missed once can still be found.

        The contact goes where the RGB detection projects onto the ground,
        using the drone's pose when that frame was taken, and falls back to
        the drone's position if it could not be projected. No checks while a
        contact is waiting for a ruling (the drone is already holding), or for
        dismiss_cooldown_s after a dismiss. A hit near a confirmed contact is
        the same person and is ignored, so RESUME doesn't re-trigger on them.

        The SETUP page can switch one detector off. Thermal off: a YOLO human
        raises the contact on its own. YOLO off: body heat raises it on its
        own, at the heat's ground position (animals included - the operator
        dismisses those).
        """
        thermal_on, rgb_on = self.node.thermal_enabled, self.node.rgb_enabled
        if self._check_until is not None and not (thermal_on and rgb_on):
            # A detector was switched off mid-check: the check needs both,
            # so it is over. With YOLO off, the heat raises a contact below.
            self._check_until = None
            self._update_hold()
        if any(c.status == "UNCONFIRMED" for c in self.contacts):
            return
        now = time.monotonic()
        if now < self._no_check_until:
            return
        x, y = self.node.drone_xy or (0.0, 0.0)
        if self._near(x, y, [(c.x, c.y) for c in self.contacts
                             if c.status == "CONFIRMED"]):
            return

        heat = self.node.thermal_hit
        if heat is not None:
            peak_k, count = heat
            heat_conf = min(1.0, count / (4.0 * self.node.person_min_pixels))
        checking = self._check_until is not None
        rgb_conf = self.node.fresh_rgb_hit()

        # Heat now, or heat that started the check still running - or no
        # heat needed, with thermal detection off.
        if rgb_conf is not None and (heat is not None or checking
                                     or not thermal_on):
            thermal_conf = heat_conf if heat is not None else 0.0
            if checking:
                thermal_conf = max(thermal_conf, self._check_heat_conf)
            obj = self.node.rgb_hit_xy
            where = f" at ({obj[0]:.1f}, {obj[1]:.1f})" if obj else ""
            why = ("thermal signature confirmed by RGB" if thermal_on
                   else "RGB human (thermal detection off)")
            self.node.get_logger().warn(
                f"Contact: {why}, "
                f"human {rgb_conf:.0%}, track {self.node.rgb_hit_track}{where}")
            # Keep the frames that triggered the contact as evidence.
            self.vis_page.frozen = self.node.display_frame
            self.ir_page.frozen = (self.node.thermal_frame
                                   if heat is not None or not checking
                                   else self._check_ir_frame)
            self._check_until = None
            # At the person, not the drone. obj is the detection projected
            # onto the ground from the pose at the moment the image was
            # taken; drone_xy is where the drone is now, which after the
            # detector's delay can be several metres further along the leg.
            cx, cy = obj if obj else (x, y)
            self._raise_contact(cx, cy, rgb_conf=rgb_conf,
                                thermal_conf=thermal_conf)
            return

        if not rgb_on:
            # YOLO off: heat is the whole test. A dismissed one is not raised
            # again: its dismissed track drops that blob in _detect_person.
            if heat is not None and not self._near(x, y, self.rejected_spots):
                obj = self.node.thermal_hit_xy
                self.node.get_logger().warn(
                    f"Contact: thermal signature ({count} px, peak "
                    f"{peak_k:.1f} K), YOLO detection off")
                self.vis_page.frozen = self.node.display_frame
                self.ir_page.frozen = self.node.thermal_frame
                cx, cy = obj if obj else (x, y)
                self._raise_contact(cx, cy, rgb_conf=0.0,
                                    thermal_conf=heat_conf, from_rgb=False)
            return

        animal = self.node.fresh_animal_only()

        if self._check_until is not None:
            if heat is not None:
                self._check_heat_conf = max(self._check_heat_conf, heat_conf)
            if animal is not None:
                # The detector has named the heat source and it is not a
                # person: no need to hold out the rest of rgb_check_s.
                self.node.get_logger().info(
                    f"Thermal signature is a {animal}, resuming search.")
                self.rejected_spots.append(self._check_xy)
                self._check_until = None
                self._update_hold()
                return
            if now >= self._check_until:
                self.node.get_logger().info(
                    "Thermal signature not confirmed by RGB, resuming search.")
                self.rejected_spots.append(self._check_xy)
                self._check_until = None
                self._update_hold()
            return

        if heat is not None and not self._near(x, y, self.rejected_spots):
            if animal is not None:
                # Already identified as an animal: don't stop the scout.
                self.node.get_logger().info(
                    f"Thermal signature is a {animal}, not holding.")
                self.rejected_spots.append((x, y))
                return
            self.node.get_logger().warn(
                f"Thermal signature ({count} px, peak {peak_k:.1f} K), "
                f"checking with the RGB camera")
            self._check_until = now + self.node.rgb_check_s
            self._check_xy = (x, y)
            self._check_heat_conf = heat_conf
            self._check_ir_frame = self.node.thermal_frame
            self._update_hold()

    def _near(self, x: float, y: float,
              spots: List[Tuple[float, float]]) -> bool:
        return any(np.hypot(sx - x, sy - y) < self.CONTACT_RADIUS
                   for sx, sy in spots)

    def _simulate_contact(self):
        """Lets you build and test the whole interface before the detector
        exists. Delete once real detections arrive."""
        bx, by = self.node.drone_xy or (0.0, 0.0)
        self._raise_contact(
            bx + np.random.uniform(-3, 3),
            by + np.random.uniform(-3, 3),
            rgb_conf=float(np.random.uniform(0.4, 0.95)),
            thermal_conf=float(np.random.uniform(0.3, 0.98)))

    def _toggle_blink(self):
        self._blink = not self._blink

    def _end_mission_if_courier_done(self):
        """The courier is done, and the scout flies home on the same message.
        Let go of it, and stop checking for heat on the way: a contact still
        waiting for a ruling does keep it there, since that is a person's
        decision to make."""
        if self._mission_over or self.node.courier_state not in ("HOME", "RETURN_FAILED"):
            return
        self._mission_over = True
        self.hold_for_confirmed = False
        self._check_until = None
        if not self._area_since_confirm:
            self._no_check_until = float("inf")
        self._update_hold()

    def _set_area(self, min_x: float, min_y: float, max_x: float, max_y: float):
        """The operator drew an area. If the scout has already been sent home,
        this is a new search, so heat checks come back on."""
        self._area_since_confirm = True
        if self._mission_over:
            self._no_check_until = 0.0
        self.node.publish_search_area(min_x, min_y, max_x, max_y)

    def refresh(self):
        self._end_mission_if_courier_done()
        self._check_detections()
        self.map_page.contacts = self.contacts
        self.contacts_page.refresh(self.contacts)
        if self._select_newest:
            # New contact is at row 0; select it so CONFIRM/DISMISS act on it.
            self.contacts_page.list.setCurrentRow(0)
            self._select_newest = False
        self.stack.currentWidget().update()

        self.page_label.setText(
            ["MAP", "VISUAL", "THERMAL", "CONTACTS", "SETUP"][self.page])
        if self.page == PAGE_SETTINGS:
            self.settings_page.refresh()

        pending = sum(1 for c in self.contacts if c.status == "UNCONFIRMED")
        if pending:
            self.caution.setText(f"CONTACT  {pending}")
            self.caution.setStyleSheet(
                f"color:{ALERT if self._blink else PANEL}; border:none;")
        else:
            self.caution.setText("")

        if self.node.estopped:
            self.hold_label.setText("E-STOP")
        elif self._check_until is not None:
            self.hold_label.setText("IR CHECK")
        else:
            self.hold_label.setText("SEARCH HOLD" if self.node.holding else "")

        # Without the detector nothing can raise a contact, so say so. A
        # detector switched off on the SETUP page says OFF, on every page,
        # so a changed rule is never a surprise.
        ai = self.node.detector_alive()
        if not self.node.rgb_enabled:
            self.ai_label.setText("AI OFF")
            ai_tone = MUTED
        else:
            self.ai_label.setText("AI OK" if ai else "AI ---")
            ai_tone = CONFIRM if ai else RED
        self.ai_label.setStyleSheet(
            f"color:{ai_tone}; border:none; padding-left:24px;")
        self.ir_label.setText("" if self.node.thermal_enabled else "IR OFF")

        live = self.node.drone_xy is not None
        self.link_label.setText("LINK OK" if live else "LINK ---")
        self.link_label.setStyleSheet(
            f"color:{CONFIRM if live else MUTED}; border:none; padding-left:24px;")

        self._sync_bezels()


def pump_ros(executor, budget_s: float = 0.015):
    """Run every ROS callback that is waiting, then hand back to Qt.

    rclpy.spin_once(node) runs ONE callback per call - it adds and removes the
    node each time, which throws away the rest of the ready set. At one call
    per 20 ms tick that is 50 callbacks a second for three camera feeds,
    odometry and the rest, so the queues filled and everything shown was
    seconds old. This drains the lot each tick instead. The budget keeps a
    burst of camera frames from freezing the window; whatever is left over
    runs on the next tick.

    One empty wait is not proof the queues are empty: rclpy reports nothing
    ready on every other wait after running a callback, so stop only after
    two in a row.
    """
    deadline = time.monotonic() + budget_s
    empty = 0
    while time.monotonic() < deadline and empty < 2:
        try:
            handler, _, _ = executor.wait_for_ready_callbacks(timeout_sec=0)
        except TimeoutException:
            empty += 1
            continue
        except ShutdownException:
            return
        empty = 0
        handler()
        if handler.exception() is not None:
            raise handler.exception()


def main():
    rclpy.init()
    node = GroundStationNode()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(node)
    window.show()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    ros_timer = QTimer()
    ros_timer.timeout.connect(lambda: pump_ros(executor))
    ros_timer.start(20)

    try:
        code = app.exec_()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()