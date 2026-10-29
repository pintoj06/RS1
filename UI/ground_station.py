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

Keyboard equivalents: 1-4 left, F1-F6 right, Q/W/R bottom, Esc cancels an
area drag. E raises a fake contact for testing the interface.

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
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy, qos_profile_sensor_data)

from geometry_msgs.msg import Point32, PointStamped, PolygonStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from vision_msgs.msg import Detection2DArray
import tf2_ros

from cv_bridge import CvBridge
import cv2
import yaml

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter, QPen,
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
PLANNED     = "#00FF88"            # planned sweep route — bright green.
                                   # Kept clear of ALERT's yellow-green so
                                   # an unruled contact still reads as the
                                   # one thing wanting a decision.

PAGE_MAP, PAGE_VIS, PAGE_IR, PAGE_CONTACTS = range(4)

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
        self._detections_time = 0.0
        # (peak kelvin, hot pixel count) for the latest frame, None if nothing.
        self.thermal_hit: Optional[Tuple[float, int]] = None

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
        # Planned sweep route, in the map frame once transformed.
        self.search_path: List[Tuple[float, float]] = []

        self.drone_xy: Optional[Tuple[float, float]] = None
        self.drone_yaw = 0.0
        self.trail = deque(maxlen=3000)

        self.base_frame = f"{self.robot_name}_base_link"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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

        self.decision_pub = self.create_publisher(String, "operator/decision", 10)

        # Tells search_node to stop and hover. Transient local so a search
        # node started later still picks up the current state.
        hold_qos = QoSProfile(depth=1,
                              reliability=QoSReliabilityPolicy.RELIABLE,
                              durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.hold_pub = self.create_publisher(Bool, "search/hold", hold_qos)
        self.holding = False
        self.hold_pub.publish(Bool(data=False))

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

    def _on_detections(self, msg: Detection2DArray):
        now = time.monotonic()
        self._detections_time = now
        scores = [r.hypothesis.score for d in msg.detections for r in d.results]
        if scores:
            self.rgb_hit = (max(scores), now)

    def fresh_rgb_hit(self) -> Optional[float]:
        """Confidence of a human seen in the last rgb_fresh_s, else None."""
        if self.rgb_hit is None:
            return None
        conf, seen = self.rgb_hit
        return conf if time.monotonic() - seen < self.rgb_fresh_s else None

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
                self.thermal_hit = self._detect_person(kelvin, frame)
                self.thermal_frame = frame
            elif msg.encoding in ("mono8", "8UC1"):
                mono = self.bridge.imgmsg_to_cv2(msg, "mono8")
                self.thermal_frame = cv2.applyColorMap(mono, cv2.COLORMAP_INFERNO)
            else:
                self.thermal_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Thermal decode failed: {exc}", once=True)

    def _detect_person(self, kelvin: np.ndarray,
                       frame: np.ndarray) -> Optional[Tuple[float, int]]:
        """Threshold for body heat. Boxes the hot region on the frame so a
        frozen frame shows what triggered the check."""
        hot = kelvin >= self.person_min_k
        count = int(np.count_nonzero(hot))
        if count < self.person_min_pixels:
            return None
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
                rclpy.time.Time(), timeout=Duration(seconds=0.05))
        except Exception:
            return

        grid = np.asarray(msg.data, dtype=np.int8).reshape(h, w)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[grid > 0] = COVERED            # everything else stays transparent
        # Row 0 of an OccupancyGrid is the bottom of the world, row 0 of an
        # image is the top - same flip the SLAM map does.
        self.coverage_image = np.flipud(rgba).copy()

        res = msg.info.resolution
        ox = msg.info.origin.position.x + tf.transform.translation.x
        oy = msg.info.origin.position.y + tf.transform.translation.y
        self.coverage_bounds = (ox, oy, ox + w * res, oy + h * res)

        # Percentage of the cells the scout counts as searchable. It already
        # leaves obstacles and unreachable ground out of its grid, so this is
        # progress against what can actually be flown, not the whole box.
        searched = int(np.count_nonzero(grid > 0))
        total = int(np.count_nonzero(grid >= 0))
        self.coverage_percent = (100.0 * searched / total) if total else None

    def _on_path(self, msg: Path):
        """Store the scout's planned sweep, shifted into this map's frame.

        Same odom-to-map offset the coverage overlay needs; without it the
        route would be drawn about a metre from where the drone actually flies.
        """
        if not msg.poses:
            self.search_path = []
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, msg.header.frame_id,
                rclpy.time.Time(), timeout=Duration(seconds=0.05))
        except Exception:
            return
        dx = tf.transform.translation.x
        dy = tf.transform.translation.y
        self.search_path = [(p.pose.position.x + dx, p.pose.position.y + dy)
                            for p in msg.poses]

    def _poll_pose(self):
        """Ask TF where the drone is, in the map's own frame.

        Using TF rather than /odometry matters: the search node plans in the
        odom frame while the map lives in the map frame, and SLAM corrections
        pull those apart. Looking up into whatever frame the map header
        declares keeps the drone marker on the map over a long mission.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.05))
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
                    self.odom_frame, self.base_frame, rclpy.time.Time(),
                    timeout=Duration(seconds=0.05))
            except Exception:
                return  # normal at startup, before transforms flow
            x = tf.transform.translation.x
            y = tf.transform.translation.y
        self.drone_xy = (x, y)

        q = tf.transform.rotation
        self.drone_yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y ** 2 + q.z ** 2))

        if not self.trail or (abs(self.trail[-1][0] - x) > 0.05 or
                              abs(self.trail[-1][1] - y) > 0.05):
            self.trail.append((x, y))

    def set_hold(self, hold: bool):
        if hold == self.holding:
            return
        self.holding = hold
        self.hold_pub.publish(Bool(data=hold))
        self.get_logger().info("Search HOLD" if hold else "Search resumed")

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

        if self.node.drone_xy:
            pos = self._world_to_widget(*self.node.drone_xy)
            p.save()
            p.translate(pos)
            p.rotate(-np.degrees(self.node.drone_yaw))
            p.setBrush(QColor(DRONE))
            p.setPen(QPen(QColor(TEXT), 1))
            p.drawPolygon(QPointF(12, 0), QPointF(-7, 7), QPointF(-7, -7))
            p.restore()

        # How much of the searchable ground has been swept. The one number
        # that answers "is this area done?", so it stays on screen rather than
        # living only in the search node's terminal.
        if self.show_coverage and self.node.coverage_percent is not None:
            p.setPen(QColor(CONFIRM if self.node.coverage_percent >= 99.5 else TEXT))
            p.setFont(QFont("DejaVu Sans Mono", 10, QFont.Bold))
            p.drawText(18, self.height() - 18,
                       f"SEARCHED  {self.node.coverage_percent:5.1f}%")

        if self.draw_mode and self._drag_a is None:
            p.setPen(QColor(ALERT))
            p.setFont(QFont("DejaVu Sans Mono", 10, QFont.Bold))
            p.drawText(18, 28, "DRAG TO SET SEARCH AREA   (ESC TO CANCEL)")

        if self.PIP_MODES[self.pip_mode][1]:
            self._draw_pips(p)

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
            p.drawRect(x, y, pix.width(), pix.height())
            p.setPen(QColor(MUTED))
            p.setFont(QFont("DejaVu Sans Mono", 8))
            p.drawText(x + 5, y + 14, "IR" if source == "thermal" else "VIS")
            y -= pix.height() + 8


class VideoPage(QWidget):
    """Full-bleed camera feed with a frozen-frame option."""

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

    def toggle_freeze(self):
        """Freezing matters operationally: the drone keeps flying while the
        operator studies a frame, and without this the evidence is gone by the
        time they have looked at it."""
        frame = getattr(self.node, f"{self.source}_frame")
        self.frozen = None if self.frozen is not None else frame

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

        self.map_page.on_area_set = self.node.publish_search_area

        self.stack = QStackedWidget()
        for page in (self.map_page, self.vis_page, self.ir_page,
                     self.contacts_page):
            self.stack.addWidget(page)
        self.stack.setStyleSheet(f"border:1px solid {LINE};")

        self._build_layout()
        self._build_shortcuts()
        self.select_page(PAGE_MAP)

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

        self.left_buttons = [BezelButton(f"L{i+1}") for i in range(4)]
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
        grid.addLayout(bottom, 2, 1)

        grid.setColumnStretch(1, 1)
        grid.setRowStretch(1, 1)

    def _status_strip(self) -> QWidget:
        bar = QFrame()
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

        self.link_label = QLabel("LINK ---")
        self.link_label.setFont(QFont("DejaVu Sans Mono", 10))
        self.link_label.setStyleSheet(f"color:{MUTED}; border:none; padding-left:24px;")
        row.addWidget(self.link_label)
        return bar

    def _build_shortcuts(self):
        for i, key in enumerate(["1", "2", "3", "4"]):
            QShortcut(QKeySequence(key), self, lambda n=i: self.select_page(n))
        for i in range(RIGHT_SLOTS):
            QShortcut(QKeySequence(f"F{i+1}"), self,
                      lambda n=i: self._press_right(n))
        QShortcut(QKeySequence("Q"), self, lambda: self._rule("CONFIRMED"))
        QShortcut(QKeySequence("W"), self, lambda: self._rule("DISMISSED"))
        QShortcut(QKeySequence("R"), self, self.node.launch)
        # Fake contact for testing the interface; keyboard only.
        QShortcut(QKeySequence("E"), self, self._simulate_contact)
        # Escape stands down an armed area draw without setting one.
        QShortcut(QKeySequence("Escape"), self, self._cancel_draw)

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
        if self.page in (PAGE_VIS, PAGE_IR):
            v = self.vis_page if self.page == PAGE_VIS else self.ir_page
            return ([("FREEZE", v.toggle_freeze, v.frozen is not None)]
                    + [blank] * (RIGHT_SLOTS - 1))
        if self.page == PAGE_CONTACTS:
            return [("RESUME", self._resume_search, False)] + [blank] * (RIGHT_SLOTS - 1)
        return [blank] * RIGHT_SLOTS

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
                   ALERT if pending else TEXT)]
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
        if decision == "CONFIRMED":
            self.node.publish_confirmed_point(c.x, c.y)
            self.hold_for_confirmed = True
        else:
            self._no_check_until = time.monotonic() + self.node.dismiss_cooldown_s
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
                       thermal_conf: float):
        self.contacts.insert(0, Contact(
            id=self._next_id, x=x, y=y,
            rgb_confidence=rgb_conf, thermal_confidence=thermal_conf))
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

        A rejected spot only stops the drone holding there again. Heat and a
        human seen together there (e.g. on a later sweep row) still raise a
        contact, so a person the detector missed once can still be found.

        The contact goes at the drone's position; the cameras look down from
        close above, so that is within a few metres. No checks while a
        contact is waiting for a ruling (the drone is already holding), or for
        dismiss_cooldown_s after a dismiss. A hit near a confirmed contact is
        the same person and is ignored, so RESUME doesn't re-trigger on them.
        """
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

        # Heat now, or heat that started the check still running.
        if rgb_conf is not None and (heat is not None or checking):
            thermal_conf = heat_conf if heat is not None else 0.0
            if checking:
                thermal_conf = max(thermal_conf, self._check_heat_conf)
            self.node.get_logger().warn(
                f"Contact: thermal signature confirmed by RGB, "
                f"human {rgb_conf:.0%}")
            # Keep the frames that triggered the contact as evidence.
            self.vis_page.frozen = self.node.display_frame
            self.ir_page.frozen = (self.node.thermal_frame if heat is not None
                                   else self._check_ir_frame)
            self._check_until = None
            self._raise_contact(x, y, rgb_conf=rgb_conf,
                                thermal_conf=thermal_conf)
            return

        if self._check_until is not None:
            if heat is not None:
                self._check_heat_conf = max(self._check_heat_conf, heat_conf)
            if now >= self._check_until:
                self.node.get_logger().info(
                    "Thermal signature not confirmed by RGB, resuming search.")
                self.rejected_spots.append(self._check_xy)
                self._check_until = None
                self._update_hold()
            return

        if heat is not None and not self._near(x, y, self.rejected_spots):
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

    def refresh(self):
        self._check_detections()
        self.map_page.contacts = self.contacts
        self.contacts_page.refresh(self.contacts)
        if self._select_newest:
            # New contact is at row 0; select it so CONFIRM/DISMISS act on it.
            self.contacts_page.list.setCurrentRow(0)
            self._select_newest = False
        self.stack.currentWidget().update()

        self.page_label.setText(
            ["MAP", "VISUAL", "THERMAL", "CONTACTS"][self.page])

        pending = sum(1 for c in self.contacts if c.status == "UNCONFIRMED")
        if pending:
            self.caution.setText(f"CONTACT  {pending}")
            self.caution.setStyleSheet(
                f"color:{ALERT if self._blink else PANEL}; border:none;")
        else:
            self.caution.setText("")

        if self._check_until is not None:
            self.hold_label.setText("IR CHECK")
        else:
            self.hold_label.setText("SEARCH HOLD" if self.node.holding else "")

        # Without the detector nothing can raise a contact, so say so.
        ai = self.node.detector_alive()
        self.ai_label.setText("AI OK" if ai else "AI ---")
        self.ai_label.setStyleSheet(
            f"color:{CONFIRM if ai else RED}; border:none; padding-left:24px;")

        live = self.node.drone_xy is not None
        self.link_label.setText("LINK OK" if live else "LINK ---")
        self.link_label.setStyleSheet(
            f"color:{CONFIRM if live else MUTED}; border:none; padding-left:24px;")

        self._sync_bezels()


def main():
    rclpy.init()
    node = GroundStationNode()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(node)
    window.show()

    # spin_once with a zero timeout drains whatever ROS messages are waiting
    # and returns immediately, so the GUI never stalls.
    ros_timer = QTimer()
    ros_timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0))
    ros_timer.start(20)

    try:
        code = app.exec_()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()