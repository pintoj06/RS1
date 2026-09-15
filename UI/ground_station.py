#!/usr/bin/env python3
"""AeroAid ground station  AMPCD-style operator interface.

    ros2 run 41068_ignition_bringup ground_station.py --ros-args \
        -r __ns:=/parrot1 -p robot_name:=parrot1

Layout follows multifunction-display convention: one large display area with
option select buttons around the bezel. The buttons are generic; the labels
change to suit whichever page is up, and the label tells you what the button
currently does.

Slot allocation is deliberate and fixed:
    left column    page selection  never changes, so it can be hit blind
    right column   actions for the current page
    bottom row     contact ruling  always available, because a decision on a
                   possible person should never be more than one press away

Keyboard equivalents: 1-4 left, F1-F5 right, Q/W/E bottom.

Qt owns the event loop; rclpy is pumped from a QTimer (see main()). Never
block inside a callback or the window freezes.
"""

import sys
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy, qos_profile_sensor_data)

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import String
import tf2_ros

from cv_bridge import CvBridge
import cv2

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
# loudest colour in the palette and is reserved for one thing only  a contact
# nobody has ruled on yet. Nothing else is allowed to compete with it.
# ---------------------------------------------------------------------------
BASE = "#1C2229"
PANEL = "#232B33"
BEZEL = "#151A1F"
LINE = "#333E49"
TEXT = "#E4E9ED"
MUTED = "#8A98A6"
AMBER = "#E8A33D"
GREEN = "#4CAF6D"
RED = "#D9534F"
CYAN = "#5AA9E6"

MAP_UNKNOWN = (58, 66, 74)
MAP_FREE = (226, 232, 238)
MAP_WALL = (26, 31, 36)

PAGE_MAP, PAGE_VIS, PAGE_IR, PAGE_CONTACTS = range(4)


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
    """Caches the latest data. No Qt, no drawing  the GUI reads whatever is
    current when it repaints, which decouples frame rate from message rate."""

    def __init__(self):
        super().__init__("ground_station")

        self.declare_parameter("robot_name", "parrot1")
        self.robot_name = str(self.get_parameter("robot_name").value)

        self.bridge = CvBridge()
        self.rgb_frame: Optional[np.ndarray] = None
        self.thermal_frame: Optional[np.ndarray] = None

        self.map_image: Optional[np.ndarray] = None
        self.map_resolution = 0.0
        self.map_origin = (0.0, 0.0)
        self.map_frame = ""

        self.drone_xy: Optional[Tuple[float, float]] = None
        self.drone_yaw = 0.0
        self.trail = deque(maxlen=3000)

        self.base_frame = f"{self.robot_name}_base_link"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # SLAM publishes the map transient-local. A default subscription
        # receives nothing at all, with no error  this QoS is required.
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
        self.create_subscription(OccupancyGrid, "map", self._on_map, map_qos)

        self.decision_pub = self.create_publisher(String, "operator/decision", 10)
        self.create_timer(0.1, self._poll_pose)

        self.get_logger().info(f"Ground station up for {self.robot_name}")

    def _on_rgb(self, msg: Image):
        try:
            self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warn(f"RGB decode failed: {exc}", once=True)

    def _on_thermal(self, msg: Image):
        """Thermal may arrive mono or colour depending on the xacro. Handle
        both, and false-colour mono so heat reads instantly rather than as
        grey mush."""
        try:
            if msg.encoding in ("mono8", "8UC1"):
                mono = self.bridge.imgmsg_to_cv2(msg, "mono8")
                self.thermal_frame = cv2.applyColorMap(mono, cv2.COLORMAP_INFERNO)
            else:
                self.thermal_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Thermal decode failed: {exc}", once=True)

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
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_frame = msg.header.frame_id

    def _poll_pose(self):
        """Ask TF where the drone is, in the map's own frame.

        Using TF rather than /odometry matters: the search node plans in the
        odom frame while the map lives in the map frame, and SLAM corrections
        pull those apart. Looking up into whatever frame the map header
        declares keeps the drone marker on the map over a long mission.
        """
        if not self.map_frame:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
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

    def send_decision(self, contact_id: int, decision: str):
        self.decision_pub.publish(String(data=f"{decision}:{contact_id}"))
        self.get_logger().info(f"Operator {decision} contact {contact_id}")


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

    Generic hardware, page-specific label  the whole point of the layout.
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

    PIP_MODES = [
        ("OFF",  []),
        ("IR",   ["thermal"]),
        ("VIS",  ["rgb"]),
        ("BOTH", ["thermal", "rgb"]),
    ]

    def __init__(self, node: GroundStationNode):
        super().__init__()
        self.pip_mode = 1        # IR by default
        self.node = node
        self.contacts: List[Contact] = []
        self.show_trail = True
        self.show_contacts = True
        self.zoom = 1.0
        self.setStyleSheet(f"background:{BASE};")
        self._scale, self._ox, self._oy = 1.0, 0.0, 0.0

    def cycle_pip(self):
        self.pip_mode = (self.pip_mode + 1) % len(self.PIP_MODES)

    def _world_to_widget(self, x: float, y: float) -> QPointF:
        res = self.node.map_resolution
        ox, oy = self.node.map_origin
        h = self.node.map_image.shape[0]
        px = (x - ox) / res
        py = h - (y - oy) / res
        return QPointF(self._ox + px * self._scale, self._oy + py * self._scale)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BASE))

        if self.node.map_image is None:
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "No map. Start the simulation with nav2:=true")
            return

        img = self.node.map_image
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()

        # Fit to the widget, then apply zoom, keeping the drone centred when
        # zoomed in  panning a map by hand during a search is a distraction.
        fit = min(self.width() / w, self.height() / h)
        self._scale = fit * self.zoom
        if self.zoom > 1.0 and self.node.drone_xy:
            dx = (self.node.drone_xy[0] - self.node.map_origin[0]) / self.node.map_resolution
            dy = h - (self.node.drone_xy[1] - self.node.map_origin[1]) / self.node.map_resolution
            self._ox = self.width() / 2 - dx * self._scale
            self._oy = self.height() / 2 - dy * self._scale
        else:
            self._ox = (self.width() - w * self._scale) / 2
            self._oy = (self.height() - h * self._scale) / 2

        p.drawPixmap(QRectF(self._ox, self._oy, w * self._scale, h * self._scale),
                     QPixmap.fromImage(qimg), QRectF(0, 0, w, h))

        if self.show_trail and len(self.node.trail) > 1:
            p.setPen(QPen(QColor(CYAN), 2))
            pts = [self._world_to_widget(x, y) for x, y in self.node.trail]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)

        if self.show_contacts:
            for c in self.contacts:
                if c.status == "DISMISSED":
                    continue
                pos = self._world_to_widget(c.x, c.y)
                colour = QColor(GREEN if c.status == "CONFIRMED" else AMBER)
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
            p.setBrush(QColor(CYAN))
            p.setPen(QPen(QColor(TEXT), 1))
            p.drawPolygon(QPointF(12, 0), QPointF(-7, 7), QPointF(-7, -7))
            p.restore()

        if self.PIP_MODES[self.pip_mode][1]:
            self._draw_pips(p)

    def _draw_pips(self, p: QPainter):
        """Insets stack upward from the bottom-right corner. Two at once eats
        real map area, which is why OFF and single-feed modes exist — the
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
        self.source = source            # "rgb" or "thermal"
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
            p.setPen(QColor(AMBER))
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
            item.setForeground(QColor({"UNCONFIRMED": AMBER,
                                       "CONFIRMED": GREEN,
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
    def __init__(self, node: GroundStationNode):
        super().__init__()
        self.node = node
        self.contacts: List[Contact] = []
        self._next_id = 1
        self._blink = False
        self.page = PAGE_MAP

        self.setWindowTitle("AeroAid  ground station")
        self.resize(1440, 900)
        self.setStyleSheet(f"QMainWindow {{ background:{BEZEL}; }}"
                           f"QLabel {{ color:{TEXT}; }}")

        self.map_page = MapPage(node)
        self.vis_page = VideoPage(node, "rgb", "No camera feed")
        self.ir_page = VideoPage(node, "thermal", "No thermal feed")
        self.contacts_page = ContactsPage()

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

        self.right_buttons = [BezelButton(f"R{i+1}") for i in range(5)]
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
        self.bottom_buttons[2].clicked.connect(self._simulate_contact)
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
        self.page_label.setStyleSheet(f"color:{CYAN}; border:none; padding-left:20px;")
        row.addWidget(self.page_label)

        row.addStretch(1)

        # Master caution. A single-display layout can hide a new contact
        # behind whichever page is up, so one indicator stays visible on every
        # page and is the only thing in the interface that blinks.
        self.caution = QLabel("")
        self.caution.setFont(QFont("DejaVu Sans Mono", 11, QFont.Bold))
        self.caution.setStyleSheet("border:none;")
        row.addWidget(self.caution)

        self.link_label = QLabel("LINK ---")
        self.link_label.setFont(QFont("DejaVu Sans Mono", 10))
        self.link_label.setStyleSheet(f"color:{MUTED}; border:none; padding-left:24px;")
        row.addWidget(self.link_label)
        return bar

    def _build_shortcuts(self):
        for i, key in enumerate(["1", "2", "3", "4"]):
            QShortcut(QKeySequence(key), self, lambda n=i: self.select_page(n))
        for i in range(5):
            QShortcut(QKeySequence(f"F{i+1}"), self,
                      lambda n=i: self._press_right(n))
        QShortcut(QKeySequence("Q"), self, lambda: self._rule("CONFIRMED"))
        QShortcut(QKeySequence("W"), self, lambda: self._rule("DISMISSED"))
        QShortcut(QKeySequence("E"), self, self._simulate_contact)

    # -- page plumbing -----------------------------------------------------
    def select_page(self, index: int):
        self.page = index
        self.stack.setCurrentIndex(index)

    def _right_actions(self) -> List[Tuple[str, Callable, bool]]:
        """Label, handler and boxed-state for the right column, for whichever
        page is current. Rebuilding it each refresh is what lets a toggle show
        its own state in the label box."""
        if self.page == PAGE_MAP:
            m = self.map_page
            return [
                ("TRAIL", lambda: setattr(m, "show_trail", not m.show_trail), m.show_trail),
                ("CNTCT", lambda: setattr(m, "show_contacts", not m.show_contacts), m.show_contacts),
                # ("IR PIP", lambda: setattr(m, "pip", not m.pip), m.pip),
                (f"PIP\n{MapPage.PIP_MODES[m.pip_mode][0]}", m.cycle_pip, m.pip_mode != 0),
                ("ZOOM+", lambda: setattr(m, "zoom", min(m.zoom * 1.5, 8.0)), False),
                ("ZOOM-", lambda: setattr(m, "zoom", max(m.zoom / 1.5, 1.0)), False),
            ]
        if self.page in (PAGE_VIS, PAGE_IR):
            v = self.vis_page if self.page == PAGE_VIS else self.ir_page
            return [
                ("FREEZE", v.toggle_freeze, v.frozen is not None),
                ("", lambda: None, False),
                ("", lambda: None, False),
                ("", lambda: None, False),
                ("", lambda: None, False),
            ]
        return [("", lambda: None, False)] * 5

    def _press_right(self, slot: int):
        actions = self._right_actions()
        if slot < len(actions) and actions[slot][0]:
            actions[slot][1]()

    def _sync_bezels(self):
        pending = sum(1 for c in self.contacts if c.status == "UNCONFIRMED")
        labels = [("MAP", TEXT), ("VIS", TEXT), ("IR", TEXT),
                  (f"CNTCT\n{len(self.contacts):02d}",
                   AMBER if pending else TEXT)]
        for i, (btn, (label, tone)) in enumerate(zip(self.left_buttons, labels)):
            btn.configure(label, tone, boxed=(i == self.page))

        for btn, (label, _, boxed) in zip(self.right_buttons, self._right_actions()):
            btn.configure(label, TEXT, boxed=boxed)

        sel = self._selected()
        can_rule = sel is not None and sel.status == "UNCONFIRMED"
        self.bottom_buttons[0].configure("CONFIRM", GREEN, enabled=can_rule)
        self.bottom_buttons[1].configure("DISMISS", RED, enabled=can_rule)
        self.bottom_buttons[2].configure("SIM", MUTED)

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

    def _simulate_contact(self):
        """Lets you build and test the whole interface before the detector
        exists. Delete once real detections arrive."""
        bx, by = self.node.drone_xy or (0.0, 0.0)
        self.contacts.insert(0, Contact(
            id=self._next_id,
            x=bx + np.random.uniform(-3, 3),
            y=by + np.random.uniform(-3, 3),
            rgb_confidence=float(np.random.uniform(0.4, 0.95)),
            thermal_confidence=float(np.random.uniform(0.3, 0.98)),
        ))
        self._next_id += 1

    def _toggle_blink(self):
        self._blink = not self._blink

    def refresh(self):
        self.map_page.contacts = self.contacts
        self.contacts_page.refresh(self.contacts)
        self.stack.currentWidget().update()

        self.page_label.setText(
            ["MAP", "VISUAL", "THERMAL", "CONTACTS"][self.page])

        pending = sum(1 for c in self.contacts if c.status == "UNCONFIRMED")
        if pending:
            self.caution.setText(f"CONTACT  {pending}")
            self.caution.setStyleSheet(
                f"color:{AMBER if self._blink else PANEL}; border:none;")
        else:
            self.caution.setText("")

        live = self.node.drone_xy is not None
        self.link_label.setText("LINK OK" if live else "LINK ---")
        self.link_label.setStyleSheet(
            f"color:{GREEN if live else MUTED}; border:none; padding-left:24px;")

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
    main() 