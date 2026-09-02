#!/usr/bin/env python3
"""Scout drone search-grid sweep planner.

This node is the decision-making layer for the scout drone. It does not read
the camera and does not do any perception - that is a separate node's job.
It only needs two things Nav2/SLAM already provide: the ability to send goals
to Nav2's NavigateToPose action. It turns a rectangular search area into a
boustrophedon ("lawnmower") sweep pattern - a series of back-and-forth rows -
and sends each waypoint to Nav2 in order, one at a time.

For now the search rectangle is set via ROS parameters (hardcoded at launch).
Later this can be replaced by a subscription to a topic the mission UI
publishes to, without changing the sweep/Nav2 logic below.

Same relative-topic pattern as basic_autonomy_demo.py: this node is launched
inside a robot namespace (e.g. /parrot1), so 'navigate_to_pose' becomes
'/parrot1/navigate_to_pose' automatically.
"""

import time
from typing import List, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray


class SearchManagerNode(Node):
    """Sweeps a rectangular search area and sends waypoints to Nav2 in order."""

    def __init__(self):
        super().__init__('search_manager_node')

        # ------------------------------------------------------------------
        # Parameters - this is the "search grid" the UI will eventually set.
        # ------------------------------------------------------------------
        self.declare_parameter('robot_name', '')
        self.declare_parameter('navigate_action', 'navigate_to_pose')

        # Rectangle corners, in the robot's map frame (metres). Defaults are
        # a small area near the origin so this is quick to test on its own.
        self.declare_parameter('search_min_x', -5.0)
        self.declare_parameter('search_max_x', 5.0)
        self.declare_parameter('search_min_y', -5.0)
        self.declare_parameter('search_max_y', 5.0)

        # Distance between sweep rows. Smaller = more thorough coverage but
        # more waypoints (slower to complete).
        self.declare_parameter('lane_spacing', 2.0)

        # Pause briefly between waypoints so Nav2 settles and logs stay readable.
        self.declare_parameter('goal_pause_seconds', 1.0)

        self.robot_name = self._resolve_robot_name()
        self.map_frame = f'{self.robot_name}_map'
        self.base_frame = f'{self.robot_name}_base_link'

        # TF is used only as a readiness check: a successful lookup means
        # localisation/TF has actually started publishing, so it is safe to
        # send Nav2 goals. Same pattern basic_autonomy_demo.py uses to avoid
        # sending goals before Nav2's pipeline has anything usable yet.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------
        # Build the sweep plan once at startup.
        # ------------------------------------------------------------------
        self.waypoints: List[Tuple[float, float]] = self._build_sweep_waypoints()
        self.next_waypoint_index = 0

        self.get_logger().info(
            f'Search manager started for robot "{self.robot_name}". '
            f'Planned {len(self.waypoints)} waypoints covering the search area.'
        )

        # Nav2 action client - identical pattern to basic_autonomy_demo.py.
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            self.get_parameter('navigate_action').value,
        )
        self.goal_active = False
        self.goal_handle = None
        self.next_goal_time = 0.0
        self.last_waiting_log_time = 0.0

        # Visualisation of the plan (requirement R2: "visual output feedback of
        # the drone's planned path"). These are republished on a timer rather
        # than published once, so RViz shows them no matter when you add the
        # display or which QoS the display is using.
        latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(Path, 'search_manager/planned_path', latched_qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, 'search_manager/search_area', latched_qos
        )

        self.timer = self.create_timer(1.0, self._tick)
        self.viz_timer = self.create_timer(1.0, self._publish_visualisation)

    def _resolve_robot_name(self) -> str:
        """Same fallback logic as basic_autonomy_demo.py."""
        configured_name = str(self.get_parameter('robot_name').value).strip().strip('/')
        if configured_name:
            return configured_name

        namespace = self.get_namespace().strip('/')
        if namespace:
            return namespace

        return 'parrot1'

    def _build_sweep_waypoints(self) -> List[Tuple[float, float]]:
        """Turn the search rectangle into a boustrophedon (lawnmower) sweep.

        Produces rows spaced by lane_spacing along y, alternating the x
        direction each row so consecutive waypoints are always adjacent
        (no long diagonal jumps back to the start of the next row).
        """
        min_x = float(self.get_parameter('search_min_x').value)
        max_x = float(self.get_parameter('search_max_x').value)
        min_y = float(self.get_parameter('search_min_y').value)
        max_y = float(self.get_parameter('search_max_y').value)
        lane_spacing = float(self.get_parameter('lane_spacing').value)

        if max_x <= min_x or max_y <= min_y or lane_spacing <= 0.0:
            self.get_logger().error(
                'Invalid search area parameters; producing an empty sweep plan.'
            )
            return []

        waypoints: List[Tuple[float, float]] = []
        y = min_y
        left_to_right = True
        while y <= max_y + 1e-6:
            if left_to_right:
                waypoints.append((min_x, y))
                waypoints.append((max_x, y))
            else:
                waypoints.append((max_x, y))
                waypoints.append((min_x, y))
            left_to_right = not left_to_right
            y += lane_spacing

        return waypoints

    def _publish_visualisation(self) -> None:
        """Republish the planned path and the search-area markers.

        Called on a timer so the displays appear in RViz regardless of when
        the display was added or what QoS settings it uses.
        """
        self._publish_planned_path()
        self._publish_search_area_markers()

    def _publish_planned_path(self) -> None:
        """Publish the full sweep as a nav_msgs/Path for RViz visualisation."""
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        for x, y in self.waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose = self._pose_from_xy(x, y)
            path.poses.append(pose)

        self.path_pub.publish(path)

    def _publish_search_area_markers(self) -> None:
        """Draw the search rectangle, the sweep line, and numbered waypoints.

        Markers are much easier to see in RViz than a thin Path line, and the
        numbers make it obvious whether the drone is visiting waypoints in the
        expected order and covering the whole area.
        """
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        min_x = float(self.get_parameter('search_min_x').value)
        max_x = float(self.get_parameter('search_max_x').value)
        min_y = float(self.get_parameter('search_min_y').value)
        max_y = float(self.get_parameter('search_max_y').value)

        # 1. The search area boundary, as a closed green rectangle.
        boundary = Marker()
        boundary.header.frame_id = self.map_frame
        boundary.header.stamp = stamp
        boundary.ns = 'search_area'
        boundary.id = 0
        boundary.type = Marker.LINE_STRIP
        boundary.action = Marker.ADD
        boundary.scale.x = 0.15  # line thickness
        boundary.color.r, boundary.color.g, boundary.color.b, boundary.color.a = 0.0, 1.0, 0.0, 1.0
        boundary.pose.orientation.w = 1.0
        for x, y in [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y),
        ]:
            boundary.points.append(self._point(x, y))
        markers.markers.append(boundary)

        # 2. The sweep route itself, as a thick cyan line through the waypoints.
        route = Marker()
        route.header.frame_id = self.map_frame
        route.header.stamp = stamp
        route.ns = 'sweep_route'
        route.id = 1
        route.type = Marker.LINE_STRIP
        route.action = Marker.ADD
        route.scale.x = 0.1
        route.color.r, route.color.g, route.color.b, route.color.a = 0.0, 1.0, 1.0, 1.0
        route.pose.orientation.w = 1.0
        for x, y in self.waypoints:
            route.points.append(self._point(x, y))
        markers.markers.append(route)

        # 3. A numbered label at each waypoint. Already-visited waypoints are
        #    dimmed so progress through the sweep is obvious at a glance.
        for index, (x, y) in enumerate(self.waypoints):
            label = Marker()
            label.header.frame_id = self.map_frame
            label.header.stamp = stamp
            label.ns = 'waypoint_labels'
            label.id = 100 + index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose = self._pose_from_xy(x, y)
            label.pose.position.z = 0.5
            label.scale.z = 0.5  # text height
            visited = index < self.next_waypoint_index
            if visited:
                label.color.r, label.color.g, label.color.b, label.color.a = 0.5, 0.5, 0.5, 0.6
            else:
                label.color.r, label.color.g, label.color.b, label.color.a = 1.0, 1.0, 0.0, 1.0
            label.text = str(index + 1)
            markers.markers.append(label)

        self.marker_pub.publish(markers)

    @staticmethod
    def _point(x: float, y: float) -> Point:
        point = Point()
        point.x = x
        point.y = y
        point.z = 0.0
        return point

    @staticmethod
    def _pose_from_xy(x: float, y: float) -> Pose:
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.orientation.w = 1.0
        return pose

    def _tick(self) -> None:
        """One step of the sweep state machine."""
        if self.goal_active:
            return

        if time.monotonic() < self.next_goal_time:
            return

        if self.next_waypoint_index >= len(self.waypoints):
            self._log_waiting('Search sweep complete. No more waypoints to send.')
            return

        if not self.nav_client.server_is_ready():
            if not self.nav_client.wait_for_server(timeout_sec=0.1):
                self._log_waiting('Waiting for Nav2 NavigateToPose action server...')
                return

        if not self._tf_ready():
            self._log_waiting(
                f'Waiting for TF from {self.map_frame} to {self.base_frame} '
                'before starting the sweep...'
            )
            return

        x, y = self.waypoints[self.next_waypoint_index]
        self._send_goal(x, y)

    def _tf_ready(self) -> bool:
        """Return True once TF between the map and base frames is available."""
        return self._lookup_robot_xy() is not None

    def _lookup_robot_xy(self) -> Tuple[float, float]:
        """Return the drone's current (x, y) in the map frame, or None."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
        except Exception as exc:  # TF exceptions are numerous; keep this compact.
            self.get_logger().debug(f'TF lookup failed: {exc}')
            return None

        return (transform.transform.translation.x, transform.transform.translation.y)

    def _log_waiting(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_waiting_log_time > 5.0:
            self.last_waiting_log_time = now
            self.get_logger().info(message)

    def _send_goal(self, x: float, y: float) -> None:
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = self.map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose = self._pose_from_xy(x, y)

        self.goal_active = True
        self.get_logger().info(
            f'Sending sweep waypoint {self.next_waypoint_index + 1}/{len(self.waypoints)}: '
            f'({x:.2f}, {y:.2f}) in frame {self.map_frame}.'
        )

        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future) -> None:
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected the sweep waypoint. Retrying shortly.')
            self.goal_active = False
            self._schedule_next_attempt()
            return

        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future) -> None:
        wrapped_result = future.result()
        status = wrapped_result.status

        # Report where the drone actually ended up, next to where it was aimed,
        # so the sweep can be checked from the terminal without relying on RViz.
        target_x, target_y = self.waypoints[self.next_waypoint_index]
        actual = self._lookup_robot_xy()
        if actual is not None:
            actual_text = f'actually at ({actual[0]:.2f}, {actual[1]:.2f})'
        else:
            actual_text = 'actual position unavailable'

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                f'Reached waypoint {self.next_waypoint_index + 1}/{len(self.waypoints)}: '
                f'aimed at ({target_x:.2f}, {target_y:.2f}), {actual_text}.'
            )
        else:
            self.get_logger().warn(
                f'Waypoint {self.next_waypoint_index + 1}/{len(self.waypoints)} did not succeed '
                f'(status {status}): aimed at ({target_x:.2f}, {target_y:.2f}), {actual_text}. '
                'Moving on to the next one.'
            )

        self.next_waypoint_index += 1

        self.goal_active = False
        self.goal_handle = None
        self._schedule_next_attempt()

    def _schedule_next_attempt(self) -> None:
        pause = float(self.get_parameter('goal_pause_seconds').value)
        self.next_goal_time = time.monotonic() + pause


def main(args=None):
    rclpy.init(args=args)
    node = SearchManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # On Ctrl-C the launch system may already have shut the context down.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
