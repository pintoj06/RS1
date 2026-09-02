#!/usr/bin/env python3
"""Very small map/image/Nav2 autonomy example.

This node is intended as starter code for students. It deliberately keeps the
"autonomy" simple so the ROS structure is easy to follow:

  1. Read the current SLAM/Nav2 map (nav_msgs/OccupancyGrid).
  2. Read the latest camera image and compute one cheap brightness score.
  3. Look up the robot pose with TF.
  4. Pick a random free-space waypoint from the map.
  5. Send the waypoint to Nav2's NavigateToPose action server.
  6. Wait for the result, then choose another waypoint.

The same script works for /husky1 and /parrot1 when launched in the matching
namespace. Topic and action names are intentionally relative. For example, the
relative topic ``map`` becomes ``/husky1/map`` when the node is launched in the
``husky1`` namespace, and ``/parrot1/map`` when launched in ``parrot1``.

This is not meant to be a good exploration algorithm. It is meant to be a small,
readable template showing how perception, mapping, TF, and Nav2 can be connected
inside one ROS node.
"""

import math
import random
import time
from typing import Optional, Tuple

import numpy as np

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image
import tf2_ros


class BasicAutonomyDemo(Node):
    """A tiny map-aware random-walk autonomy example."""

    def __init__(self):
        super().__init__('basic_autonomy_demo')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        # Keep topic/action names relative. The launch file places this node in
        # /husky1 or /parrot1, so these automatically become /husky1/map,
        # /parrot1/map, /husky1/navigate_to_pose, etc.
        self.declare_parameter('robot_name', '')
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('image_topic', 'camera/image')
        self.declare_parameter('navigate_action', 'navigate_to_pose')

        # Bright images make the demo choose goals farther away; darker images
        # make it choose closer goals. This is intentionally simple, and is just
        # a placeholder for a more meaningful perception signal.
        self.declare_parameter('close_goal_min_distance', 4.0)
        self.declare_parameter('close_goal_max_distance', 7.0)
        self.declare_parameter('far_goal_min_distance', 7.0)
        self.declare_parameter('far_goal_max_distance', 15.0)
        self.declare_parameter('bright_image_threshold', 0.25)

        # OccupancyGrid cells are normally -1 unknown, 0 free, and 100 occupied.
        # We accept cells up to free_cell_threshold, and reject candidates close
        # to occupied/unknown cells using occupied_margin_cells.
        self.declare_parameter('free_cell_threshold', 20)
        self.declare_parameter('occupied_margin_cells', 4)

        # Pause briefly between goals so the terminal output is readable and so
        # Nav2 has time to settle after each result.
        self.declare_parameter('goal_pause_seconds', 1.0)

        # ------------------------------------------------------------------
        # Namespace and frame setup
        # ------------------------------------------------------------------
        self.robot_name = self._resolve_robot_name()
        self.map_frame = f'{self.robot_name}_map'
        self.base_frame = f'{self.robot_name}_base_link'

        # Latest sensor/map data. Callbacks update these; the timer uses them.
        self.map_msg: Optional[OccupancyGrid] = None
        self.map_array: Optional[np.ndarray] = None
        self.latest_brightness: Optional[float] = None
        self.last_image_process_time = 0.0

        # Nav2 action state. goal_active is True while Nav2 is executing a goal.
        self.goal_active = False
        self.goal_handle = None
        self.next_goal_time = 0.0
        self.last_feedback_log_time = 0.0
        self.last_waiting_log_time = 0.0

        # TF is used to find the robot pose in the map frame. The launch file
        # remaps /tf to the namespaced tf topic so each robot has its own TF tree.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.get_parameter('map_topic').value,
            self._map_callback,
            1,
        )
        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self._image_callback,
            1,
        )
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            self.get_parameter('navigate_action').value,
        )

        # A timer drives the simple autonomy state machine. This avoids doing
        # slow work inside subscription callbacks.
        self.timer = self.create_timer(1.0, self._tick)

        self.get_logger().info(
            f'Autonomy demo started for robot "{self.robot_name}". '
            f'Using map frame "{self.map_frame}" and base frame "{self.base_frame}".'
        )

    def _resolve_robot_name(self) -> str:
        """Return the robot name used to construct frame IDs.

        The launch file passes robot_name explicitly. If students remove that
        parameter, the node can still infer the robot name from its namespace.
        """
        configured_name = str(self.get_parameter('robot_name').value).strip().strip('/')
        if configured_name:
            return configured_name

        namespace = self.get_namespace().strip('/')
        if namespace:
            return namespace

        # This fallback is mainly for debugging if the node is launched without
        # a namespace. The normal launch file passes /husky1 or /parrot1.
        return 'husky1'

    def _map_callback(self, msg: OccupancyGrid) -> None:
        """Store the latest map as both a ROS message and a 2D NumPy array."""
        self.map_msg = msg

        # OccupancyGrid.data is a flat row-major list. Reshape it so that
        # map_array[iy, ix] gives the cell value at grid coordinates (ix, iy).
        self.map_array = np.asarray(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width)
        )

    def _image_callback(self, msg: Image) -> None:
        """Compute a tiny image feature from the latest camera image.

        The node receives camera images at the simulator's camera rate, but this
        demo only processes at 1 Hz. That keeps the example lightweight and
        avoids image processing slowing down navigation.
        """
        now = time.monotonic()
        if now - self.last_image_process_time < 1.0:
            return
        self.last_image_process_time = now

        brightness = self._estimate_image_brightness(msg)
        if brightness is not None:
            self.latest_brightness = brightness

    def _estimate_image_brightness(self, msg: Image) -> Optional[float]:
        """Return a fast approximate image brightness between 0.0 and 1.0.

        This deliberately avoids cv_bridge/OpenCV to keep dependencies minimal.
        It samples only a subset of pixels, so the computation stays fast even if
        the camera image is large. Students can replace this function with more
        meaningful perception later.
        """
        encoding = msg.encoding.lower()
        channels = None
        channel_order = None

        if encoding in ('rgb8', 'bgr8'):
            channels = 3
            channel_order = encoding
        elif encoding in ('rgba8', 'bgra8'):
            channels = 4
            channel_order = encoding
        elif encoding in ('mono8', '8uc1'):
            channels = 1
            channel_order = encoding
        else:
            self.get_logger().warn(
                f'Unsupported image encoding "{msg.encoding}". Brightness will not be updated.',
                throttle_duration_sec=10.0,
            )
            return None

        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            rows = data.reshape((msg.height, msg.step))
        except ValueError:
            self.get_logger().warn('Could not reshape image data. Brightness will not be updated.')
            return None

        # Sample roughly 80 pixels across the shorter image dimension. The exact
        # number is not important; this is only a cheap behaviour signal.
        stride = max(1, min(msg.width, msg.height) // 80)

        if channels == 1:
            sampled = rows[::stride, :msg.width:stride]
            return float(np.mean(sampled)) / 255.0

        useful_cols = msg.width * channels
        sampled = rows[::stride, :useful_cols].reshape((-1, msg.width, channels))[:, ::stride, :]

        # Average RGB/BGR channels only; ignore alpha if present.
        if channel_order.startswith('rgb'):
            rgb = sampled[..., :3]
        elif channel_order.startswith('bgr'):
            rgb = sampled[..., :3][..., ::-1]
        else:
            rgb = sampled[..., :3]

        return float(np.mean(rgb)) / 255.0

    def _tick(self) -> None:
        """One step of the autonomy state machine."""
        if self.goal_active:
            return

        if time.monotonic() < self.next_goal_time:
            return

        # Wait for the Nav2 action server. This is the action endpoint that RViz
        # also uses when you click a Nav2 goal.
        if not self.nav_client.server_is_ready():
            if not self.nav_client.wait_for_server(timeout_sec=0.1):
                self._log_waiting('Waiting for Nav2 NavigateToPose action server...')
                return

        if self.map_msg is None or self.map_array is None:
            self._log_waiting('Waiting for an OccupancyGrid map...')
            return

        robot_pose = self._lookup_robot_pose()
        if robot_pose is None:
            self._log_waiting(f'Waiting for TF from {self.map_frame} to {self.base_frame}...')
            return

        goal_xy = self._choose_goal(robot_pose)
        if goal_xy is None:
            self._log_waiting('No suitable free-space waypoint found in the current map yet...')
            return

        self._send_goal(goal_xy)

    def _log_waiting(self, message: str) -> None:
        """Log repeated waiting messages at low rate."""
        now = time.monotonic()
        if now - self.last_waiting_log_time > 5.0:
            self.last_waiting_log_time = now
            self.get_logger().info(message)

    def _lookup_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        """Look up the robot pose in the robot's map frame using TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
        except Exception as exc:  # TF exceptions are numerous; keep this compact for students.
            self.get_logger().debug(f'TF lookup failed: {exc}')
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = self._yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
        return (translation.x, translation.y, yaw)

    def _choose_goal(self, robot_pose: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Choose the next random goal from known free space.

        The only role of the image in this demo is to bias goal distance:
        brighter images choose farther goals, darker images choose closer goals.
        This is intentionally artificial but easy to replace.
        """
        assert self.map_msg is not None
        assert self.map_array is not None

        brightness = self.latest_brightness
        if brightness is None:
            # Neutral default until the first camera image arrives.
            brightness = 0.5

        bright_threshold = float(self.get_parameter('bright_image_threshold').value)
        if brightness >= bright_threshold:
            min_distance = float(self.get_parameter('far_goal_min_distance').value)
            max_distance = float(self.get_parameter('far_goal_max_distance').value)
            mode = 'farther'
        else:
            min_distance = float(self.get_parameter('close_goal_min_distance').value)
            max_distance = float(self.get_parameter('close_goal_max_distance').value)
            mode = 'closer'

        robot_x, robot_y, _ = robot_pose

        # First try with the brightness-selected distance band.
        result = self._sample_free_goal(robot_x, robot_y, min_distance, max_distance, max_tries=1500)

        # If that fails, relax the distance filter. This makes the demo more
        # robust while the map is still small or mostly unknown.
        if result is None:
            result = self._sample_free_goal(robot_x, robot_y, 1.0, max(max_distance, 8.0), max_tries=1500)
            mode = 'fallback'

        if result is not None:
            goal_x, goal_y = result
            goal_yaw = random.uniform(-math.pi, math.pi)
            self.get_logger().info(
                f'Brightness={brightness:.2f}; choosing a {mode} free-space goal '
                f'at ({goal_x:.2f}, {goal_y:.2f}).'
            )
            return (goal_x, goal_y, goal_yaw)

        return None

    def _sample_free_goal(
        self,
        robot_x: float,
        robot_y: float,
        min_distance: float,
        max_distance: float,
        max_tries: int,
    ) -> Optional[Tuple[float, float]]:
        """Randomly sample map cells until a suitable free-space cell is found."""
        assert self.map_msg is not None
        assert self.map_array is not None

        width = self.map_msg.info.width
        height = self.map_msg.info.height
        margin = int(self.get_parameter('occupied_margin_cells').value)

        if width <= 2 * margin or height <= 2 * margin:
            return None

        for _ in range(max_tries):
            ix = random.randint(margin, width - margin - 1)
            iy = random.randint(margin, height - margin - 1)
            if not self._is_free_with_margin(ix, iy, margin):
                continue

            wx, wy = self._map_cell_to_world(ix, iy)
            distance = math.hypot(wx - robot_x, wy - robot_y)
            if min_distance <= distance <= max_distance:
                return (wx, wy)

        return None

    def _is_free_with_margin(self, ix: int, iy: int, margin: int) -> bool:
        """Check that a candidate cell and nearby cells are known free space."""
        assert self.map_array is not None
        free_threshold = int(self.get_parameter('free_cell_threshold').value)

        window = self.map_array[iy - margin:iy + margin + 1, ix - margin:ix + margin + 1]

        # OccupancyGrid: -1 = unknown, 0 = free, 100 = occupied.
        # This skeleton deliberately only chooses known free space.
        if np.any(window < 0):
            return False
        if np.any(window > free_threshold):
            return False
        return True

    def _map_cell_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """Convert OccupancyGrid cell coordinates into map-frame metres."""
        assert self.map_msg is not None
        info = self.map_msg.info
        resolution = info.resolution
        origin = info.origin

        # The map origin is the pose of grid cell (0, 0) in the map frame. The
        # +0.5 uses the centre of the grid cell rather than the lower-left edge.
        local_x = (ix + 0.5) * resolution
        local_y = (iy + 0.5) * resolution
        origin_yaw = self._yaw_from_quaternion(
            origin.orientation.x,
            origin.orientation.y,
            origin.orientation.z,
            origin.orientation.w,
        )

        # Most 2D maps have zero origin yaw, but this handles rotated map origins
        # too. It is a useful pattern when converting between grid and world.
        cos_yaw = math.cos(origin_yaw)
        sin_yaw = math.sin(origin_yaw)
        world_x = origin.position.x + cos_yaw * local_x - sin_yaw * local_y
        world_y = origin.position.y + sin_yaw * local_x + cos_yaw * local_y
        return (world_x, world_y)

    def _send_goal(self, goal_xy_yaw: Tuple[float, float, float]) -> None:
        """Send one NavigateToPose action goal to Nav2."""
        x, y, yaw = goal_xy_yaw

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = self.map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = self._quaternion_from_yaw(yaw)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.goal_active = True
        self.last_feedback_log_time = 0.0
        self.get_logger().info(f'Sending Nav2 goal: ({x:.2f}, {y:.2f}) in frame {self.map_frame}.')

        # Nav2 action use happens in three stages:
        #   1. send the goal request;
        #   2. wait for Nav2 to accept/reject the goal;
        #   3. if accepted, wait for the final result.
        send_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback,
        )
        send_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future) -> None:
        """Called when Nav2 accepts or rejects the goal request."""
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().warn('Nav2 goal was rejected.')
            self.goal_active = False
            self.next_goal_time = time.monotonic() + float(self.get_parameter('goal_pause_seconds').value)
            return

        self.get_logger().info('Nav2 goal accepted. Waiting for result...')
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _feedback_callback(self, feedback_msg) -> None:
        """Called whenever Nav2 sends feedback while driving to the goal."""
        # Feedback can arrive quickly. Log at low rate so students can see it
        # without flooding the terminal.
        now = time.monotonic()
        if now - self.last_feedback_log_time < 5.0:
            return
        self.last_feedback_log_time = now

        feedback = feedback_msg.feedback
        distance = getattr(feedback, 'distance_remaining', None)
        if distance is not None:
            self.get_logger().info(f'Nav2 feedback: distance remaining {distance:.2f} m.')

    def _result_callback(self, future) -> None:
        """Called once Nav2 has finished, failed, or cancelled the goal."""
        wrapped_result = future.result()
        status = wrapped_result.status
        status_text = self._goal_status_to_text(status)

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Nav2 goal finished: {status_text}.')
        else:
            self.get_logger().warn(f'Nav2 goal finished: {status_text}. Choosing another goal soon.')

        self.goal_active = False
        self.goal_handle = None
        self.next_goal_time = time.monotonic() + float(self.get_parameter('goal_pause_seconds').value)

    @staticmethod
    def _goal_status_to_text(status: int) -> str:
        names = {
            GoalStatus.STATUS_UNKNOWN: 'UNKNOWN',
            GoalStatus.STATUS_ACCEPTED: 'ACCEPTED',
            GoalStatus.STATUS_EXECUTING: 'EXECUTING',
            GoalStatus.STATUS_CANCELING: 'CANCELING',
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
            GoalStatus.STATUS_ABORTED: 'ABORTED',
        }
        return names.get(status, f'UNRECOGNISED_STATUS_{status}')

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        """Convert a quaternion into a 2D yaw angle in radians."""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
        """Convert a 2D yaw angle into a quaternion tuple."""
        half_yaw = 0.5 * yaw
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


def main(args=None):
    rclpy.init(args=args)
    node = BasicAutonomyDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
