#!/usr/bin/env python3


"Wait for the operator to confirm a contact on the ground station, then fly to it, using NAV2"

import math

import rclpy
from action_msgs.msg import GoalStatus
# Point is a plain (x, y, z) location. Markers are built out of lists of these.
# PointStamped is what the ground station publishes for a confirmed contact.
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
# The scout's global costmap, relayed into this robot's frame for Nav2.
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
# Duration is used to put a time limit on TF lookups so they can't block.
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
# ColorRGBA lets us colour the target marker.
# Empty is a message with no content: publishing one releases the care package.
# Bool carries the ground station's E-Stop state.
from std_msgs.msg import Bool, ColorRGBA, Empty
# TFMessage is what /<robot>/tf carries; used to read the scout's TF tree.
from tf2_msgs.msg import TFMessage
# tf2_ros answers "where is the drone right now?" by reading the transform tree.
import tf2_ros
# Marker = one drawing in RViz. MarkerArray = several sent together.
from visualization_msgs.msg import Marker, MarkerArray


class CourierNode(Node):
    def __init__(self):
        super().__init__('courier_node')

        # Things that can be changed without editing the code
        self.declare_parameter('robot_name', 'parrot2')
        # Absolute (leading "/") because the ground station runs on the
        # scout's namespace, a different robot to this node.
        self.declare_parameter('confirmed_point_topic', '/parrot1/operator/confirmed_point')
        # The scout's namespace. Its TF tree (/<scout_name>/tf) is read to
        # move a confirmed contact out of the scout's SLAM map frame.
        self.declare_parameter('scout_name', 'parrot1')
        # True: fly to the contact position the scout reported.
        # False: ignore it and fly to the fixed target_world_x/y below.
        self.declare_parameter('use_contact_position', True)
        # Fixed delivery point in Gazebo world coordinates, used only when
        # use_contact_position is False: directly above person1 in
        # worlds/large_demo.sdf. Keep in step with that pose.
        self.declare_parameter('target_world_x', -6.36)
        self.declare_parameter('target_world_y', -3.07)
        # Drop the care package once this close to the contact, rather than
        # waiting for Nav2 to call the goal reached (0.25 m). The package
        # lands within a metre or two of the person either way, and the last
        # stretch is the slow, fiddly one - often next to rock, where Nav2
        # can spend a long time creeping in or give up altogether.
        self.declare_parameter('release_radius', 1.5)   # metres

        robot_name = self.get_parameter('robot_name').value
        scout_name = self.get_parameter('scout_name').value
        self.use_contact_position = self.get_parameter('use_contact_position').value
        # Goals are stamped in the frame Nav2 plans in. That is now the odom
        # frame, not the SLAM map frame - see global_frame in the nav2 params.
        # Must stay in step with those, or Nav2 will not understand our goals.
        self.map_frame = f'{robot_name}_odom'  # e.g. "parrot2_odom", NOT "odom"
        self.scout_odom_frame = f'{scout_name}_odom'

        # Every drone's odom frame is the Gazebo world frame: the odometry
        # plugin reports world pose, so at spawn parrot1 reads (2, 0) and
        # parrot2 reads (2.47, -1.10) - not (0, 0). A world point, or a point in
        # the scout's odom frame, is therefore already in this robot's odom
        # frame with no spawn offset.
        self.delivery_point = (
            self.get_parameter('target_world_x').value,
            self.get_parameter('target_world_y').value,
        )

        # The latest confirmed contact, waiting to be converted into
        # this robot's frame (needs the scout's TF, which may lag).
        self.pending_contact = None

        self.target = None  # (x, y) in this robot's own odom frame, once confirmed
        self.sent = False   # set True once the delivery goal has been sent
        self.busy = False
        self.released = False   # the care package has been dropped
        # E-Stop from the ground station: cancel the flight and hover until
        # RESUME, then fly the same delivery again.
        self.estopped = False
        self.goal_handle = None

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Releases the care package slung under the drone. Bridged to the
        # DetachableJoint in Gazebo (config/gazebo_bridge_parrot2.yaml); the
        # package then falls onto the contact.
        self.release_pub = self.create_publisher(Empty, 'care_package/detach', 10)
        # 10 Hz: at 1 m/s the drone moves 0.1 m between checks, so it is
        # never far inside release_radius when the package goes.
        self.release_timer = self.create_timer(0.1, self.check_release)

        # Same latched QoS the ground station publishes with, so this still
        # gets the message even if it starts listening after a confirm.
        latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.confirmed_point_sub = self.create_subscription(
            PointStamped,
            self.get_parameter('confirmed_point_topic').value,
            self.on_confirmed_point,
            latched_qos,
        )
        # Absolute, and the same latched QoS the ground station publishes it
        # with, so a courier that starts late still sees a STOP already on.
        estop_qos = QoSProfile(depth=1,
                               reliability=QoSReliabilityPolicy.RELIABLE,
                               durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/aeroaid/estop', self.on_estop, estop_qos)

        self.timer = self.create_timer(1.0, self.tick)  # runs once a second

        # ------------------------------------------------------------------
        # VISUALISATION - everything below is only for RViz. It does not
        # affect how the drone flies; it just draws what is going on so the
        # delivery can be seen (and screenshotted) rather than only logged.
        # ------------------------------------------------------------------

        # The drone's own body frame, e.g. "parrot2_base_link". Asking TF for
        # map -> base_link tells us where the drone is on the map.
        self.base_frame = f'{robot_name}_base_link'

        # The buffer stores recent transforms; the listener fills it from the
        # /tf topic in the background. Both are needed for lookups to work.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # A second buffer for the scout's TF tree. Each robot publishes TF on
        # its own /<robot>/tf topic, and the listener above is remapped to
        # this robot's, so the scout's topics are subscribed to directly.
        self.scout_tf_buffer = tf2_ros.Buffer()
        self.create_subscription(
            TFMessage, f'/{scout_name}/tf',
            lambda msg: self.store_scout_tf(msg, static=False), 100)
        self.create_subscription(
            TFMessage, f'/{scout_name}/tf_static',
            lambda msg: self.store_scout_tf(msg, static=True),
            QoSProfile(depth=100, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))

        # The scout's map, for planning round what it has already seen. The
        # courier's own lidar only reaches 10 m, so on its own it plans a
        # straight line to the contact and finds the mountain on arrival.
        # The scout has usually flown past that mountain by then and mapped
        # it. Its global costmap is relayed, frame relabelled, to the topic
        # the static_layer in nav2_params_parrot2*.yaml reads.
        self.declare_parameter('scout_costmap_topic', f'/{scout_name}/global_costmap/costmap')
        # The scout publishes at 5 Hz and Nav2 rebuilds the static layer on
        # every message, so pass on at most one this often. Obstacles do not
        # move; a few seconds' delay on a newly mapped face changes nothing.
        self.declare_parameter('scout_costmap_period', 2.0)   # seconds
        # Both are latched: Nav2 publishes its costmap transient-local, and
        # Nav2's static layer only listens transient-local.
        map_qos = QoSProfile(depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.scout_costmap_pub = self.create_publisher(
            OccupancyGrid, 'scout_costmap', map_qos)
        self._scout_costmap_sent = None
        self.create_subscription(
            OccupancyGrid, self.get_parameter('scout_costmap_topic').value,
            self.relay_scout_costmap, map_qos)

        # Publisher for the blue breadcrumb trail of where the drone has been.
        self.trail_pub = self.create_publisher(Marker, 'courier_trail', 10)

        # One Marker that we keep appending points to, rather than creating a
        # new one each time. Same ns + id every publish means RViz replaces
        # the old drawing instead of stacking copies on top of each other.
        self.trail = Marker()
        self.trail.header.frame_id = self.map_frame  # coordinates are map-frame
        self.trail.ns = 'courier_trail'
        self.trail.id = 0
        self.trail.type = Marker.POINTS   # dots; use LINE_STRIP for a line
        self.trail.action = Marker.ADD
        self.trail.scale.x = 0.2          # POINTS needs BOTH x and y set:
        self.trail.scale.y = 0.2          # they are the dot's width/height
        self.trail.color.r = 0.0
        self.trail.color.g = 0.4
        self.trail.color.b = 1.0          # blue, to tell it apart from the scout's red trail
        self.trail.color.a = 1.0          # opacity - defaults to 0 = INVISIBLE
        self.trail.pose.orientation.w = 1.0  # identity rotation (points are absolute)

        # Publisher for the delivery target marker.
        self.target_pub = self.create_publisher(MarkerArray, 'courier_target', 10)

        # Two extra timers. Redrawing on a timer (instead of publishing once)
        # means the displays appear correctly no matter when you add them in
        # RViz.
        self.trail_timer = self.create_timer(0.5, self.record_trail)
        self.target_timer = self.create_timer(1.0, self.publish_target_marker)

    def store_scout_tf(self, msg, static):
        for t in msg.transforms:
            if static:
                self.scout_tf_buffer.set_transform_static(t, 'scout_tf')
            else:
                self.scout_tf_buffer.set_transform(t, 'scout_tf')

    def relay_scout_costmap(self, msg):
        """Pass the scout's costmap on to this robot's Nav2, in this robot's frame.

        Relabelling the frame is all the conversion there is: every drone's
        odom frame is the Gazebo world frame (see the note in __init__), so a
        cell at (x, y) in parrot1_odom is at the same (x, y) in parrot2_odom.
        Nav2's static layer would otherwise look for a parrot1_odom ->
        parrot2_odom transform, and none exists - each robot has its own TF.
        """
        now = self.get_clock().now()
        period = float(self.get_parameter('scout_costmap_period').value)
        if (self._scout_costmap_sent is not None
                and (now - self._scout_costmap_sent).nanoseconds < period * 1e9):
            return
        if self._scout_costmap_sent is None:
            self.get_logger().info(
                f'Got the scout\'s map ({msg.info.width} x {msg.info.height} '
                f'cells at {msg.info.resolution:.2f} m); planning round what '
                f'it has seen.')
        self._scout_costmap_sent = now
        msg.header.frame_id = self.map_frame
        self.scout_costmap_pub.publish(msg)

    def on_confirmed_point(self, msg):
        """Called when the operator confirms a contact on the ground station.

        The contact is stored and converted in tick(), because the
        conversion needs the scout's TF, which may not have arrived yet.
        """
        if self.target is not None or self.pending_contact is not None:
            return  # already have one - ignore repeat confirms
        self.get_logger().info(
            f'Contact confirmed at ({msg.point.x:.2f}, {msg.point.y:.2f}) '
            f'in {msg.header.frame_id or "(no frame)"}'
        )
        self.pending_contact = msg

    def on_estop(self, msg):
        if msg.data == self.estopped:
            return
        self.estopped = msg.data
        if self.estopped:
            self.get_logger().warn('E-STOP: hovering until the operator presses RESUME.')
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
        else:
            self.get_logger().info('E-Stop released.')
            self.tick()

    def contact_to_own_frame(self, msg):
        """Return the contact as (x, y) in this robot's odom frame, or None if
        the scout's TF isn't available yet.

        The ground station stamps the point in the scout's SLAM map frame.
        map -> odom is SLAM's drift correction, so applying it gives the
        point in the scout's odom frame, which (see __init__) is the world
        frame and so also this robot's odom frame.
        """
        frame = msg.header.frame_id
        if frame == self.scout_odom_frame:
            return (msg.point.x, msg.point.y)
        try:
            t = self.scout_tf_buffer.lookup_transform(
                self.scout_odom_frame, frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().info(
                f'Waiting for scout transform {frame} -> {self.scout_odom_frame}: {e}',
                throttle_duration_sec=5.0)
            return None
        # Drones fly level, so a 2D (yaw-only) transform is enough.
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
        px, py = msg.point.x, msg.point.y
        x = t.transform.translation.x + math.cos(yaw) * px - math.sin(yaw) * py
        y = t.transform.translation.y + math.sin(yaw) * px + math.cos(yaw) * py
        return (x, y)

    def tick(self):
        """Called every second; sends the delivery goal once, once a contact is confirmed."""
        if self.estopped:
            return
        if self.sent:  # already sent (or in flight/finished) - nothing more to do
            return
        if self.target is None and self.pending_contact is not None:
            if self.use_contact_position:
                self.target = self.contact_to_own_frame(self.pending_contact)
            else:
                self.target = self.delivery_point
            if self.target is not None:
                source = 'contact position' if self.use_contact_position else 'fixed point'
                self.get_logger().info(
                    f'Delivery target ({source}): '
                    f'({self.target[0]:.2f}, {self.target[1]:.2f}) in {self.map_frame}'
                )
        if self.target is None:
            if self.pending_contact is not None:
                return  # confirmed, still waiting on the scout's TF

            self.get_logger().info('Waiting for the operator to confirm a contact...')
            return
        if self.busy:  # goal is already in flight
            return
        if not self.nav_client.server_is_ready():  # Nav2 server not up yet
            self.get_logger().info('Waiting for Nav2 to be ready...')
            return
        self.send_goal(self.target)

    def send_goal(self, point):
        """Send the one delivery waypoint to Nav2."""
        x, y = point

        yaw = 0.0

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        # An angle as a quaternion. For a flat 2D turn, only z and w matter.
        goal.pose.pose.orientation.z = 0.0
        goal.pose.pose.orientation.w = 1.0

        self.busy = True
        self.sent = True
        self.get_logger().info(f'Flying to delivery point ({x:.1f}, {y:.1f})')

        # Stage 1: send it. goal_response() gets called when Nav2 answers.
        send_future = self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self.goal_response)

    def goal_response(self, future):
        """Stage 2: Nav2 has accepted or rejected the goal."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected the delivery goal.')
            self.busy = False
            self.sent = False  # allow tick() to retry
            return

        self.goal_handle = goal_handle
        if self.estopped:
            # E-Stop arrived while this goal was still being accepted.
            goal_handle.cancel_goal_async()

        # Stage 3: ask to be told when its finished flying.
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_finished)

    def check_release(self):
        """Drop the package as soon as the drone is within release_radius."""
        if self.released or not self.busy or self.estopped or self.target is None:
            return
        here = self.own_xy()
        if here is None:
            return
        dist = math.hypot(self.target[0] - here[0], self.target[1] - here[1])
        if dist > float(self.get_parameter('release_radius').value):
            return
        self.release(f'{dist:.1f} m from the contact')
        # Hover where it dropped instead of creeping the last metre in.
        # goal_finished sees self.released and does not count this as a miss.
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()

    def release(self, where):
        self.released = True
        self.release_pub.publish(Empty())
        self.get_logger().info(f'Care package released, {where}.')

    def own_xy(self):
        """This drone's (x, y) in its odom frame, or None before TF arrives."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y

    def goal_finished(self, future):
        """Stage 3: Nav2 has arrived, failed or given up"""
        self.goal_handle = None
        self.busy = False
        if self.released:
            # Dropped on the way in (check_release) and the goal cancelled,
            # or arrived just after. Either way the delivery is done.
            self.get_logger().info('Delivery complete. Hovering over the contact.')
            return
        if self.estopped:
            # Cancelled by the E-Stop: not delivered, so fly it again on
            # RESUME instead of giving up. The package stays on board.
            self.sent = False
            return
        if future.result().status == GoalStatus.STATUS_SUCCEEDED:
            # Only if the 10 Hz check has not already: Nav2's goal tolerance
            # is tighter than release_radius, so this is a backstop.
            self.release('on arrival')
        else:
            self.get_logger().warn('Could not reach the delivery point.')

    # ----------------------------------------------------------------------
    # VISUALISATION METHODS
    # ----------------------------------------------------------------------

    def record_trail(self):
        """Add the drone's current position to the blue trail.

        Runs twice a second. Each call asks TF where the drone is and appends
        that position to a growing list of dots, so afterwards you can see
        exactly where it actually flew.
        """
        try:
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

        p = Point()
        p.x = t.transform.translation.x
        p.y = t.transform.translation.y
        p.z = 0.0  # draw flat on the ground
        self.trail.points.append(p)

        if len(self.trail.points) > 2000:
            self.trail.points.pop(0)

        self.trail.header.stamp = self.get_clock().now().to_msg()
        self.trail_pub.publish(self.trail)

    def publish_target_marker(self):
        """Draw the single delivery point, coloured by whether it's been sent yet."""
        if self.target is None:  # nothing confirmed yet - nothing to draw
            return
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        x, y = self.target

        dot = Marker()
        dot.header.frame_id = self.map_frame
        dot.header.stamp = stamp
        dot.ns = 'delivery_target'
        dot.id = 0
        dot.type = Marker.SPHERE
        dot.action = Marker.ADD
        dot.scale.x = dot.scale.y = dot.scale.z = 0.5
        dot.pose.position.x = float(x)
        dot.pose.position.y = float(y)
        dot.pose.position.z = 0.2
        dot.pose.orientation.w = 1.0
        # Yellow while waiting for the scout / flying there, green once delivered.
        if self.released:
            dot.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        else:
            dot.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        markers.markers.append(dot)

        label = Marker()
        label.header.frame_id = self.map_frame
        label.header.stamp = stamp
        label.ns = 'delivery_target_label'
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(x)
        label.pose.position.y = float(y)
        label.pose.position.z = 0.9
        label.scale.z = 0.5
        label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        label.text = 'Delivery target'
        markers.markers.append(label)

        self.target_pub.publish(markers)


def main():
    rclpy.init()
    node = CourierNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
