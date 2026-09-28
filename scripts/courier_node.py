#!/usr/bin/env python3


"Wait for the scout drone to finish searching, then fly to one delivery point, using NAV2"

import rclpy
from action_msgs.msg import GoalStatus
# Point is a plain (x, y, z) location. Markers are built out of lists of these.
from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
# Duration is used to put a time limit on TF lookups so they can't block.
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
# ColorRGBA lets us colour the target marker.
# Empty carries no data - it is just a "this happened" signal from the scout.
from std_msgs.msg import ColorRGBA, Empty
# tf2_ros answers "where is the drone right now?" by reading the transform tree.
import tf2_ros
# Marker = one drawing in RViz. MarkerArray = several sent together.
from visualization_msgs.msg import Marker, MarkerArray


class CourierNode(Node):
    def __init__(self):
        super().__init__('courier_node')

        # Things that can be changed without editing the code
        self.declare_parameter('robot_name', 'parrot2')
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', -6.0)
        # Absolute (leading "/") because the scout is a different robot, in a
        # different namespace, to this node.
        self.declare_parameter('scout_complete_topic', '/parrot1/search_node/sweep_complete')

        robot_name = self.get_parameter('robot_name').value
        # Goals are stamped in the frame Nav2 plans in. That is now the odom
        # frame, not the SLAM map frame - see global_frame in the nav2 params.
        # Must stay in step with those, or Nav2 will not understand our goals.
        self.map_frame = f'{robot_name}_odom'  # e.g. "parrot2_odom", NOT "odom"

        self.target = (
            self.get_parameter('target_x').value,
            self.get_parameter('target_y').value,
        )

        self.scout_ready = False  # set True once the scout's signal arrives
        self.sent = False         # set True once the delivery goal has been sent
        self.busy = False

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Same latched QoS the scout publishes with, so this still gets the
        # message even if it starts listening after the scout already sent it.
        latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.scout_sub = self.create_subscription(
            Empty,
            self.get_parameter('scout_complete_topic').value,
            self.on_scout_complete,
            latched_qos,
        )

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

    def on_scout_complete(self, _msg):
        """Called once, when the scout's sweep-complete message arrives."""
        if not self.scout_ready:
            self.get_logger().info('Scout has finished searching. Ready to fly to the delivery point.')
        self.scout_ready = True

    def tick(self):
        """Called every second; sends the delivery goal once, once the scout is done."""
        if self.sent:  # already sent (or in flight/finished) - nothing more to do
            return
        if not self.scout_ready:
            self.get_logger().info('Waiting for the scout drone to finish searching...')
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

        # Stage 3: ask to be told when its finished flying.
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_finished)

    def goal_finished(self, future):
        """Stage 3: Nav2 has arrived, failed or given up"""
        if future.result().status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Delivered. ')
        else:
            self.get_logger().warn('Could not reach the delivery point.')
        self.busy = False

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
        if self.sent and not self.busy:
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
