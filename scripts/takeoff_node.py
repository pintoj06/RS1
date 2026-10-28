#!/usr/bin/env python3


"Hold a drone on its landing pad on the ute until the operator presses LAUNCH, then climb it to flight altitude and exit"

import rclpy
# Twist is a velocity command; only linear.z is used here.
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool


class TakeoffNode(Node):
    def __init__(self):
        super().__init__('takeoff_node')

        # Height to climb to, in metres above the Gazebo world origin (the
        # ground around the ute is at z = 0). Nav2 is a 2D planner and never
        # commands height, so the drone holds this altitude for the whole
        # flight once this node exits. Keep it in step with search_node.py's
        # flight_altitude for the scout.
        self.declare_parameter('target_altitude', 10.0)       # metres
        self.declare_parameter('max_climb_speed', 1.5)        # m/s
        # Climb speed per metre still to go. Slows the drone down over the
        # last metre or so instead of overshooting the target height.
        self.declare_parameter('gain', 1.5)
        self.declare_parameter('tolerance', 0.05)             # metres
        # Give up after this long (sim time, counted from the launch command)
        # so the launch file can still start Nav2 rather than waiting forever
        # on a stuck drone.
        self.declare_parameter('timeout', 60.0)               # seconds
        # True: sit on the pad until a True arrives on launch_topic (the
        # ground station's LAUNCH button). False: climb straight away.
        # Absolute topic, because one button launches every drone.
        self.declare_parameter('wait_for_launch', True)
        self.declare_parameter('launch_topic', '/aeroaid/launch')

        self.target = float(self.get_parameter('target_altitude').value)
        self.max_speed = float(self.get_parameter('max_climb_speed').value)
        self.gain = float(self.get_parameter('gain').value)
        self.tolerance = float(self.get_parameter('tolerance').value)
        self.timeout = float(self.get_parameter('timeout').value)

        self.altitude = None   # latest z from odometry, once one arrives
        self.launched = not self.get_parameter('wait_for_launch').value
        self.start_time = None
        self.done = False
        self.succeeded = False

        # The Gazebo VelocityControl plugin moves the drone directly from
        # cmd_vel, with gravity off, so a z velocity is all a climb needs.
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        # Raw Gazebo odometry, not the EKF output: the EKF does not fuse z,
        # but the odometry publisher runs in 3D (see parrot.gazebo.xacro), so
        # its z is the drone's height in the world.
        self.create_subscription(Odometry, 'odometry', self.on_odometry, 10)

        # Transient local, same as the ground station's publisher, so a press
        # made before this node was listening is still delivered.
        launch_qos = QoSProfile(depth=1,
                                reliability=QoSReliabilityPolicy.RELIABLE,
                                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, self.get_parameter('launch_topic').value,
                                 self.on_launch, launch_qos)

        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz

        if self.launched:
            self.get_logger().info(f'Climbing to {self.target:.1f} m.')
        else:
            self.get_logger().info(
                f'On the pad. Waiting for LAUNCH from the ground station, '
                f'then climbing to {self.target:.1f} m.')

    def on_odometry(self, msg):
        self.altitude = msg.pose.pose.position.z

    def on_launch(self, msg):
        if msg.data and not self.launched:
            self.launched = True
            self.get_logger().info('LAUNCH received.')

    def tick(self):
        # Sending nothing while on the pad is enough to keep the drone there:
        # VelocityControl holds it at zero velocity until told otherwise.
        if self.done or not self.launched or self.altitude is None:
            return

        now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = now
            self.get_logger().info(f'Taking off from {self.altitude:.2f} m.')

        error = self.target - self.altitude
        if abs(error) <= self.tolerance:
            self.finish(True)
            return

        elapsed = (now - self.start_time).nanoseconds / 1e9
        if elapsed > self.timeout:
            self.get_logger().error(
                f'Takeoff timed out after {elapsed:.0f} s at {self.altitude:.2f} m '
                f'(target {self.target:.1f} m).')
            self.finish(False)
            return

        cmd = Twist()
        cmd.linear.z = max(-self.max_speed, min(self.max_speed, self.gain * error))
        self.cmd_pub.publish(cmd)

    def finish(self, succeeded):
        # VelocityControl keeps applying the last command it got, so the
        # drone has to be told to stop or it would keep climbing.
        stop = Twist()
        for _ in range(5):
            self.cmd_pub.publish(stop)
        if succeeded:
            self.get_logger().info(f'Reached {self.altitude:.2f} m, hovering.')
        self.succeeded = succeeded
        self.done = True


def main():
    rclpy.init()
    node = TakeoffNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    succeeded = node.succeeded
    node.destroy_node()
    rclpy.try_shutdown()
    # The launch file starts Nav2 when this process exits, whatever the
    # exit code; a non-zero code just makes a failed takeoff easy to spot.
    raise SystemExit(0 if succeeded else 1)


if __name__ == '__main__':
    main()
