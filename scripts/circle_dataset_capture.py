#!/usr/bin/env python3
"""Fly the drone in a tight circle, saving camera frames to disk.

Meant for gathering raw footage to build a YOLO training set: the drone holds
a constant forward speed and yaw rate (which traces a circle of radius ~=
linear_speed / angular_speed) at whatever altitude it spawns at - this Parrot
model only flies in 2D (see README's "Stage 2: Launch the Parrot drone"), so
there is no Z control to climb with. A frame from the RGB camera is saved to
output_dir every capture_period seconds.

Example (parrot1, matching how the other demo scripts are launched):

    ros2 run 41068_ignition_bringup circle_dataset_capture.py --ros-args \\
        -r __ns:=/parrot1 \\
        -p robot_name:=parrot1 \\
        -p output_dir:=/home/student/yolo_dataset
"""

import os
from typing import Optional

import cv2
from cv_bridge import CvBridge

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image


class CircleDatasetCapture(Node):
    """Circle in place while periodically saving camera frames."""

    def __init__(self):
        super().__init__('circle_dataset_capture')

        self.declare_parameter('robot_name', 'parrot1')
        self.declare_parameter('image_topic', 'camera/image')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('capture_period', 2.0)
        self.declare_parameter('output_dir', os.path.expanduser('~/yolo_dataset'))
        self.declare_parameter('control_rate', 20.0)

        self.robot_name = str(self.get_parameter('robot_name').value)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.angular_speed = float(self.get_parameter('angular_speed').value)
        self.capture_period = float(self.get_parameter('capture_period').value)
        self.output_dir = os.path.expanduser(str(self.get_parameter('output_dir').value))
        control_rate = float(self.get_parameter('control_rate').value)

        os.makedirs(self.output_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.latest_image: Optional[Image] = None
        self.image_count = 0

        self.cmd_vel_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').value, 10
        )
        self.image_sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._image_callback, 1
        )

        self.control_timer = self.create_timer(1.0 / control_rate, self._control_tick)
        self.capture_timer = self.create_timer(self.capture_period, self._capture_tick)

        radius = self.linear_speed / self.angular_speed
        self.get_logger().info(
            f'circle_dataset_capture started for "{self.robot_name}": '
            f'circling at radius ~{radius:.2f} m, saving to {self.output_dir} '
            f'every {self.capture_period:.1f}s.'
        )

    def _image_callback(self, msg: Image) -> None:
        self.latest_image = msg

    def _control_tick(self) -> None:
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = self.angular_speed
        self.cmd_vel_pub.publish(twist)

    def _capture_tick(self) -> None:
        if self.latest_image is None:
            return
        cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
        filename = os.path.join(
            self.output_dir, f'{self.robot_name}_{self.image_count:05d}.png'
        )
        cv2.imwrite(filename, cv_image)
        self.get_logger().info(f'Saved {filename}')
        self.image_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = CircleDatasetCapture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_vel_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
