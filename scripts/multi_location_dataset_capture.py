#!/usr/bin/env python3
"""Teleport person1 and the drone around a set of locations, saving frames.

Alternative to circle_dataset_capture.py's continuous orbit: instead of
flying a smooth loop (which costs time, and risks flight-control issues -
see git history of that script for the debugging involved), this script
uses Ignition's /world/<world>/set_pose service to instantly teleport both
person1 and the drone. For each location in the `locations` parameter, it:

  1. Teleports person1 to that (x, y), standing upright (person_z, identity
     orientation - matches the fix in worlds/large_demo.sdf; this script
     does not rotate the person, only repositions it).
  2. Teleports the drone through `photos_per_location` evenly-spaced points
     on a circle of `orbit_radius` around that location, at a constant
     `altitude`, facing toward person1 at each point, saving one camera
     frame per point.
  3. Moves on to the next location.

Frames from location index i go into <output_dir>/location_{i:02d}/, so
"location 1 will have 15 photos, location 2 etc." as requested - each
folder is a separate, self-contained batch from one person1 position.
Numbering continues after whatever location_NN folders already exist in
output_dir, so running this again with a new `locations` list appends new
folders rather than overwriting an earlier run's location_00, _01, etc.

This intentionally does NOT publish any cmd_vel - the drone is teleported,
not flown. Don't run this at the same time as circle_dataset_capture.py or
any nav2/teleop node that might also be publishing cmd_vel; a zero Twist is
published once at startup and after every teleport purely as a safeguard
against a stray leftover command from an earlier run still being held by
the VelocityControl plugin.

Verified against the running sim before writing this (2026-09-30): the
service is `/world/large_demo/set_pose`, request type `ignition.msgs.Pose`,
response `ignition.msgs.Boolean` - confirmed with a live teleport of both
person1 (static) and parrot1 (dynamic) via the `ign service` CLI. Called
here the same way, via subprocess, rather than pulling in ignition-transport
Python bindings.

Example - current defaults run the "Bulk Human Dataset" plan (20 random
locations, 10 photos each) into that folder; override `-p locations:=...`,
`-p photos_per_location:=...` and `-p output_dir:=...` for anything else:

    ros2 run 41068_ignition_bringup multi_location_dataset_capture.py --ros-args \\
        -r __ns:=/parrot1 \\
        -p robot_name:=parrot1
"""

import math
import os
import re
import subprocess
import time
from typing import Optional

import cv2
from cv_bridge import CvBridge

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image


def _yaw_to_quaternion_zw(yaw: float):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class MultiLocationDatasetCapture(Node):
    """Teleport person1 + drone through a list of locations, capturing frames."""

    def __init__(self):
        super().__init__('multi_location_dataset_capture')

        self.declare_parameter('robot_name', 'parrot1')
        self.declare_parameter('image_topic', 'camera/image')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('world_name', 'large_demo')
        self.declare_parameter('person_model_name', 'person1')

        self.declare_parameter('altitude', 10.0)
        # Empirically calibrated (2026-09-30) by teleporting the drone and
        # visually checking saved frames at R=1.0/1.8/2.5/3.2/4.5 m: 4.5 (an
        # earlier estimate based on FOV math alone, never actually rendered)
        # pushes the person completely out of frame; 3.2 already risks
        # clipping them at the edge; 2.5 reliably keeps them in-frame while
        # still solidly off-center/toward a corner. Re-verify visually if
        # you change altitude or camera FOV again.
        self.declare_parameter('orbit_radius', 2.5)
        self.declare_parameter('person_z', 0.505)
        # "Bulk Human Dataset" plan (2026-10-01): 20 randomly-generated
        # locations, 10 photos each, rather than the usual hand-picked
        # tree-adjacent clearings. Generated with a seeded RNG uniformly
        # over roughly the mapped extent of blue_mountains (x in [-26,30],
        # y in [-10,24]) and filtered to each sit within 2-12m of some real
        # terrain geometry, so none of them land off the edge of the map
        # in empty void the way location_10 did in the previous batch (see
        # that run's "near map edge" caveat) - but otherwise not curated
        # the way earlier clearings were (no per-point tree-adjacency
        # check), since randomness was the point here.
        self.declare_parameter('photos_per_location', 10)
        self.declare_parameter(
            'locations',
            [
                9.81, -9.15, -2.37, -8.99, -24.51, -3.24, 27.6, 1.44, -20.81, -6.71,
                13.46, -8.44, -13.24, -0.16, -21.53, -2.09, -20.34, -0.55, -5.27, -2.88,
                -13.17, -8.91, 23.08, 0.7, 25.21, 5.6, 5.44, -1.07, -3.63, -2.54,
                29.86, 7.32, -20.91, -8.4, -19.86, 11.33, 18.36, 4.35, -22.44, 2.98,
            ],
        )

        self.declare_parameter(
            'output_dir',
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'YoloTraining',
                'Bulk Human Dataset',
            ),
        )
        # How long to discard incoming frames after every teleport before
        # starting to wait for the first genuinely-fresh one, then how long
        # to wait for that fresh frame before giving up and saving whatever
        # is latest anyway.
        self.declare_parameter('frame_settle_time', 0.5)
        self.declare_parameter('frame_wait_timeout', 3.0)

        self.robot_name = str(self.get_parameter('robot_name').value)
        self.world_name = str(self.get_parameter('world_name').value)
        self.person_model_name = str(self.get_parameter('person_model_name').value)
        self.altitude = float(self.get_parameter('altitude').value)
        self.orbit_radius = float(self.get_parameter('orbit_radius').value)
        self.person_z = float(self.get_parameter('person_z').value)
        self.photos_per_location = int(self.get_parameter('photos_per_location').value)
        self.frame_settle_time = float(self.get_parameter('frame_settle_time').value)
        self.frame_wait_timeout = float(self.get_parameter('frame_wait_timeout').value)
        self.output_dir = os.path.expanduser(str(self.get_parameter('output_dir').value))

        locations_flat = list(self.get_parameter('locations').value)
        if len(locations_flat) % 2 != 0:
            raise ValueError('locations parameter must be a flat [x0,y0,x1,y1,...] list')
        self.locations = [
            (locations_flat[i], locations_flat[i + 1])
            for i in range(0, len(locations_flat), 2)
        ]

        os.makedirs(self.output_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.latest_image: Optional[Image] = None

        self.cmd_vel_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').value, 10
        )
        self.image_sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._image_callback, 1
        )

        self.get_logger().info(
            f'multi_location_dataset_capture started for "{self.robot_name}": '
            f'{len(self.locations)} location(s), {self.photos_per_location} photos each, '
            f'orbit_radius={self.orbit_radius:.2f} m, altitude={self.altitude:.2f} m, '
            f'saving to {self.output_dir}.'
        )

    def _image_callback(self, msg: Image) -> None:
        self.latest_image = msg

    def _set_pose(self, name: str, x: float, y: float, z: float, yaw: float) -> bool:
        qz, qw = _yaw_to_quaternion_zw(yaw)
        req = (
            f'name: "{name}", '
            f'position: {{x: {x!r}, y: {y!r}, z: {z!r}}}, '
            f'orientation: {{x: 0, y: 0, z: {qz!r}, w: {qw!r}}}'
        )
        # Retried: under sustained load the server can occasionally take
        # longer than the client-side timeout to reply - Ignition logs that
        # server-side as "NodeShared::RecvSrvRequest() error sending
        # response: Host unreachable" once our client has already given up
        # and exited. Silently moving on when that happens leaves the
        # entity sitting wherever it was before, so the next frame gets
        # captured with it in the wrong place (verified live 2026-09-30
        # against multi_entity_dataset_capture.py's copy of this method).
        last_result = None
        for attempt in range(4):
            last_result = subprocess.run(
                [
                    'ign', 'service', '-s', f'/world/{self.world_name}/set_pose',
                    '--reqtype', 'ignition.msgs.Pose',
                    '--reptype', 'ignition.msgs.Boolean',
                    '--timeout', '4000',
                    '--req', req,
                ],
                capture_output=True, text=True, timeout=8.0,
            )
            if last_result.returncode == 0 and 'true' in last_result.stdout:
                if attempt > 0:
                    self.get_logger().info(f'set_pose for "{name}" succeeded on retry {attempt + 1}.')
                return True
            time.sleep(0.3 * (attempt + 1))
        self.get_logger().error(
            f'set_pose for "{name}" failed after 4 attempts - it is almost certainly '
            f'NOT at the intended position, so frames captured now may be missing it: '
            f'rc={last_result.returncode} stdout={last_result.stdout!r} stderr={last_result.stderr!r}'
        )
        return False

    def _wait_for_fresh_frame(self) -> None:
        # Discard whatever arrives during the settle window (this may
        # include stale, pre-teleport frames still in flight - comparing
        # arrival *timestamps* to detect "new" doesn't catch this, since a
        # message queued before the teleport can still be delivered after
        # it), then wait for the *next* arrival after that, which is
        # guaranteed to postdate the settle window.
        self.latest_image = None
        settle_deadline = time.monotonic() + self.frame_settle_time
        while rclpy.ok() and time.monotonic() < settle_deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.latest_image = None

        deadline = time.monotonic() + self.frame_wait_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_image is not None:
                return
        self.get_logger().warning('Timed out waiting for a fresh camera frame; using latest available.')

    def _save_frame(self, location_dir: str, shot_index: int) -> None:
        if self.latest_image is None:
            self.get_logger().warning('No image received yet - skipping this shot.')
            return
        cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
        filename = os.path.join(location_dir, f'{self.robot_name}_{shot_index:05d}.png')
        cv2.imwrite(filename, cv_image)
        self.get_logger().info(f'Saved {filename}')

    def _next_location_index(self) -> int:
        # Continue numbering after whatever location_NN folders already
        # exist in output_dir, rather than always starting at 0 - so a
        # second run with new locations doesn't overwrite an earlier
        # batch's location_00, location_01, etc.
        existing = []
        if os.path.isdir(self.output_dir):
            for entry in os.listdir(self.output_dir):
                m = re.fullmatch(r'location_(\d+)', entry)
                if m:
                    existing.append(int(m.group(1)))
        return max(existing, default=-1) + 1

    def run(self) -> None:
        self.cmd_vel_pub.publish(Twist())

        start_index = self._next_location_index()
        if start_index > 0:
            self.get_logger().info(
                f'Found existing location_00..{start_index - 1:02d} in {self.output_dir} - '
                f'continuing from location_{start_index:02d} instead of overwriting them.'
            )

        for offset, (px, py) in enumerate(self.locations):
            loc_index = start_index + offset
            location_dir = os.path.join(self.output_dir, f'location_{loc_index:02d}')
            os.makedirs(location_dir, exist_ok=True)

            self.get_logger().info(
                f'--- Location {loc_index}: person1 -> ({px:.2f}, {py:.2f}) ---'
            )
            self._set_pose(self.person_model_name, px, py, self.person_z, 0.0)

            for shot_index in range(self.photos_per_location):
                theta = 2.0 * math.pi * shot_index / self.photos_per_location
                drone_x = px + self.orbit_radius * math.cos(theta)
                drone_y = py + self.orbit_radius * math.sin(theta)
                yaw = math.atan2(py - drone_y, px - drone_x)

                self._set_pose(self.robot_name, drone_x, drone_y, self.altitude, yaw)
                self.cmd_vel_pub.publish(Twist())
                self._wait_for_fresh_frame()
                self._save_frame(location_dir, shot_index)

            self.get_logger().info(f'Finished location {loc_index}.')

        self.get_logger().info('All locations complete - shutting down.')


def main(args=None):
    rclpy.init(args=args)
    node = MultiLocationDatasetCapture()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_vel_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
