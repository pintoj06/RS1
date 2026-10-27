#!/usr/bin/env python3
"""Take one straight-down aerial photo of the running Gazebo world.

The ground station draws this underneath the SLAM map, so the operator can
see the terrain they are drawing a search area on instead of grey pixels.

Run it ONCE per world while the simulation is up (any launch of that world
will do), then commit the PNG + YAML it writes into maps/:

    python3 scripts/capture_aerial_map.py
    python3 scripts/capture_aerial_map.py --world large_demo \
        --x-min -18.77 --x-max 51.23 --y-min -8.77 --y-max 36.5

How it works:
  1. Spawn a temporary static camera very high above the middle of the area,
     pointing straight down. From that high a perspective camera is nearly
     orthographic: for the Blue Mountains area it sits ~640 m up, where a 30 m
     rock top leans by under 2 m at the image edge.
  2. Bridge its image into ROS, wait for a frame that has finished rendering,
     and save it.
  3. Remove the camera again, so the sim is left as it was.

The YAML records which world rectangle the image covers, in Gazebo world
coordinates. The ground station shifts that into the scout's map frame.
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

MODEL_NAME = 'aerial_capture'
TOPIC = '/aerial_capture/image'

# Default area: the trimmed Blue Mountains ground plane, from
# blue_mountains_gazebo/README.md.
DEFAULT_BOUNDS = (-18.77, 51.23, -8.77, 36.5)

# Smallest <horizontal_fov> SDF accepts, plus a little margin.
MIN_HFOV = 0.11


def camera_sdf(cx, cy, height, hfov, width_px, height_px):
    """A static model holding one camera that looks straight down.

    Pitch +90 deg turns the camera's forward (+x) axis to point down; the
    extra yaw of +90 deg then puts world +y at the top of the image and world
    +x to the right, i.e. an ordinary north-up map.
    """
    return f"""<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{MODEL_NAME}">
    <static>true</static>
    <pose>{cx} {cy} {height} 0 {math.pi / 2} {math.pi / 2}</pose>
    <link name="link">
      <sensor name="camera" type="camera">
        <always_on>1</always_on>
        <update_rate>2</update_rate>
        <topic>{TOPIC}</topic>
        <camera>
          <horizontal_fov>{hfov}</horizontal_fov>
          <image>
            <width>{width_px}</width>
            <height>{height_px}</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>{height - 300.0}</near>
            <far>{height + 50.0}</far>
          </clip>
        </camera>
      </sensor>
    </link>
  </model>
</sdf>
"""


def ign_service(world, service, reqtype, req):
    cmd = ['ign', 'service', '-s', f'/world/{world}/{service}',
           '--reqtype', reqtype, '--reptype', 'ignition.msgs.Boolean',
           '--timeout', '5000', '--req', req]
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.returncode == 0 and 'data: true' in out.stdout


class FrameGrabber(Node):
    def __init__(self):
        super().__init__('aerial_capture')
        self.bridge = CvBridge()
        self.frames = 0
        self.frame = None
        self.create_subscription(Image, TOPIC, self.on_image, qos_profile_sensor_data)

    def on_image(self, msg):
        self.frames += 1
        self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')


def main():
    here = os.path.dirname(os.path.realpath(__file__))
    default_out = os.path.join(here, '..', 'maps', 'blue_mountains_aerial')

    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--world', default='large_demo', help='Gazebo world name')
    ap.add_argument('--x-min', type=float, default=DEFAULT_BOUNDS[0])
    ap.add_argument('--x-max', type=float, default=DEFAULT_BOUNDS[1])
    ap.add_argument('--y-min', type=float, default=DEFAULT_BOUNDS[2])
    ap.add_argument('--y-max', type=float, default=DEFAULT_BOUNDS[3])
    ap.add_argument('--width-px', type=int, default=2048)
    ap.add_argument('--altitude', type=float, default=1000.0,
                    help='Camera height (m); lowered automatically if the FOV '
                         'would drop below what SDF allows')
    ap.add_argument('--out', default=default_out,
                    help='Output path without extension (.png and .yaml are written)')
    args = ap.parse_args()

    span_x = args.x_max - args.x_min
    span_y = args.y_max - args.y_min
    height_px = int(round(args.width_px * span_y / span_x))
    # SDF refuses a horizontal FOV under 0.1 rad, which caps how high the
    # camera can go and still frame exactly this area. Fly lower if needed.
    max_altitude = (span_x / 2.0) / math.tan(MIN_HFOV / 2.0)
    altitude = min(args.altitude, max_altitude)
    # Gazebo derives the vertical FOV from the aspect ratio, so matching the
    # pixel aspect to the area's aspect makes the image cover it exactly.
    hfov = 2.0 * math.atan((span_x / 2.0) / altitude)
    cx = (args.x_min + args.x_max) / 2.0
    cy = (args.y_min + args.y_max) / 2.0

    sdf = camera_sdf(cx, cy, altitude, hfov, args.width_px, height_px)
    with tempfile.NamedTemporaryFile('w', suffix='.sdf', delete=False) as f:
        f.write(sdf)
        sdf_path = f.name

    rclpy.init()
    node = FrameGrabber()
    bridge = subprocess.Popen(
        ['ros2', 'run', 'ros_ign_bridge', 'parameter_bridge',
         f'{TOPIC}@sensor_msgs/msg/Image[ignition.msgs.Image'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    spawned = False
    try:
        print(f'Spawning aerial camera {altitude:.0f} m above ({cx:.1f}, {cy:.1f})...')
        spawned = ign_service(args.world, 'create', 'ignition.msgs.EntityFactory',
                              f'sdf_filename: "{sdf_path}"')
        if not spawned:
            print(f'Could not spawn the camera. Is the "{args.world}" world running?',
                  file=sys.stderr)
            return 1

        # The first frames can arrive before textures and lighting have
        # loaded, so skip a few and keep a later one.
        deadline = time.monotonic() + 60.0
        while node.frames < 4 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if node.frame is None:
            print('No image arrived from the aerial camera within 60 s.', file=sys.stderr)
            return 1
        if float(np.mean(node.frame)) < 2.0:
            print('Warning: the image is almost black - check the world has a light.',
                  file=sys.stderr)

        out = os.path.normpath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out + '.png', node.frame)
        with open(out + '.yaml', 'w') as f:
            yaml.safe_dump({
                'image': os.path.basename(out) + '.png',
                'frame': 'gazebo world',
                'world': args.world,
                # The world rectangle the image covers, in metres. Image top
                # is y_max, image right is x_max.
                'x_min': args.x_min, 'x_max': args.x_max,
                'y_min': args.y_min, 'y_max': args.y_max,
            }, f, sort_keys=False)
        print(f'Saved {out}.png ({args.width_px}x{height_px}) and {out}.yaml')
        return 0
    finally:
        if spawned:
            ign_service(args.world, 'remove', 'ignition.msgs.Entity',
                        f'name: "{MODEL_NAME}" type: MODEL')
        bridge.terminate()
        os.unlink(sdf_path)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
