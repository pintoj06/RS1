#!/usr/bin/env python3
"""Teleport person1/kangaroo/wombat and the drone through mixed-subject scenes.

Generalises multi_location_dataset_capture.py (single subject: person1) to
scenes containing any combination of the three ground entities in
worlds/large_demo.sdf - person1 (static), demo_animal (kangaroo, dynamic),
and wombat1 (wombat, dynamic). For each scene it:

  1. Teleports every one of the three entities: the ones active in this
     scene go to a computed position near a shared "centroid" point, spread
     apart from each other by a varying distance (so different scenes of
     the same category aren't all identically spaced); every entity NOT
     active in this scene gets teleported to a fixed parking spot far from
     any camera position, so it can never accidentally appear in frame.
  2. Waits for the dynamic entities (kangaroo/wombat) to settle under
     gravity - they're placed ~1m above the local ground and dropped, per
     the "Dynamic, so they settle onto the terrain" comment already in
     worlds/large_demo.sdf for these two models.
  3. Teleports the drone through `photos_per_location` evenly-spaced points
     on a circle around the centroid, at a constant altitude, saving one
     frame per point - same approach as multi_location_dataset_capture.py.
  4. Moves to the next scene.

Frames go into <output_dir>/scene_{i:02d}_<entities>/, e.g.
scene_00_human/, scene_05_human-kangaroo/, scene_40_human-kangaroo-wombat/.
Numbering continues after whatever scene_NN folders already exist in
output_dir, so re-running with a different plan doesn't overwrite earlier
output (same approach as multi_location_dataset_capture.py).

The default plan (48 scenes total, matching what was asked for):
    5  human only            10 kangaroo only    10 wombat only
    5  human + kangaroo       5 human + wombat     5 wombat + kangaroo
    8  human + kangaroo + wombat

Fixes a duplicate-frame bug found in multi_location_dataset_capture.py:
that script's "wait for a fresh frame" check compared wall-clock arrival
times, but a message already in flight *before* a teleport can still
arrive and get timestamped *after* it - so occasionally a stale,
pre-teleport frame got saved as if it were fresh. Fixed here (and should be
back-ported there) by discarding the current image immediately after every
teleport, waiting out a fixed settle window while continuing to discard
anything that arrives during it, and only then waiting for the *next*
arrival - which is guaranteed to be rendered after the settle window closed.

Example:

    ros2 run 41068_ignition_bringup multi_entity_dataset_capture.py --ros-args \\
        -r __ns:=/parrot1 \\
        -p robot_name:=parrot1 \\
        -p output_dir:=/home/student/41068_ignition_bringup_v1_backup/41068_ignition_bringup/YoloTraining/multi_entity
"""

import itertools
import math
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import cv2
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image

HUMAN = 'person1'
KANGAROO = 'demo_animal'
WOMBAT = 'wombat1'
ALL_ENTITIES = (HUMAN, KANGAROO, WOMBAT)
SHORT_NAME = {HUMAN: 'human', KANGAROO: 'kangaroo', WOMBAT: 'wombat'}

# person1 is static, so it can safely sit close to a tree (no force
# resolution needed between two things that never move). kangaroo/wombat
# are dynamic rigid bodies with real extent (the kangaroo model alone has
# a tail collision box reaching 0.91m from its origin) - dropping one
# within, say, 0.3m of a tree trunk overlaps its collision mesh deeply
# enough that the physics engine's contact solver applies a huge
# corrective impulse, launching it kilometres away in one step. Verified
# live (2026-09-30): this exact failure, with the kangaroo ending up at
# world position (-7522, -10775, -6954) after being placed 0.32m from a
# tree. This is the minimum clearance kept from every known tree trunk
# when placing a dynamic entity.
DYNAMIC_ENTITY_TREE_CLEARANCE = 1.6


def _load_tree_trunk_positions() -> List[Tuple[float, float]]:
    model_sdf = os.path.join(
        get_package_share_directory('41068_ignition_bringup'),
        'models', 'blue_mountains', 'model.sdf',
    )
    with open(model_sdf) as f:
        content = f.read()
    poses = re.findall(r'<collision name="[^"]*trunk[^"]*">\s*<pose>([-\d.]+) ([-\d.]+)', content)
    return [(float(x), float(y)) for x, y in poses]

# Ten clearings verified earlier in this dataset (real tree collision
# positions checked against blue_mountains' model.sdf, flat/low-elevation
# ground confirmed) - reused here as scene centroids, cycling through them
# so every category gets background variety rather than reusing just one.
CLEARINGS = [
    (-14.8, -4.7), (-7.6, -5.6), (6.6, -7.5), (7.9, 7.8), (-6.0, 6.5),
    (11.0, 4.0), (-7.5, 15.0), (11.5, -2.0), (-13.5, 15.0), (-23.0, -2.3),
]

# (entities in scene, how many scenes, orbit_radius for this category,
# per-scene entity-to-centroid spacing in metres).
#
# Framing math, empirically calibrated in circle_dataset_capture.py: at
# altitude 10m with this camera's FOV, a single point offset ~2.5m from
# "directly below the drone" reliably stays in frame; ~3.2m already risks
# clipping; ~4.5m falls out entirely. For a *group*, the drone's own
# orbit_radius offset and an entity's spacing from the centroid can add up
# on the far side of the orbit (when that entity ends up roughly opposite
# the drone's current offset direction, which happens for every entity at
# some point across a full 360 degree orbit) - so orbit_radius + spacing
# must itself stay within that same ~2.5m safe budget, not each term
# separately. Verified live (2026-09-30): an earlier version of this table
# used orbit_radius=2.0 with spacing up to 2.0 (worst case 4.0m) for pairs,
# and two consecutive test shots each showed only ONE of the two animals -
# never both together. Re-tuned below so worst case stays <=~2.1m for
# pairs and <=~1.6m for the (harder, 3-way) triple case, then re-verified.
CATEGORY_SPEC = [
    ((HUMAN,), 5, 2.5, [0.0]),
    # Lower orbit_radius than the human-only case: these clearings were
    # picked to sit 0.7-1.3m from a tree (fine for static person1), so the
    # tree-clearance nudge below will almost always shift a dynamic animal
    # placed at zero offset - leaving less budget for the orbit's own
    # off-centre push before the combined offset risks leaving frame.
    ((KANGAROO,), 10, 1.5, [0.0]),
    ((WOMBAT,), 10, 1.5, [0.0]),
    ((HUMAN, KANGAROO), 5, 1.2, [0.5, 0.7, 0.9, 0.6, 0.8]),
    ((HUMAN, WOMBAT), 5, 1.2, [0.6, 0.8, 0.5, 0.9, 0.7]),
    ((WOMBAT, KANGAROO), 5, 1.2, [0.5, 0.7, 0.9, 0.6, 0.8]),
    ((HUMAN, KANGAROO, WOMBAT), 8, 1.0, [0.4, 0.5, 0.6, 0.45, 0.55, 0.4, 0.5, 0.6]),
]

# Far enough from every clearing (nearest is ~50m away, well outside any
# camera's ~6m ground footprint) to never appear in frame, but NOT extreme:
# an earlier version of this used (500, 500) and it crashed the whole
# simulation - ODE's broad-phase spatial hash uses fixed-precision bounds,
# and an object that far from the rest of the scene (which spans roughly
# -25..15 here) overflows it ("aabbBound >= dMinIntExact" assertion,
# aborting the ign gazebo server entirely). Verified live that 50 does not
# reproduce this after the (500, 500) crash was found and fixed.
PARK_POSITIONS = {
    HUMAN: (50.0, 50.0, 2.0),
    KANGAROO: (55.0, 50.0, 2.0),
    WOMBAT: (60.0, 50.0, 2.0),
}


def _entity_offsets(entities: Tuple[str, ...], spacing: float, start_angle: float) -> Dict[str, Tuple[float, float]]:
    """Positions for `entities` on a circle of radius `spacing` around (0, 0)."""
    if len(entities) == 1:
        return {entities[0]: (0.0, 0.0)}
    offsets = {}
    for i, name in enumerate(entities):
        angle = start_angle + 2.0 * math.pi * i / len(entities)
        offsets[name] = (spacing * math.cos(angle), spacing * math.sin(angle))
    return offsets


def _is_clear(x: float, y: float, trees: List[Tuple[float, float]], min_clearance: float) -> bool:
    return all(math.hypot(x - tx, y - ty) >= min_clearance for tx, ty in trees)


def _push_clear_of_trees(
    x: float, y: float, trees: List[Tuple[float, float]], min_clearance: float,
) -> Tuple[float, float]:
    """Find the nearest point to (x, y) that clears every tree by min_clearance.

    Only meant for dynamic entities (kangaroo/wombat) - see
    DYNAMIC_ENTITY_TREE_CLEARANCE for why. person1 is static and doesn't
    need this: two things that never move don't need contact resolved
    between them, so overlapping a tree costs it nothing.

    An earlier version nudged locally away from whichever tree was
    currently violated (or, in a second attempt, away from all violators at
    once). Both got stuck: in a dense enough cluster, three-plus trees can
    be close enough together that no point *near* (x, y) clears all of
    them, so any local nudge just settles into a stable-but-still-violating
    equilibrium (verified live 2026-09-30 - repeated runs converged to
    ~1.3-1.5m against a 1.6m requirement and stopped improving). This
    instead does an expanding radial search - checking a ring of angles at
    increasing radii out from the original point - which finds a genuinely
    clear spot outside the whole cluster rather than a local optimum
    inside it, at the cost of a possibly larger shift from the intended
    position.
    """
    if _is_clear(x, y, trees, min_clearance):
        return x, y
    for radius in (0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.0):
        for angle_deg in range(0, 360, 15):
            angle = math.radians(angle_deg)
            cx, cy = x + radius * math.cos(angle), y + radius * math.sin(angle)
            if _is_clear(cx, cy, trees, min_clearance):
                return cx, cy
    return x, y  # give up (extremely dense cluster) - best effort, original point


def build_scenes() -> List[dict]:
    """Expand CATEGORY_SPEC into one dict per scene: entities, positions, orbit_radius."""
    trees = _load_tree_trunk_positions()
    scenes = []
    clearing_cycle = itertools.cycle(CLEARINGS)
    for entities, count, orbit_radius, spacings in CATEGORY_SPEC:
        for i in range(count):
            cx, cy = next(clearing_cycle)
            spacing = spacings[i % len(spacings)]
            start_angle = (i * 47.0) * math.pi / 180.0  # vary arrangement per scene
            offsets = _entity_offsets(entities, spacing, start_angle)
            positions = {}
            for name, (dx, dy) in offsets.items():
                x, y = cx + dx, cy + dy
                if name != HUMAN:
                    x, y = _push_clear_of_trees(x, y, trees, DYNAMIC_ENTITY_TREE_CLEARANCE)
                positions[name] = (x, y)
            # Orbit around where the entities actually ended up, not the
            # original clearing point - the tree-clearance search above can
            # shift a dynamic entity by a couple of metres to escape a dense
            # cluster, and orbiting the stale pre-shift point would then aim
            # the drone's off-centre push at the wrong spot entirely.
            actual_cx = sum(p[0] for p in positions.values()) / len(positions)
            actual_cy = sum(p[1] for p in positions.values()) / len(positions)
            scenes.append({
                'entities': entities,
                'centroid': (actual_cx, actual_cy),
                'positions': positions,
                'orbit_radius': orbit_radius,
            })
    return scenes


def _yaw_to_quaternion_zw(yaw: float):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class MultiEntityDatasetCapture(Node):

    def __init__(self):
        super().__init__('multi_entity_dataset_capture')

        self.declare_parameter('robot_name', 'parrot1')
        self.declare_parameter('image_topic', 'camera/image')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('world_name', 'large_demo')

        self.declare_parameter('altitude', 10.0)
        self.declare_parameter('person_z', 0.505)
        # Dynamic entities are dropped from this height and left to settle
        # onto the terrain (see module docstring) rather than placed at a
        # guessed ground height directly.
        self.declare_parameter('animal_drop_z', 1.0)
        self.declare_parameter('photos_per_location', 15)

        self.declare_parameter(
            'output_dir',
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'YoloTraining',
                'multi_entity',
            ),
        )
        # How long to let dynamic entities settle after being placed in a
        # new scene, before starting that scene's first shot.
        self.declare_parameter('scene_settle_time', 1.2)
        # How long to discard incoming frames after every drone teleport,
        # before starting to wait for the first genuinely-fresh one - the
        # fix for the duplicate-frame bug (see module docstring).
        self.declare_parameter('frame_settle_time', 0.5)
        self.declare_parameter('frame_wait_timeout', 3.0)

        self.robot_name = str(self.get_parameter('robot_name').value)
        self.world_name = str(self.get_parameter('world_name').value)
        self.altitude = float(self.get_parameter('altitude').value)
        self.person_z = float(self.get_parameter('person_z').value)
        self.animal_drop_z = float(self.get_parameter('animal_drop_z').value)
        self.photos_per_location = int(self.get_parameter('photos_per_location').value)
        self.scene_settle_time = float(self.get_parameter('scene_settle_time').value)
        self.frame_settle_time = float(self.get_parameter('frame_settle_time').value)
        self.frame_wait_timeout = float(self.get_parameter('frame_wait_timeout').value)
        self.output_dir = os.path.expanduser(str(self.get_parameter('output_dir').value))

        os.makedirs(self.output_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.latest_image: Optional[Image] = None

        self.cmd_vel_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').value, 10
        )
        self.image_sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._image_callback, 1
        )

        self.scenes = build_scenes()
        self.get_logger().info(
            f'multi_entity_dataset_capture started for "{self.robot_name}": '
            f'{len(self.scenes)} scene(s), {self.photos_per_location} photos each, '
            f'saving to {self.output_dir}.'
        )

    def _image_callback(self, msg: Image) -> None:
        self.latest_image = msg

    def _set_pose(self, name: str, x: float, y: float, z: float, yaw: float = 0.0) -> bool:
        qz, qw = _yaw_to_quaternion_zw(yaw)
        req = (
            f'name: "{name}", '
            f'position: {{x: {x!r}, y: {y!r}, z: {z!r}}}, '
            f'orientation: {{x: 0, y: 0, z: {qz!r}, w: {qw!r}}}'
        )
        # Retried: under sustained load (this world's dense collision
        # geometry keeps the physics step busy) the server can occasionally
        # take longer than the client-side timeout to reply - Ignition logs
        # that server-side as "NodeShared::RecvSrvRequest() error sending
        # response: Host unreachable" once our client has already given up
        # and exited. Silently moving on when that happens leaves whichever
        # entity failed to teleport sitting wherever it was before - e.g. an
        # animal still parked off in a corner - so the next frame gets
        # captured with it missing from view (verified live 2026-09-30:
        # this exact failure mode, reported as "animals aren't in frame").
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

    def _spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _wait_for_fresh_frame(self) -> None:
        # Discard whatever arrives during the settle window (this may
        # include stale, pre-teleport frames still in flight), then wait
        # for the *next* arrival after that - guaranteed to postdate it.
        self.latest_image = None
        self._spin_for(self.frame_settle_time)
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

    def _next_scene_index(self) -> int:
        existing = []
        if os.path.isdir(self.output_dir):
            for entry in os.listdir(self.output_dir):
                m = re.match(r'scene_(\d+)_', entry)
                if m:
                    existing.append(int(m.group(1)))
        return max(existing, default=-1) + 1

    def _place_scene_entities(self, scene: dict) -> None:
        for entity in ALL_ENTITIES:
            if entity in scene['positions']:
                x, y = scene['positions'][entity]
                z = self.person_z if entity == HUMAN else self.animal_drop_z
                self._set_pose(entity, x, y, z)
            else:
                px, py, pz = PARK_POSITIONS[entity]
                self._set_pose(entity, px, py, pz)

    def run(self) -> None:
        self.cmd_vel_pub.publish(Twist())

        start_index = self._next_scene_index()
        if start_index > 0:
            self.get_logger().info(
                f'Found existing scene_00..{start_index - 1:02d} in {self.output_dir} - '
                f'continuing from scene_{start_index:02d} instead of overwriting them.'
            )

        for offset, scene in enumerate(self.scenes):
            scene_index = start_index + offset
            label = '-'.join(SHORT_NAME[e] for e in scene['entities'])
            scene_dir = os.path.join(self.output_dir, f'scene_{scene_index:02d}_{label}')
            os.makedirs(scene_dir, exist_ok=True)

            cx, cy = scene['centroid']
            self.get_logger().info(
                f'--- Scene {scene_index} ({label}): centroid ({cx:.2f}, {cy:.2f}), '
                f'positions {scene["positions"]} ---'
            )
            self._place_scene_entities(scene)
            self._spin_for(self.scene_settle_time)

            orbit_radius = scene['orbit_radius']
            for shot_index in range(self.photos_per_location):
                theta = 2.0 * math.pi * shot_index / self.photos_per_location
                drone_x = cx + orbit_radius * math.cos(theta)
                drone_y = cy + orbit_radius * math.sin(theta)
                yaw = math.atan2(cy - drone_y, cx - drone_x)

                self._set_pose(self.robot_name, drone_x, drone_y, self.altitude, yaw)
                self.cmd_vel_pub.publish(Twist())
                self._wait_for_fresh_frame()
                self._save_frame(scene_dir, shot_index)

            self.get_logger().info(f'Finished scene {scene_index}.')

        self.get_logger().info('All scenes complete - shutting down.')


def main(args=None):
    rclpy.init(args=args)
    node = MultiEntityDatasetCapture()
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
