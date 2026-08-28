#!/usr/bin/env python3
"""Small example of changing an Ignition Gazebo world from a ROS 2 node.

This is deliberately simple: it moves a small animal marker around the forest
floor, and cycles a demo tree between healthy, fire, and burnt appearances. The
goal is to provide a compact Python file students can read and modify for their
own dynamic-world ideas.

Implementation note:
The demo models are already present in ``worlds/large_demo.sdf``. This node only
moves existing models by calling Gazebo's ``/world/<world>/set_pose`` service.
That keeps the example simple and avoids relying on dynamic model insertion.

This script uses the ``ign`` / ``gz`` command-line tool through ``subprocess``.
That is not the most efficient possible Gazebo API, but it is very explicit and
readable for a teaching example: students can copy the printed service command
idea and experiment with Gazebo services from a terminal.
"""

import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable, List

import rclpy
from rclpy.node import Node


@dataclass
class Pose2D:
    """Minimal 2D pose container used by this demo.

    Gazebo still receives a full 3D pose, but the demo only changes x, y, z, and
    yaw. Roll and pitch are always zero.
    """

    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0


class DynamicWorldDemo(Node):
    """Move simple pre-existing Gazebo models to create a dynamic scene."""

    def __init__(self):
        super().__init__('dynamic_world_demo')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        # The models named below are defined directly inside worlds/large_demo.sdf.
        # If students rename models in the world file, update these parameters too.
        self.declare_parameter('world_name', 'large_demo')
        self.declare_parameter('animal_name', 'demo_animal')
        self.declare_parameter('tree_healthy_name', 'demo_tree_healthy')
        self.declare_parameter('tree_fire_name', 'demo_tree_fire')
        self.declare_parameter('tree_burnt_name', 'demo_tree_burnt')

        # Initial demo locations.
        self.declare_parameter('tree_x', 4.0)
        self.declare_parameter('tree_y', -4.0)
        self.declare_parameter('animal_x', -4.0)
        self.declare_parameter('animal_y', 4.0)

        # update_period is intentionally left at 0.1 s. This gives a visibly
        # smooth animal marker while still keeping the example simple.
        self.declare_parameter('update_period', 0.1)
        self.declare_parameter('tree_state_period', 3.0)
        self.declare_parameter('step_size', 0.2)
        self.declare_parameter('wander_limit', 9.0)

        self.world_name = self.get_parameter('world_name').value
        self.animal_name = self.get_parameter('animal_name').value
        self.tree_names = {
            'healthy': self.get_parameter('tree_healthy_name').value,
            'fire': self.get_parameter('tree_fire_name').value,
            'burnt': self.get_parameter('tree_burnt_name').value,
        }
        self.tree_x = float(self.get_parameter('tree_x').value)
        self.tree_y = float(self.get_parameter('tree_y').value)
        self.update_period = float(self.get_parameter('update_period').value)
        self.tree_state_period = float(self.get_parameter('tree_state_period').value)
        self.step_size = float(self.get_parameter('step_size').value)
        self.wander_limit = float(self.get_parameter('wander_limit').value)

        # The demo tree is implemented as three simple models. Only one is shown
        # at the tree location at a time; the inactive states are moved below the
        # world. This is crude but easy to understand and reliable.
        self.tree_states = ['healthy', 'fire', 'burnt']
        self.tree_state_index = 0
        self.seconds_since_tree_change = 0.0
        self.hidden_pose = Pose2D(0.0, 0.0, -30.0, 0.0)

        self.animal_pose = Pose2D(
            x=float(self.get_parameter('animal_x').value),
            y=float(self.get_parameter('animal_y').value),
            z=0.0,
            yaw=0.0,
        )

        # Gazebo Fortress normally uses the ``ign`` command. Newer Gazebo
        # versions use ``gz``. This keeps the example usable in either case.
        self.gz_command = 'ign' if shutil.which('ign') else 'gz'
        self.msg_prefix = 'ignition.msgs' if self.gz_command == 'ign' else 'gz.msgs'

        self.initialised = False
        self.timer = self.create_timer(self.update_period, self.update)

        self.get_logger().info(
            f'Dynamic world demo started for world [{self.world_name}]. '
            'Start the normal large_demo Gazebo simulation first, then this add-on node.'
        )

    # ------------------------------------------------------------------
    # Gazebo command helper
    # ------------------------------------------------------------------
    def _run(
        self,
        command: Iterable[str],
        *,
        timeout: float = 2.0,
        warn_on_failure: bool = True,
    ) -> bool:
        """Run a Gazebo command-line service call and return True on success."""
        command = list(command)
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            if warn_on_failure:
                self.get_logger().warn(f'Command timed out: {" ".join(command)}')
            return False
        except FileNotFoundError:
            if warn_on_failure:
                self.get_logger().warn(f'Command not found: {command[0]}')
            return False

        if result.returncode != 0:
            if warn_on_failure:
                self.get_logger().warn(
                    f'Command failed: {" ".join(command)}\n'
                    f'stdout: {result.stdout.strip()}\n'
                    f'stderr: {result.stderr.strip()}'
                )
            return False
        return True

    def set_model_pose(self, name: str, pose: Pose2D, *, warn_on_failure: bool = True) -> bool:
        """Move an existing Gazebo model by calling the set_pose service."""

        # The request string is in Gazebo Transport protobuf text format. It is
        # the same style of request that can be sent manually with:
        #   ign service -s /world/large_demo/set_pose --req 'name: "..." ...'
        request = (
            f'name: "{name}" '
            f'position {{ x: {pose.x:.6f} y: {pose.y:.6f} z: {pose.z:.6f} }} '
            f'orientation {{ z: {math.sin(pose.yaw / 2.0):.6f} '
            f'w: {math.cos(pose.yaw / 2.0):.6f} }}'
        )
        return self._run([
            self.gz_command, 'service',
            '-s', f'/world/{self.world_name}/set_pose',
            '--reqtype', f'{self.msg_prefix}.Pose',
            '--reptype', f'{self.msg_prefix}.Boolean',
            '--timeout', '1000',
            '--req', request,
        ], timeout=2.0, warn_on_failure=warn_on_failure)

    # ------------------------------------------------------------------
    # Demo logic
    # ------------------------------------------------------------------
    def update(self):
        """Timer callback: initialise once, then move the demo objects."""
        if not self.initialised:
            self.initialised = self.reset_demo_models()
            return

        self.move_animal()
        self.seconds_since_tree_change += self.update_period
        if self.seconds_since_tree_change >= self.tree_state_period:
            self.seconds_since_tree_change = 0.0
            self.tree_state_index = (self.tree_state_index + 1) % len(self.tree_states)
            self.update_tree_state()

    def reset_demo_models(self) -> bool:
        """Move all demo models to their initial visible/hidden positions."""
        self.get_logger().info('Resetting existing dynamic demo animal and tree markers.')
        ok = self.set_model_pose(self.animal_name, self.animal_pose)
        ok = self.update_tree_state() and ok
        if not ok:
            self.get_logger().warn(
                'Could not move the demo models. Make sure the main simulation is running '
                'with world:=large_demo, because the demo models are defined in that world file.'
            )
        return ok

    def move_animal(self):
        """Move the animal marker with a small bounded random walk."""
        # Change heading randomly, then step forward. This is intentionally not
        # physically realistic; it just creates a moving target/obstacle idea.
        self.animal_pose.yaw += random.uniform(-0.7, 0.7)
        self.animal_pose.x += self.step_size * math.cos(self.animal_pose.yaw)
        self.animal_pose.y += self.step_size * math.sin(self.animal_pose.yaw)

        # Keep the animal inside a square region by reflecting off the boundary.
        if abs(self.animal_pose.x) > self.wander_limit:
            self.animal_pose.x = max(min(self.animal_pose.x, self.wander_limit), -self.wander_limit)
            self.animal_pose.yaw = math.pi - self.animal_pose.yaw
        if abs(self.animal_pose.y) > self.wander_limit:
            self.animal_pose.y = max(min(self.animal_pose.y, self.wander_limit), -self.wander_limit)
            self.animal_pose.yaw = -self.animal_pose.yaw

        self.set_model_pose(self.animal_name, self.animal_pose)

    def update_tree_state(self) -> bool:
        """Show one tree state and hide the other two states below the world."""
        visible_state = self.tree_states[self.tree_state_index]
        visible_pose = Pose2D(self.tree_x, self.tree_y, 0.0, 0.0)

        ok = True
        for state, name in self.tree_names.items():
            ok = self.set_model_pose(
                name,
                visible_pose if state == visible_state else self.hidden_pose,
            ) and ok

        if ok:
            self.get_logger().info(f'Demo tree state: {visible_state}')
        return ok


def main(args: List[str] = None):
    rclpy.init(args=args)
    node = DynamicWorldDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
