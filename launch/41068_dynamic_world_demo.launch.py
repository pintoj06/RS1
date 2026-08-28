from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # This launch file intentionally does not start Gazebo, robots, SLAM,
    # Nav2, or RViz. Start the normal simulation first, then run this launch
    # file from a second terminal as an add-on dynamic-world example.
    args = [
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        DeclareLaunchArgument(
            'world',
            default_value='large_demo',
            choices=['large_demo'],
            description='Gazebo world name to modify. The demo is intended for large_demo.',
        ),
    ]
    for arg in args:
        ld.add_action(arg)

    ld.add_action(Node(
        package='41068_ignition_bringup',
        executable='dynamic_world_demo.py',
        name='dynamic_world_demo',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'world_name': LaunchConfiguration('world'),
        }],
    ))

    return ld
