from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    robot = LaunchConfiguration('robot')
    use_sim_time = LaunchConfiguration('use_sim_time')

    ld.add_action(DeclareLaunchArgument(
        'robot',
        default_value='husky1',
        choices=['husky1', 'parrot1'],
        description='Robot namespace to control. Use husky1 or parrot1.',
    ))
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time',
    ))

    # This launch file intentionally does not start Gazebo, robots, SLAM,
    # Nav2, or RViz. Start the normal simulation first, then run this launch
    # file from a separate terminal as an autonomy-code example.
    ld.add_action(Node(
        package='41068_ignition_bringup',
        executable='basic_autonomy_demo.py',
        namespace=robot,
        name='basic_autonomy_demo',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_name': robot,
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    ))

    return ld
