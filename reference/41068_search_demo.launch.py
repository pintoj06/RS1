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
        default_value='parrot1',
        choices=['husky1', 'parrot1'],
        description='Robot namespace to run the search sweep on. Use husky1 or parrot1.',
    ))
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time',
    ))

    # Search rectangle corners and row spacing, in the robot's map frame
    # (metres). These stand in for what the mission UI will eventually set.
    for name, default in [
        ('search_min_x', '-5.0'),
        ('search_max_x', '5.0'),
        ('search_min_y', '-5.0'),
        ('search_max_y', '5.0'),
        ('lane_spacing', '2.0'),
    ]:
        ld.add_action(DeclareLaunchArgument(name, default_value=default))

    # This launch file intentionally does not start Gazebo, robots, SLAM,
    # Nav2, or RViz. Start the normal simulation first (with slam:=true
    # nav2:=true), then run this launch file from a separate terminal.
    ld.add_action(Node(
        package='41068_ignition_bringup',
        executable='search_manager_node.py',
        namespace=robot,
        name='search_manager_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_name': robot,
            'search_min_x': LaunchConfiguration('search_min_x'),
            'search_max_x': LaunchConfiguration('search_max_x'),
            'search_min_y': LaunchConfiguration('search_min_y'),
            'search_max_y': LaunchConfiguration('search_max_y'),
            'lane_spacing': LaunchConfiguration('lane_spacing'),
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    ))

    return ld
