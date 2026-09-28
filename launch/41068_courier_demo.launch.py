from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    robot = LaunchConfiguration('robot')
    use_sim_time = LaunchConfiguration('use_sim_time')
    confirmed_point_topic = LaunchConfiguration('confirmed_point_topic')
    scout_spawn_x = LaunchConfiguration('scout_spawn_x')
    scout_spawn_y = LaunchConfiguration('scout_spawn_y')
    own_spawn_x = LaunchConfiguration('own_spawn_x')
    own_spawn_y = LaunchConfiguration('own_spawn_y')

    ld.add_action(DeclareLaunchArgument(
        'robot',
        default_value='parrot2',
        description='Robot namespace for the courier drone.',
    ))
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time',
    ))
    ld.add_action(DeclareLaunchArgument(
        'confirmed_point_topic',
        default_value='/parrot1/operator/confirmed_point',
        description='Absolute topic the ground station publishes a confirmed contact to.',
    ))
    # These four must stay in step with the x/y spawn args used for parrot1
    # and parrot2 in 41068_ignition.launch.py - they are how courier_node.py
    # converts a confirmed point from the scout's frame into its own.
    ld.add_action(DeclareLaunchArgument(
        'scout_spawn_x', default_value='2.0',
        description="Scout drone's spawn x in 41068_ignition.launch.py.",
    ))
    ld.add_action(DeclareLaunchArgument(
        'scout_spawn_y', default_value='0.0',
        description="Scout drone's spawn y in 41068_ignition.launch.py.",
    ))
    ld.add_action(DeclareLaunchArgument(
        'own_spawn_x', default_value='2.0',
        description="Courier drone's own spawn x in 41068_ignition.launch.py.",
    ))
    ld.add_action(DeclareLaunchArgument(
        'own_spawn_y', default_value='-2.0',
        description="Courier drone's own spawn y in 41068_ignition.launch.py.",
    ))

    # This launch file intentionally does not start Gazebo, robots, SLAM,
    # Nav2, or RViz. Start the main simulation (with courier:=True so the
    # parrot2 robot is spawned), the scout's search_node.py, and the ground
    # station first, then run this launch file from a separate terminal.
    ld.add_action(Node(
        package='41068_ignition_bringup',
        executable='courier_node.py',
        namespace=robot,
        name='courier_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_name': robot,
            'confirmed_point_topic': confirmed_point_topic,
            'scout_spawn_x': scout_spawn_x,
            'scout_spawn_y': scout_spawn_y,
            'own_spawn_x': own_spawn_x,
            'own_spawn_y': own_spawn_y,
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    ))

    return ld
