from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    robot = LaunchConfiguration('robot')
    use_sim_time = LaunchConfiguration('use_sim_time')
    confirmed_point_topic = LaunchConfiguration('confirmed_point_topic')
    target_world_x = LaunchConfiguration('target_world_x')
    target_world_y = LaunchConfiguration('target_world_y')
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
    # Fixed delivery point in world coordinates: above person1 in
    # worlds/large_demo.sdf.
    ld.add_action(DeclareLaunchArgument(
        'target_world_x', default_value='-6.36',
        description='Delivery point x in Gazebo world coordinates.',
    ))
    ld.add_action(DeclareLaunchArgument(
        'target_world_y', default_value='-3.07',
        description='Delivery point y in Gazebo world coordinates.',
    ))
    # These two must stay in step with parrot2's spawn x/y in
    # 41068_ignition.launch.py - they are how courier_node.py converts the
    # world point into its own odom frame.
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
            'target_world_x': target_world_x,
            'target_world_y': target_world_y,
            'own_spawn_x': own_spawn_x,
            'own_spawn_y': own_spawn_y,
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    ))

    return ld
