from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    robot = LaunchConfiguration('robot')
    use_sim_time = LaunchConfiguration('use_sim_time')
    target_x = LaunchConfiguration('target_x')
    target_y = LaunchConfiguration('target_y')
    scout_complete_topic = LaunchConfiguration('scout_complete_topic')

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
        'target_x',
        default_value='0.0',
        description='X coordinate of the delivery point, in the courier odom frame.',
    ))
    ld.add_action(DeclareLaunchArgument(
        'target_y',
        default_value='-6.0',
        description='Y coordinate of the delivery point, in the courier odom frame.',
    ))
    ld.add_action(DeclareLaunchArgument(
        'scout_complete_topic',
        default_value='/parrot1/search_node/sweep_complete',
        description='Absolute topic the scout drone publishes to once its sweep is done.',
    ))

    # This launch file intentionally does not start Gazebo, robots, SLAM,
    # Nav2, or RViz. Start the main simulation (with courier:=True so the
    # parrot2 robot is spawned) and the scout's search_node.py first, then
    # run this launch file from a separate terminal.
    ld.add_action(Node(
        package='41068_ignition_bringup',
        executable='courier_node.py',
        namespace=robot,
        name='courier_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_name': robot,
            'target_x': target_x,
            'target_y': target_y,
            'scout_complete_topic': scout_complete_topic,
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    ))

    return ld
