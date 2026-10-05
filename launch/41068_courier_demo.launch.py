from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ld = LaunchDescription()

    robot = LaunchConfiguration('robot')
    use_sim_time = LaunchConfiguration('use_sim_time')
    confirmed_point_topic = LaunchConfiguration('confirmed_point_topic')
    target_world_x = LaunchConfiguration('target_world_x')
    target_world_y = LaunchConfiguration('target_world_y')
    use_contact_position = LaunchConfiguration('use_contact_position')

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
    ld.add_action(DeclareLaunchArgument(
        'use_contact_position', default_value='True',
        description='Fly to the contact position confirmed by the scout. '
                    'False flies to the fixed target_world_x/y instead.',
    ))
    # Fixed delivery point in world coordinates, used only when
    # use_contact_position is False: above person1 in worlds/large_demo.sdf.
    ld.add_action(DeclareLaunchArgument(
        'target_world_x', default_value='-6.36',
        description='Fixed delivery point x in Gazebo world coordinates.',
    ))
    ld.add_action(DeclareLaunchArgument(
        'target_world_y', default_value='-3.07',
        description='Fixed delivery point y in Gazebo world coordinates.',
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
            'use_contact_position': ParameterValue(use_contact_position, value_type=bool),
            'target_world_x': ParameterValue(target_world_x, value_type=float),
            'target_world_y': ParameterValue(target_world_y, value_type=float),
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    ))

    return ld
