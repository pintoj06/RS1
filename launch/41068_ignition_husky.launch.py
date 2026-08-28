from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ld = LaunchDescription()

    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz = LaunchConfiguration('rviz')
    slam = LaunchConfiguration('slam')
    nav2 = LaunchConfiguration('nav2')
    world = LaunchConfiguration('world')

    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time',
    ))
    ld.add_action(DeclareLaunchArgument(
        'rviz',
        default_value='False',
        description='Flag to launch RViz',
    ))
    ld.add_action(DeclareLaunchArgument(
        'slam',
        default_value='False',
        description='Flag to launch SLAM Toolbox',
    ))
    ld.add_action(DeclareLaunchArgument(
        'nav2',
        default_value='False',
        description='Flag to launch Nav2. Nav2 also starts SLAM.',
    ))
    ld.add_action(DeclareLaunchArgument(
        'world',
        default_value='simple_trees',
        description='Which world to load',
        choices=['simple_trees', 'large_demo'],
    ))

    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('41068_ignition_bringup'),
            'launch',
            '41068_ignition.launch.py',
        ])),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'rviz': rviz,
            'slam': slam,
            'nav2': nav2,
            'world': world,
            'husky': 'True',
            'parrot': 'False',
        }.items(),
    ))

    return ld
