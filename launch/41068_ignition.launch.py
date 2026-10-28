import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


_TRUE_STRINGS = "['true', '1', 'yes', 'on']"


def _is_true_expression(name):
    return ["'", LaunchConfiguration(name), "'.lower() in ", _TRUE_STRINGS]


def _if_true(name):
    return IfCondition(PythonExpression(_is_true_expression(name)))


def _if_all_true(*names):
    pieces = []
    for i, name in enumerate(names):
        if i > 0:
            pieces.append(' and ')
        pieces.extend(_is_true_expression(name))
    return IfCondition(PythonExpression(pieces))


def add_robot(
    ld,
    *,
    pkg_path,
    config_path,
    use_sim_time,
    enabled_arg,
    namespace,
    frame_prefix,
    gz_model_name,
    xacro_parts,
    bridge_config,
    localization_config,
    x,
    y,
    z,
    yaw='0.0',
    spawn_delay=0.0,
    takeoff_altitude=None,
    extra_xacro_args='',
):
    """Add one robot instance to the launch description.

    If takeoff_altitude is given, the robot is a drone spawned on its landing
    pad: once it is in Gazebo, takeoff_node.py climbs it to that height. The
    takeoff node is returned so Nav2 can wait for it (see
    add_navigation_instance). Otherwise returns None.

    Convention used here:
      - ROS nodes/topics are under /<namespace>/...
      - Gazebo entity/model is named <gz_model_name>
      - Gazebo plugin topics are under /model/<gz_model_name>/...
      - TF topics are under /<namespace>/tf and /<namespace>/tf_static
      - TF frame ids use <frame_prefix>, e.g. husky1_base_link
    """

    robot_description_content = ParameterValue(
        Command([
            'xacro ',
            PathJoinSubstitution([pkg_path] + xacro_parts),
            ' ',
            'prefix:=', frame_prefix,
            ' ',
            'gz_model_name:=', gz_model_name,
            ' ', extra_xacro_args,
        ]),
        value_type=str,
    )

    robot_enabled = _if_true(enabled_arg)

    ld.add_action(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=namespace,
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
        condition=robot_enabled,
    ))

    ld.add_action(Node(
        package='robot_localization',
        executable='ekf_node',
        namespace=namespace,
        name='robot_localization',
        output='screen',
        parameters=[
            PathJoinSubstitution([config_path, localization_config]),
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
            ('odometry/filtered', 'odom'),
        ],
        condition=robot_enabled,
    ))

    robot_spawner = Node(
        package='ros_ign_gazebo',
        executable='create',
        name='spawn_' + namespace,
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-topic', '/' + namespace + '/robot_description',
            '-name', gz_model_name,
            '-x', x,
            '-y', y,
            '-z', z,
            '-Y', yaw,
        ],
    )

    # Start Gazebo first, then insert robot entities in a predictable order.
    ld.add_action(TimerAction(
        period=spawn_delay,
        actions=[robot_spawner],
        condition=robot_enabled,
    ))

    takeoff = None
    if takeoff_altitude is not None:
        takeoff = Node(
            package='41068_ignition_bringup',
            executable='takeoff_node.py',
            namespace=namespace,
            name='takeoff_node',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'target_altitude': float(takeoff_altitude),
            }],
        )
        # The spawner exits once the robot is in Gazebo.
        ld.add_action(RegisterEventHandler(OnProcessExit(
            target_action=robot_spawner,
            on_exit=[takeoff],
        )))

    ld.add_action(Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        namespace=namespace,
        name='gazebo_bridge',
        output='screen',
        parameters=[{
            'config_file': PathJoinSubstitution([config_path, bridge_config]),
            'use_sim_time': use_sim_time,
        }],
        condition=robot_enabled,
    ))

    return takeoff


def add_navigation_instance(
    ld,
    *,
    pkg_path,
    use_sim_time,
    robot_arg,
    robot_namespace,
    start_delay,
    nav2_config_suffix=None,
    after_takeoff=None,
):
    """Start SLAM/Nav2 for one robot.

    start_delay counts from launch, or, if after_takeoff (a robot's takeoff
    node) is given, from the moment that drone finishes climbing. SLAM must
    not start while the drone is still on the ute: its lidar would map the
    ute's cab as a wall right next to the drone.
    """
    nav_launch_arguments = {
        'use_sim_time': use_sim_time,
        'namespace': robot_namespace,
        'config_filename_suffix': '_' + robot_namespace,
        'slam': LaunchConfiguration('slam'),
        'nav2': LaunchConfiguration('nav2'),
    }
    if nav2_config_suffix is not None:
        nav_launch_arguments['nav2_config_filename_suffix'] = nav2_config_suffix

    nav_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                pkg_path,
                'launch',
                '41068_navigation.launch.py',
            ])
        ),
        launch_arguments=nav_launch_arguments.items(),
    )

    # Scope is important: without it, repeated IncludeLaunchDescription calls can
    # leak/overwrite launch configurations such as namespace and config suffix.
    nav_timer = TimerAction(
        period=start_delay,
        actions=[GroupAction(
            scoped=True,
            actions=[nav_include],
        )],
        condition=_if_true(robot_arg),
    )
    if after_takeoff is None:
        ld.add_action(nav_timer)
    else:
        ld.add_action(RegisterEventHandler(OnProcessExit(
            target_action=after_takeoff,
            on_exit=[nav_timer],
        )))


def add_rviz_instance(
    ld,
    *,
    config_path,
    use_sim_time,
    robot_arg,
    robot_namespace,
    rviz_config,
    start_delay,
):
    # Each RViz process is itself placed in the robot namespace so the Nav2 RViz
    # panel resolves relative action/service names such as navigate_to_pose and
    # lifecycle_manager_navigation to the correct robot.
    ld.add_action(TimerAction(
        period=start_delay,
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            namespace=robot_namespace,
            name='rviz2',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['-d', PathJoinSubstitution([config_path, rviz_config])],
            remappings=[
                ('/tf', '/' + robot_namespace + '/tf'),
                ('/tf_static', '/' + robot_namespace + '/tf_static'),
            ],
        )],
        condition=_if_all_true('rviz', robot_arg),
    ))


def generate_launch_description():

    ld = LaunchDescription()

    pkg_path = FindPackageShare('41068_ignition_bringup')
    config_path = PathJoinSubstitution([pkg_path, 'config'])

    use_sim_time_launch_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time',
    )
    use_sim_time = LaunchConfiguration('use_sim_time')
    ld.add_action(use_sim_time_launch_arg)

    rviz_launch_arg = DeclareLaunchArgument(
        'rviz',
        default_value='False',
        description='Flag to launch RViz',
    )
    ld.add_action(rviz_launch_arg)

    slam_launch_arg = DeclareLaunchArgument(
        'slam',
        default_value='False',
        description='Flag to launch SLAM Toolbox for each enabled robot',
    )
    ld.add_action(slam_launch_arg)

    nav2_launch_arg = DeclareLaunchArgument(
        'nav2',
        default_value='False',
        description='Flag to launch Nav2 for each enabled robot. Nav2 also starts SLAM.',
    )
    ld.add_action(nav2_launch_arg)

    husky_launch_arg = DeclareLaunchArgument(
        'husky',
        default_value='True',
        description='Launch the Husky UGV instance in namespace /husky1',
    )
    ld.add_action(husky_launch_arg)

    parrot_launch_arg = DeclareLaunchArgument(
        'parrot',
        default_value='False',
        description='Launch the Parrot drone instance in namespace /parrot1',
    )
    ld.add_action(parrot_launch_arg)

    courier_launch_arg = DeclareLaunchArgument(
        'courier',
        default_value='False',
        description='Launch a second Parrot drone (the courier) in namespace /parrot2',
    )
    ld.add_action(courier_launch_arg)

    large_courier_launch_arg = DeclareLaunchArgument(
        'large_courier',
        default_value='False',
        description='Use the large courier model (urdf_parrot/courier_large.urdf.xacro, '
                    'nav2_params_parrot2_large.yaml) for /parrot2 instead of the standard Parrot',
    )
    ld.add_action(large_courier_launch_arg)
    large_courier = PythonExpression(_is_true_expression('large_courier'))

    world_launch_arg = DeclareLaunchArgument(
        'world',
        default_value='simple_trees',
        description='Which world to load',
        choices=['simple_trees', 'large_demo'],
    )
    ld.add_action(world_launch_arg)

    headless_launch_arg = DeclareLaunchArgument(
        'headless',
        default_value='False',
        description='Run Gazebo without its GUI client (server only, no rendering window)',
    )
    ld.add_action(headless_launch_arg)

    # Load common Gazebo server systems from a shared config file. This keeps
    # required systems such as Sensors out of individual robot and world files,
    # so custom student worlds do not need to copy plugin blocks.
    server_config_file = PathJoinSubstitution([
        config_path,
        'ignition_server.config',
    ])
    ld.add_action(SetEnvironmentVariable(
        name='IGN_GAZEBO_SERVER_CONFIG_PATH',
        value=server_config_file,
    ))
    ld.add_action(SetEnvironmentVariable(
        name='GZ_SIM_SERVER_CONFIG_PATH',
        value=server_config_file,
    ))
    # Make world-local models (e.g. model://blue_mountains, used by large_demo)
    # resolvable without requiring each user to export a resource path manually.
    models_path = PathJoinSubstitution([pkg_path, 'models'])
    for resource_path_var in ('IGN_GAZEBO_RESOURCE_PATH', 'GZ_SIM_RESOURCE_PATH'):
        ld.add_action(SetEnvironmentVariable(
            name=resource_path_var,
            value=[models_path, os.pathsep, os.environ.get(resource_path_var, '')],
        ))

    # -s runs the server only (no gzclient GUI window) for headless operation,
    # e.g. on VMs without a GPU or when driving the sim without a desktop.
    headless_flag = PythonExpression(
        ["' -s' if '", LaunchConfiguration('headless'), "'.lower() in ", _TRUE_STRINGS, " else ''"]
    )

    # Start Gazebo once.
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ros_ign_gazebo'),
                'launch',
                'ign_gazebo.launch.py',
            ])
        ),
        launch_arguments={
            'ign_args': [
                PathJoinSubstitution([
                    pkg_path,
                    'worlds',
                    [LaunchConfiguration('world'), '.sdf'],
                ]),
                ' -r',
                headless_flag,
            ]
        }.items(),
    ))

    # Bridge the global simulation clock once.
    ld.add_action(Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge_clock',
        output='screen',
        parameters=[{
            'config_file': PathJoinSubstitution([config_path, 'gazebo_bridge_clock.yaml']),
            'use_sim_time': use_sim_time,
        }],
    ))

    add_robot(
        ld,
        pkg_path=pkg_path,
        config_path=config_path,
        use_sim_time=use_sim_time,
        enabled_arg='husky',
        namespace='husky1',
        frame_prefix='husky1_',
        gz_model_name='husky1',
        xacro_parts=['urdf_husky', 'husky.urdf.xacro'],
        bridge_config='gazebo_bridge_husky1.yaml',
        localization_config='robot_localization_husky1.yaml',
        x='0.0',
        y='0.0',
        z='0.4',
        spawn_delay=3.0,
    )

    # Both drones start on their landing pads on the AeroAid ute
    # (worlds/large_demo.sdf, model://aeroaid_ute). Spawn x/y are the pad
    # centres; spawn z puts the drone on the pad surface (z = 1.03).
    scout_takeoff = add_robot(
        ld,
        pkg_path=pkg_path,
        config_path=config_path,
        use_sim_time=use_sim_time,
        enabled_arg='parrot',
        namespace='parrot1',
        frame_prefix='parrot1_',
        gz_model_name='parrot1',
        xacro_parts=['urdf_parrot', 'parrot.urdf.xacro'],
        bridge_config='gazebo_bridge_parrot1.yaml',
        localization_config='robot_localization_parrot1.yaml',
        # Scout pad. The ground station's scout_spawn_x/y match this.
        x='2.0',
        y='0.0',
        z='1.08',
        spawn_delay=6.0,
        # Must match flight_altitude in 41068_search.launch.py.
        takeoff_altitude=10.0,
    )

    courier_takeoff = add_robot(
        ld,
        pkg_path=pkg_path,
        config_path=config_path,
        use_sim_time=use_sim_time,
        enabled_arg='courier',
        namespace='parrot2',
        frame_prefix='parrot2_',
        gz_model_name='parrot2',
        xacro_parts=['urdf_parrot', PythonExpression([
            "'courier_large.urdf.xacro' if ", large_courier, " else 'parrot.urdf.xacro'",
        ])],
        bridge_config='gazebo_bridge_parrot2.yaml',
        localization_config='robot_localization_parrot2.yaml',
        # Courier pad. Odom frames are the Gazebo world frame (see
        # courier_node.py), so nothing else depends on this position.
        x='2.47',
        y='-1.10',
        # Puts the bottom of the care package on the pad (z = 1.03): it hangs
        # 0.277 m below base_link on the large courier, 0.126 m on the
        # standard Parrot (see the care_package calls in the two xacros).
        z=PythonExpression(["'1.31' if ", large_courier, " else '1.16'"]),
        spawn_delay=9.0,
        # 2 m below the scout. The drone has gravity off and Nav2 only
        # drives it in x/y, so it holds this height for the whole flight.
        takeoff_altitude=8.0,
        # The courier carries the care package to the confirmed contact.
        extra_xacro_args='care_package:=true',
    )

    add_navigation_instance(
        ld,
        pkg_path=pkg_path,
        use_sim_time=use_sim_time,
        robot_arg='husky',
        robot_namespace='husky1',
        start_delay=7.0,
    )

    add_navigation_instance(
        ld,
        pkg_path=pkg_path,
        use_sim_time=use_sim_time,
        robot_arg='parrot',
        robot_namespace='parrot1',
        start_delay=0.0,
        after_takeoff=scout_takeoff,
    )

    add_navigation_instance(
        ld,
        pkg_path=pkg_path,
        use_sim_time=use_sim_time,
        robot_arg='courier',
        robot_namespace='parrot2',
        # Staggered behind the scout's stack so both don't come up at once:
        # both drones take off together on LAUNCH, and starting two Nav2
        # stacks a few seconds apart can leave one lifecycle manager waiting
        # forever on a service DDS never discovers. The courier is not
        # needed until a contact is confirmed, so it can afford to wait.
        start_delay=16.0,
        after_takeoff=courier_takeoff,
        nav2_config_suffix=PythonExpression([
            "'_parrot2_large' if ", large_courier, " else '_parrot2'",
        ]),
    )

    add_rviz_instance(
        ld,
        config_path=config_path,
        use_sim_time=use_sim_time,
        robot_arg='husky',
        robot_namespace='husky1',
        rviz_config='41068_husky1.rviz',
        start_delay=22.0,
    )

    add_rviz_instance(
        ld,
        config_path=config_path,
        use_sim_time=use_sim_time,
        robot_arg='parrot',
        robot_namespace='parrot1',
        rviz_config='41068_parrot1.rviz',
        start_delay=26.0,
    )

    add_rviz_instance(
        ld,
        config_path=config_path,
        use_sim_time=use_sim_time,
        robot_arg='courier',
        robot_namespace='parrot2',
        rviz_config='41068_parrot2.rviz',
        start_delay=32.0,
    )

    return ld
