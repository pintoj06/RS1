# Launches the scout drone's search-grid sweep node.
#
# This file intentionally does NOT start Gazebo, SLAM, Nav2 or RViz. Start the
# normal simulation first (41068_ignition.launch.py), then run this from a
# second terminal once Nav2 is up.
#
#   ros2 launch 41068_ignition_bringup 41068_search.launch.py
#
# Override the search area on the command line, e.g. a 16 m box:
#   ros2 launch 41068_ignition_bringup 41068_search.launch.py \
#       min_x:=-8.0 max_x:=8.0 min_y:=-8.0 max_y:=8.0 spacing:=4.0
#
# Or an arbitrary polygon as a flat [x1,y1,x2,y2,...] list (overrides the
# min/max rectangle). Note the quoting - the list has to survive the shell:
#   ros2 launch 41068_ignition_bringup 41068_search.launch.py \
#       polygon:="[-8.0, -8.0, 8.0, -8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 8.0, -8.0, 8.0]"

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
        description='Robot namespace to run the search in. The scout drone is parrot1.',
    ))
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation clock. Must be True when running against Gazebo.',
    ))

    # --- Search area: rectangle form -------------------------------------
    # Used only when 'polygon' is left empty. Keep the area inside the world
    # or the drone will be sent somewhere it cannot reach.
    ld.add_action(DeclareLaunchArgument(
        'min_x', default_value='-4.0', description='Search rectangle minimum x (m).'))
    ld.add_action(DeclareLaunchArgument(
        'max_x', default_value='4.0', description='Search rectangle maximum x (m).'))
    ld.add_action(DeclareLaunchArgument(
        'min_y', default_value='-4.0', description='Search rectangle minimum y (m).'))
    ld.add_action(DeclareLaunchArgument(
        'max_y', default_value='4.0', description='Search rectangle maximum y (m).'))

    # --- Search area: arbitrary polygon form -----------------------------
    # Flat list of corners [x1, y1, x2, y2, ...]. When this has 3 or more
    # corners in it, it wins and the rectangle above is ignored.
    ld.add_action(DeclareLaunchArgument(
        'polygon',
        default_value='[0.0]',
        description='Search polygon as a flat [x1,y1,x2,y2,...] list. '
                    'Leave as the default to use the min/max rectangle instead.',
    ))

    # --- Sweep spacing ----------------------------------------------------
    # Distance between the parallel legs of the lawnmower pattern. Smaller =
    # more overlap and no gaps, but a longer flight. The node logs whether
    # this is narrow enough for the camera footprint at the current altitude.
    ld.add_action(DeclareLaunchArgument(
        'spacing',
        default_value='0.8',
        description='Spacing between sweep legs (m). Must be smaller than the '
                    'camera ground footprint or coverage will have gaps.',
    ))

    ld.add_action(Node(
        package='41068_ignition_bringup',
        executable='search_node.py',
        namespace=robot,
        name='search_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_name': robot,
            'min_x': LaunchConfiguration('min_x'),
            'max_x': LaunchConfiguration('max_x'),
            'min_y': LaunchConfiguration('min_y'),
            'max_y': LaunchConfiguration('max_y'),
            'spacing': LaunchConfiguration('spacing'),
            'polygon': LaunchConfiguration('polygon'),
        }],
        # This package publishes TF inside the robot namespace (/parrot1/tf).
        # Without these two remaps the node's TF lookups read the global /tf,
        # find nothing, and every "where is the drone?" query fails.
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    ))

    return ld
