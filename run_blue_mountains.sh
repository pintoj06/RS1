#!/usr/bin/env bash
# Start the whole AeroAid stack in the Blue Mountains (Three Sisters) world:
# Gazebo + scout (parrot1) + courier (parrot2) + Nav2 + ground station UI
# + the scout's RGB person detector.
#
#   ./run_blue_mountains.sh          start everything
#   ./run_blue_mountains.sh kill     stop everything and clean up strays
#
# Each stage gets its own window so one node's log can be read on its own.
# Uses gnome-terminal tabs if installed, otherwise xterm windows, otherwise
# a tmux session (for a machine with no display).
#
# Overridable from the environment, e.g.
#   SIM_WAIT=45 LARGE_COURIER=false ./run_blue_mountains.sh
#   HEADLESS=true ./run_blue_mountains.sh     no Gazebo window, faster

set -u

PKG=41068_ignition_bringup
SCOUT=parrot1
COURIER=parrot2
SESSION=aeroaid

# large_demo is the world that includes model://blue_mountains.
WORLD=large_demo

# The workspace is two levels up from this script (<ws>/src/RS1/...), so this
# works whatever the workspace is called.
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
WS=${WS:-$(cd "$HERE/../.." && pwd)}

# Aerial photo drawn under the ground station's map, made once with
# scripts/capture_aerial_map.py. If it is missing the map just shows without it.
AERIAL_MAP=${AERIAL_MAP:-$HERE/maps/blue_mountains_aerial.yaml}

LARGE_COURIER=${LARGE_COURIER:-true}

# Blue Mountains is a much heavier world than simple_trees; give Gazebo,
# SLAM and Nav2 longer to come up before anything tries to plan.
SIM_WAIT=${SIM_WAIT:-30}

# HEADLESS=true runs Gazebo without its 3D window. The ground station still
# shows the map and both camera feeds; drawing the Blue Mountains scene a
# second time for the window is a big share of the load on a laptop.
HEADLESS=${HEADLESS:-false}

# Every node that does a TF lookup needs these: this package pushes TF into
# each robot's namespace instead of the global /tf.
TF_REMAP="-r /tf:=/${SCOUT}/tf -r /tf_static:=/${SCOUT}/tf_static"

cleanup() {
    echo "Stopping everything..."
    pkill -f 'ign gazebo'      2>/dev/null
    pkill -f parameter_bridge  2>/dev/null
    pkill -f ruby              2>/dev/null
    pkill -f ground_station    2>/dev/null
    pkill -f yolo_person_detector 2>/dev/null
    pkill -f search_node       2>/dev/null
    pkill -f courier_node      2>/dev/null
    pkill -f rviz2             2>/dev/null
    pkill -f takeoff_node      2>/dev/null
    # The rest of the sim launch. Killing Gazebo alone leaves these running,
    # and a leftover Nav2 or SLAM node shares its name with the next run's.
    local stack="ros2 launch $PKG|slam_toolbox|ekf_node|robot_state_publisher"
    stack+="|controller_server|planner_server|smoother_server|behavior_server"
    stack+="|bt_navigator|waypoint_follower|velocity_smoother|lifecycle_manager"
    pkill -f "$stack"          2>/dev/null
    tmux kill-session -t "$SESSION" 2>/dev/null
    sleep 2
    pkill -9 -f "$stack|ign gazebo|parameter_bridge" 2>/dev/null
    sleep 1
    ros2 daemon stop >/dev/null 2>&1
    # FastDDS shared-memory files left by processes that did not exit
    # cleanly. Stale ones stop new nodes discovering each other's services
    # (e.g. a Nav2 lifecycle manager waiting forever on smoother_server).
    rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
    ros2 daemon start >/dev/null 2>&1
    echo "Done."
}

if [[ "${1:-}" == "kill" ]]; then
    cleanup
    exit 0
fi

# Always start from a clean slate. A stale Gazebo from the last run will
# happily keep publishing sensor topics while the new launch fails around it,
# which looks like a code problem and is not one.
cleanup

if [[ ! -f "$WS/install/setup.bash" ]]; then
    echo "No workspace at $WS/install - run colcon build first." >&2
    exit 1
fi

if command -v gnome-terminal >/dev/null; then
    TERM_KIND=gnome
elif command -v xterm >/dev/null && [[ -n "${DISPLAY:-}" ]]; then
    TERM_KIND=xterm
elif command -v tmux >/dev/null; then
    TERM_KIND=tmux
else
    echo "Need gnome-terminal, xterm or tmux to open the stage windows." >&2
    exit 1
fi

# Open one stage in its own window. 'exec bash' keeps it alive after the
# node exits, so you can still read why it died.
tab() {
    local title="$1"; shift
    local cmd="source /opt/ros/humble/setup.bash;
               source $WS/install/setup.bash;
               $*;
               echo; echo '--- $title exited ---';
               exec bash"
    case "$TERM_KIND" in
        gnome) gnome-terminal --tab --title="$title" -- bash -c "$cmd" ;;
        xterm) xterm -T "$title" -geometry 110x30 -e bash -c "$cmd" & ;;
        tmux)
            if tmux has-session -t "$SESSION" 2>/dev/null; then
                tmux new-window -t "$SESSION" -n "$title" "bash -c \"$cmd\""
            else
                tmux new-session -d -s "$SESSION" -n "$title" "bash -c \"$cmd\""
            fi ;;
    esac
}

echo "1/5  Simulation (Blue Mountains) + scout + courier + Nav2"
tab "sim" "ros2 launch $PKG 41068_ignition.launch.py \
    world:=$WORLD parrot:=true courier:=true large_courier:=$LARGE_COURIER \
    husky:=false nav2:=true rviz:=false headless:=$HEADLESS"

echo "     waiting ${SIM_WAIT}s for the simulation to come up..."
sleep "$SIM_WAIT"

echo "2/5  Ground station"
tab "gui" "ros2 run $PKG ground_station.py --ros-args \
    -r __ns:=/$SCOUT -p robot_name:=$SCOUT -p aerial_map:=$AERIAL_MAP $TF_REMAP"

sleep 2

# The scout idles until a search area is drawn in the ground station, and
# both drones stay on the ute until LAUNCH is pressed there.
# Confirms contacts: one is raised only when the thermal camera sees body
# heat AND this detector sees a human there (see _check_detections in
# UI/ground_station.py).
# Needs ultralytics - see the top of scripts/yolo_person_detector.py.
echo "3/5  Scout RGB person detector (YOLO)"
tab "yolo" "ros2 run $PKG yolo_person_detector.py --ros-args \
    -r __ns:=/$SCOUT -p robot_name:=$SCOUT -p use_sim_time:=true"

sleep 2

echo "4/5  Scout search"
tab "scout" "ros2 launch $PKG 41068_search.launch.py robot:=$SCOUT"

sleep 2

echo "5/5  Courier"
tab "courier" "ros2 launch $PKG 41068_courier_demo.launch.py robot:=$COURIER"

cat <<EOF

All five stages started ($TERM_KIND).

  Both drones are sitting on the ute. In the ground station:
    1. SET AREA, then drag a box on the map (the aerial photo is shown
       until SLAM starts).
    2. LAUNCH (bottom right, or R). The drones climb to 10 m and 8 m,
       SLAM and Nav2 come up, and the scout flies the search area.

  Check it worked:   ros2 node list
  Stop everything:   $0 kill
EOF

if [[ "$TERM_KIND" == tmux ]]; then
    echo "  Logs:              tmux attach -t $SESSION   (Ctrl-b n for next window)"
fi

cat <<EOF

If the map stays empty, check the lidar is returning something:
  ros2 topic echo /$SCOUT/scan --field ranges --once | head -3

If Nav2 goals keep aborting, check localisation first (near zero = healthy):
  timeout 8 ros2 run tf2_ros tf2_echo ${SCOUT}_map ${SCOUT}_odom \\
    --ros-args -r /tf:=/${SCOUT}/tf -r /tf_static:=/${SCOUT}/tf_static
EOF
