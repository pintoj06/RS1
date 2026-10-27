#!/usr/bin/env bash
# Start the whole AeroAid stack in the Blue Mountains (Three Sisters) world:
# Gazebo + scout (parrot1) + courier (parrot2) + Nav2 + ground station UI.
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

# Every node that does a TF lookup needs these: this package pushes TF into
# each robot's namespace instead of the global /tf.
TF_REMAP="-r /tf:=/${SCOUT}/tf -r /tf_static:=/${SCOUT}/tf_static"

cleanup() {
    echo "Stopping everything..."
    pkill -f 'ign gazebo'      2>/dev/null
    pkill -f parameter_bridge  2>/dev/null
    pkill -f ruby              2>/dev/null
    pkill -f ground_station    2>/dev/null
    pkill -f search_node       2>/dev/null
    pkill -f courier_node      2>/dev/null
    pkill -f rviz2             2>/dev/null
    tmux kill-session -t "$SESSION" 2>/dev/null
    sleep 1
    ros2 daemon stop >/dev/null 2>&1
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

echo "1/4  Simulation (Blue Mountains) + scout + courier + Nav2 + RViz"
tab "sim" "ros2 launch $PKG 41068_ignition.launch.py \
    world:=$WORLD parrot:=true courier:=true large_courier:=$LARGE_COURIER \
    husky:=false nav2:=true rviz:=true"

echo "     waiting ${SIM_WAIT}s for the simulation to come up..."
sleep "$SIM_WAIT"

echo "2/4  Ground station"
tab "gui" "ros2 run $PKG ground_station.py --ros-args \
    -r __ns:=/$SCOUT -p robot_name:=$SCOUT -p aerial_map:=$AERIAL_MAP $TF_REMAP"

sleep 2

# The scout idles until a search area is drawn in the ground station.
echo "3/4  Scout search"
tab "scout" "ros2 launch $PKG 41068_search.launch.py robot:=$SCOUT"

sleep 2

echo "4/4  Courier"
tab "courier" "ros2 launch $PKG 41068_courier_demo.launch.py robot:=$COURIER"

cat <<EOF

All four stages started ($TERM_KIND).

  The scout is idle. Drag a box on the ground station map to start the search.

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
