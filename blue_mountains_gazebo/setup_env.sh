#!/usr/bin/env bash
# Source this file in the same shell that starts your existing ROS launch.
BLUE_MOUNTAINS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export IGN_GAZEBO_RESOURCE_PATH="${BLUE_MOUNTAINS_ROOT}/models${IGN_GAZEBO_RESOURCE_PATH:+:${IGN_GAZEBO_RESOURCE_PATH}}"
export BLUE_MOUNTAINS_WORLD="${BLUE_MOUNTAINS_ROOT}/worlds/large_demo.sdf"
printf 'Blue Mountains world: %s\n' "$BLUE_MOUNTAINS_WORLD"
