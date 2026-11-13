#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source /opt/ros/humble/setup.bash; set -u

if ! command -v rosdep >/dev/null 2>&1 || ! command -v colcon >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3-rosdep python3-colcon-common-extensions
fi
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update
if find "$SRC_DIR" -name package.xml -not -path "*/assets/*" | grep -q .; then
    rosdep install --from-paths "$SRC_DIR" --ignore-src --rosdistro humble -y
fi

LIST="$SRC_DIR/custom/ros_packages.txt"
PKGS=$(grep -v '^\s*#' "$LIST" | tr -s ' \t' '\n' | grep -v '^$' || true)
if [ -n "$PKGS" ]; then
    echo "[haic] apt install from custom/ros_packages.txt: $PKGS"
    # shellcheck disable=SC2086
    sudo apt-get install -y $PKGS
else
    echo "[haic] custom/ros_packages.txt lists no packages"
fi
echo "[haic] ROS dependencies installed"
