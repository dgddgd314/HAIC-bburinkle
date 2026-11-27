import os
import shutil
from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription  # noqa: F401
from launch.launch_description_sources import PythonLaunchDescriptionSource  # noqa: F401
from launch_ros.actions import Node  # noqa: F401
from launch_ros.substitutions import FindPackageShare  # noqa: F401

CUSTOM = Path(__file__).resolve().parent


def _python() -> str:
    """The team conda env's python (ros2 launch itself runs under /usr/bin/python3)."""
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and (Path(prefix) / "bin" / "python3").exists():
        return str(Path(prefix) / "bin" / "python3")
    return shutil.which("python3") or "python3"


def generate_launch_description() -> LaunchDescription:
    loc = CUSTOM / "localization"
    nodes = [
        # LiDAR odometry (no map) -> TF odom -> pelvis, /lidar_odom
        ExecuteProcess(cmd=[_python(), str(loc / "lidar_odom_node.py")],
                       name="lidar_odom", output="screen"),
        # LiDAR + wall map ICP -> TF map -> odom, so map -> odom -> pelvis (self.pose())
        ExecuteProcess(cmd=[_python(), str(loc / "icp_localizer_node.py"),
                            "--ros-args", "-p", "odom_frame:=odom"],
                       name="icp_localizer", output="screen"),
    ]
    # testing only: HAIC_NO_LOCALIZATION=1 python main.py --sim-only ... (run the nodes by hand)
    if os.environ.get("HAIC_NO_LOCALIZATION") == "1":
        nodes = []
    return LaunchDescription(nodes)
