from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription  # noqa: F401
from launch.launch_description_sources import PythonLaunchDescriptionSource  # noqa: F401
from launch_ros.actions import Node  # noqa: F401
from launch_ros.substitutions import FindPackageShare  # noqa: F401


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
    ])
