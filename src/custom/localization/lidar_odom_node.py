"""LiDAR odometry node: /lidar/points (+ /imu gyro) -> /lidar_odom and TF odom -> pelvis.

No map needed. The localizer then publishes map -> odom on top of this (standard REP-105
chain map -> odom -> pelvis), so self.pose() keeps working and the pose stays continuous
even when a map match fails.
    python custom/localization/lidar_odom_node.py
"""
from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from icp_core import Pose  # noqa: E402
from icp_localizer_node import cloud_xyz, mat_to_quat, quat_to_mat  # noqa: E402
from odom_core import ScanOdometry  # noqa: E402

import rclpy  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import Imu, PointCloud2  # noqa: E402
from tf2_ros import Buffer, TransformBroadcaster, TransformListener  # noqa: E402


class LidarOdom(Node):
    def __init__(self):
        super().__init__("lidar_odom",
                         parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("child_frame", "pelvis")
        self.declare_parameter("topic", "/lidar_odom")
        self.declare_parameter("publish_tf", True)
        self.frame = str(self.get_parameter("odom_frame").value)
        self.child = str(self.get_parameter("child_frame").value)
        self.pub_tf = bool(self.get_parameter("publish_tf").value)

        self.od = ScanOdometry()
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.tf_pub = TransformBroadcaster(self)
        self.pub = self.create_publisher(Odometry, str(self.get_parameter("topic").value), 10)

        self.lock = threading.Lock()
        self.latest = None
        self.gyro_yaw = 0.0
        self.last_imu_t = None
        self.R_pel_imu = None
        self.vel = np.zeros(2)
        self.last_t = None
        self.create_subscription(PointCloud2, "/lidar/points", self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(Imu, "/imu", self.on_imu, qos_profile_sensor_data)
        threading.Thread(target=self.loop, daemon=True).start()
        self.get_logger().info(f"LiDAR odometry: TF {self.frame} -> {self.child}")

    def on_cloud(self, msg):
        with self.lock:
            self.latest = msg

    def on_imu(self, msg: Imu):
        if self.R_pel_imu is None:
            try:
                tf = self.tf_buf.lookup_transform("pelvis", msg.header.frame_id, Time())
                self.R_pel_imu = quat_to_mat(tf.transform.rotation)
            except Exception:                              # noqa: BLE001
                return
        w = self.R_pel_imu @ np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                                       msg.angular_velocity.z])
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        with self.lock:
            if self.last_imu_t is not None and 0 < t - self.last_imu_t < 0.2:
                p = self.od.pose                           # yaw rate about world z
                up = np.array([-math.sin(p.pitch), math.sin(p.roll) * math.cos(p.pitch),
                               math.cos(p.roll) * math.cos(p.pitch)])
                self.gyro_yaw += float(w @ up) * (t - self.last_imu_t)
            self.last_imu_t = t

    def loop(self):
        while rclpy.ok():
            with self.lock:
                msg, self.latest = self.latest, None
            if msg is None:
                time.sleep(0.005)
                continue
            try:
                self.process(msg)
            except Exception as e:                         # noqa: BLE001
                self.get_logger().error(f"odometry step failed: {e!r}")

    def process(self, msg: PointCloud2):
        stamp = Time.from_msg(msg.header.stamp)
        try:
            tf = self.tf_buf.lookup_transform("pelvis", msg.header.frame_id, stamp,
                                              timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception:                                  # noqa: BLE001
            try:
                tf = self.tf_buf.lookup_transform("pelvis", msg.header.frame_id, Time())
            except Exception:                              # noqa: BLE001
                return
        R = quat_to_mat(tf.transform.rotation)
        tr = tf.transform.translation
        pts = cloud_xyz(msg) @ R.T + np.array([tr.x, tr.y, tr.z])
        t = stamp.nanoseconds * 1e-9
        with self.lock:
            dyaw, self.gyro_yaw = self.gyro_yaw, 0.0
            dt = 0.0 if self.last_t is None else max(0.0, min(0.5, t - self.last_t))
        before = Pose(**vars(self.od.pose))
        pose, ok = self.od.update(pts, dyaw, tuple(self.vel * dt))
        if ok and dt > 0:
            v = np.array([pose.x - before.x, pose.y - before.y]) / dt
            self.vel = 0.5 * self.vel + 0.5 * np.clip(v, -1.5, 1.5)
        elif not ok:
            self.vel *= 0.5
        self.last_t = t
        self.publish(pose, msg.header.stamp, dt, before)

    def publish(self, pose: Pose, stamp, dt: float, before: Pose):
        T = pose.matrix()
        qx, qy, qz, qw = mat_to_quat(T[:3, :3])
        if self.pub_tf:
            tf = TransformStamped()
            tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, self.frame, self.child
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = T[:3, 3]
            r = tf.transform.rotation
            r.x, r.y, r.z, r.w = qx, qy, qz, qw
            self.tf_pub.sendTransform(tf)
        m = Odometry()
        m.header.stamp, m.header.frame_id, m.child_frame_id = stamp, self.frame, self.child
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = T[:3, 3]
        o = m.pose.pose.orientation
        o.x, o.y, o.z, o.w = qx, qy, qz, qw
        if dt > 0:                                          # body-frame twist
            c, s = math.cos(pose.yaw), math.sin(pose.yaw)
            vx, vy = (pose.x - before.x) / dt, (pose.y - before.y) / dt
            m.twist.twist.linear.x, m.twist.twist.linear.y = c * vx + s * vy, -s * vx + c * vy
            m.twist.twist.angular.z = math.atan2(math.sin(pose.yaw - before.yaw),
                                                 math.cos(pose.yaw - before.yaw)) / dt
        self.pub.publish(m)


def main():
    rclpy.init()
    node = LidarOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
