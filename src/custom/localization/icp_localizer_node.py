"""ICP localization node: /lidar/points + walls_reference.ply -> TF map -> pelvis.

Run by custom/bringup.launch.py (or by hand while `python main.py --sim-only` runs):
    python custom/localization/icp_localizer_node.py

Inputs
  /lidar/points   PointCloud2 (lidar_link), 10 Hz
  /imu            pelvis IMU: accelerometer -> tilt prior, gyro -> yaw prediction
  /tf             pelvis -> lidar_link (robot_state_publisher)
  /initialpose    PoseWithCovarianceStamped: manual re-seed; searched +-1 m, +-45 deg
  /g1/mission_status  simulator status (latched JSON): "startup" / "respawned" events carry
                  pos + yaw_deg -> re-seed (README 4: respawn is a teleport the odometry can't see)
Outputs
  odom_frame = ""     : TF map -> pelvis (full 6-DoF; self.pose() reads x, y, yaw from it)
  odom_frame = "odom" : TF map -> odom (planar correction) on top of lidar_odom_node's
                        odom -> pelvis; odometry is the prediction and carries the pose
                        through scans the map match rejects
  /localization/pose    PoseWithCovarianceStamped
  /localization/status  std_msgs/String (JSON: ok, inlier ratio, rmse, lost count)
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from icp_core import Localizer, Pose, WallMap, rot_rpy, wrap  # noqa: E402

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,  # noqa: E402
                       qos_profile_sensor_data)
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import Imu, PointCloud2  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_ros import Buffer, TransformBroadcaster, TransformListener  # noqa: E402

START_POSE = (16.0, 11.0, 0.0)        # README: G1 start (16, 11), yaw 0
# README mission start points (x, y, yaw_deg): extra guesses when tracking is lost
KNOWN_SPOTS = [(16.0, 11.0, 0.0), (15.5, 21.0, 90.0), (13.0, 20.125, 180.0),
               (22.25, 19.55, 180.0), (13.75, 9.10, 180.0)]
NOMINAL_PITCH = math.radians(13.0)    # README: standing pelvis leans ~13 deg forward


def cloud_xyz(msg: PointCloud2) -> np.ndarray:
    off = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    cols = [raw[:, off[k]:off[k] + 4].copy().view("<f4").reshape(-1) for k in ("x", "y", "z")]
    p = np.stack(cols, axis=1).astype(np.float64)
    return p[np.isfinite(p).all(axis=1)]


def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R: np.ndarray):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


class IcpLocalizer(Node):
    def __init__(self):
        super().__init__("icp_localizer",
                         parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.declare_parameter("initial_pose", list(START_POSE))
        self.declare_parameter("lost_after", 5)          # bad scans in a row -> relocalize
        self.declare_parameter("child_frame", "pelvis")  # tests: "pelvis_icp" next to ground truth
        self.child = str(self.get_parameter("child_frame").value)
        self.declare_parameter("odom_frame", "")        # "odom": run with lidar_odom_node.py
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.corr = None                                  # map <- odom planar (x, y, yaw)
        self.skipped = 0                                  # scans dropped: no odometry TF yet
        self.gen = 0                                      # bumped by every re-seed
        self.seed, self.seed_time, self.fixed = None, None, False
        self.declare_parameter("search_min_ratio", 0.75)  # stricter than tracking (0.65):
        # a lookalike corridor section can score ~0.68, the true pose 0.8-0.9
        x0, y0, yaw0 = self.get_parameter("initial_pose").value

        t = time.monotonic()
        self.loc = Localizer(WallMap())
        self.get_logger().info(f"wall map ready ({len(self.loc.map.p)} pts, {time.monotonic() - t:.1f}s)")

        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.tf_pub = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/localization/pose", 10)
        self.status_pub = self.create_publisher(String, "/localization/status", 10)

        self.lock = threading.Lock()
        self.pose = Pose(x0, y0, yaw0, 0.79, 0.0, NOMINAL_PITCH)
        self.need_search = True                           # first fix: search around the start pose
        self.bad = 0
        self.vel = np.zeros(2)                            # world-frame xy velocity estimate
        self.last_stamp = None
        self.gyro_yaw = 0.0                               # integrated yaw rate since last scan
        self.last_imu_t = None
        self.up_imu = None                                # low-passed accel direction (pelvis frame)
        self.R_pel_imu = None
        self.latest = None

        self.create_subscription(PointCloud2, "/lidar/points", self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(Imu, "/imu", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_initialpose, 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/g1/mission_status", self.on_mission_status, latched)
        self.last_seq = None
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()

    # ------------------------------------------------------------------ inputs
    def on_cloud(self, msg: PointCloud2):
        with self.lock:
            self.latest = msg                              # worker always takes the newest scan

    def on_imu(self, msg: Imu):
        if self.R_pel_imu is None:
            try:
                tf = self.tf_buf.lookup_transform("pelvis", msg.header.frame_id, Time())
                self.R_pel_imu = quat_to_mat(tf.transform.rotation)
            except Exception:                              # noqa: BLE001
                return
        a = self.R_pel_imu @ np.array([msg.linear_acceleration.x, msg.linear_acceleration.y,
                                       msg.linear_acceleration.z])
        w = self.R_pel_imu @ np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                                       msg.angular_velocity.z])
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        with self.lock:
            n = np.linalg.norm(a)
            if 5.0 < n < 15.0:
                u = a / n
                self.up_imu = u if self.up_imu is None else 0.98 * self.up_imu + 0.02 * u
            if self.last_imu_t is not None and 0 < t - self.last_imu_t < 0.2:
                # yaw rate about world z = (body rate) . (world up in body frame)
                up = self.up_imu if self.up_imu is not None else np.array([0, 0, 1.0])
                self.gyro_yaw += float(w @ up / np.linalg.norm(up)) * (t - self.last_imu_t)
            self.last_imu_t = t

    def on_initialpose(self, msg: PoseWithCovarianceStamped):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        st = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self.reseed(p.x, p.y, yaw, "/initialpose", st if st > 0 else None)

    def on_mission_status(self, msg: String):
        try:
            s = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if s.get("seq") is not None and s.get("seq") == self.last_seq:
            return
        self.last_seq = s.get("seq")
        if s.get("event") in ("startup", "respawned") and s.get("pos") and "yaw_deg" in s:
            x, y = map(float, s["pos"][:2])
            self.reseed(x, y, math.radians(float(s["yaw_deg"])), f"mission_status {s['event']}",
                        s.get("sim_time"))

    def reseed(self, x: float, y: float, yaw: float, source: str, sim_time=None):
        with self.lock:
            self.pose = Pose(x, y, yaw, self.pose.z, self.pose.roll, self.pose.pitch)
            self.seed = Pose(**vars(self.pose))
            self.seed_time = float(sim_time) if sim_time is not None else None
            self.fixed = False                             # no fix since this seed yet
            self.need_search, self.bad, self.vel[:] = True, 0, 0.0
            self.gyro_yaw = 0.0
            self.gen += 1                                  # results of in-flight scans are stale
        self.get_logger().info(f"re-seed from {source}: ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)")

    # ------------------------------------------------------------------ worker
    def loop(self):
        while rclpy.ok():
            with self.lock:
                msg, self.latest = self.latest, None
            if msg is None:
                time.sleep(0.005)
                continue
            try:
                self.process(msg)
            except Exception as e:                         # noqa: BLE001 - keep localizing
                self.get_logger().error(f"scan failed: {e!r}")

    def process(self, msg: PointCloud2):
        stamp = Time.from_msg(msg.header.stamp)
        if self.seed_time is not None and stamp.nanoseconds * 1e-9 < self.seed_time + 0.2:
            return                                         # scan from before the teleport
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

        odom = None
        if self.odom_frame:
            try:
                ot = self.tf_buf.lookup_transform(self.odom_frame, self.child, stamp,
                                                  timeout=rclpy.duration.Duration(seconds=0.3))
            except Exception:                              # noqa: BLE001 - odometry not there yet
                self.skipped += 1
                return
            Ro = quat_to_mat(ot.transform.rotation)
            odom = Pose(ot.transform.translation.x, ot.transform.translation.y,
                        math.atan2(Ro[1, 0], Ro[0, 0]), ot.transform.translation.z,
                        math.atan2(Ro[2, 1], Ro[2, 2]), math.asin(max(-1.0, min(1.0, -Ro[2, 0]))))

        t = stamp.nanoseconds * 1e-9
        with self.lock:
            prior = Pose(**vars(self.pose))
            dt = 0.0 if self.last_stamp is None else max(0.0, min(0.5, t - self.last_stamp))
            dyaw, self.gyro_yaw = self.gyro_yaw, 0.0
            up = self.up_imu
            search = self.need_search
            vel = self.vel.copy()
            gen = self.gen
            seeded = self.seed is not None and not self.fixed
            if seeded:
                prior = Pose(**vars(self.seed))
        tilt_err = 20.0 if search else 5.0
        if seeded:
            pass                                           # stay on the seed until the first fix
        elif odom is not None and self.corr is not None and (not search or self.bad > 0):
            # prediction: map <- odom correction applied to the odometry pose
            prior = self.apply_corr(odom)
            tilt_err = 4.0
        else:
            # prediction: constant velocity + gyro yaw
            prior.x += vel[0] * dt
            prior.y += vel[1] * dt
            prior.yaw = wrap(prior.yaw + dyaw)
        if odom is not None:
            prior.roll, prior.pitch, prior.z = odom.roll, odom.pitch, odom.z
        elif up is not None:
            prior.roll = float(math.atan2(up[1], up[2]))
            prior.pitch = float(math.asin(max(-1.0, min(1.0, -up[0]))))
            tilt_err = 8.0 if search else 4.0

        if search:
            # start / respawn / /initialpose: guess is good -> +-1 m, +-45 deg;
            # lost while tracking -> full yaw circle around the last pose, then the known spots
            lost = self.bad > 0 and not seeded
            if lost:
                res = self.loc.relocalize(pts, prior, radius=1.0, step=0.25, yaw_range_deg=180.0)
            else:                                          # seed is close: small, fast window
                res = self.loc.relocalize(pts, prior, radius=0.5, step=0.25, yaw_range_deg=20.0,
                                          yaw_step_deg=5.0)
                if not res.ok:
                    res = self.loc.relocalize(pts, prior, radius=1.0, step=0.25, yaw_range_deg=45.0)
            if lost and not res.ok:
                for sx, sy, syaw in KNOWN_SPOTS:
                    r2 = self.loc.relocalize(pts, Pose(sx, sy, math.radians(syaw), prior.z, prior.roll,
                                                       prior.pitch), radius=1.0, step=0.25, yaw_range_deg=60.0)
                    if r2.ok and (not res.ok or r2.inlier_ratio > res.inlier_ratio):
                        res = r2
            min_r = float(self.get_parameter("search_min_ratio").value)
            if res.ok and res.inlier_ratio < min_r:
                res = type(res)(res.pose, False, res.inlier_ratio, res.rmse, res.n_points, res.leveled)
            self.get_logger().info(
                f"search around ({prior.x:.2f}, {prior.y:.2f}, {math.degrees(prior.yaw):.0f} deg) "
                f"lost={lost} -> ok={res.ok} ({res.pose.x:.2f}, {res.pose.y:.2f}, "
                f"{math.degrees(res.pose.yaw):.0f} deg) ratio={res.inlier_ratio:.2f} "
                f"rmse={res.rmse * 100:.1f}cm leveled={res.leveled} pitch={math.degrees(res.pose.pitch):.1f} "
                f"z={res.pose.z:.2f} n={res.n_points}")
        else:
            res = self.loc.localize(pts, prior, tilt_err_deg=tilt_err)

        with self.lock:
            if gen != self.gen:                            # re-seeded while we worked: drop it
                return
            if res.ok:
                if self.last_stamp is not None and dt > 0 and not search:
                    v = np.array([res.pose.x - self.pose.x, res.pose.y - self.pose.y]) / dt
                    self.vel = 0.6 * self.vel + 0.4 * np.clip(v, -1.5, 1.5)
                self.pose, self.bad, self.need_search = res.pose, 0, False
                self.fixed = True
                if odom is not None:
                    cyaw = wrap(res.pose.yaw - odom.yaw)
                    c, s_ = math.cos(cyaw), math.sin(cyaw)
                    self.corr = (res.pose.x - (c * odom.x - s_ * odom.y),
                                 res.pose.y - (s_ * odom.x + c * odom.y), cyaw)
            else:
                self.bad += 1
                self.pose = prior                          # dead-reckon on the prediction
                self.vel *= 0.5
                if self.bad >= int(self.get_parameter("lost_after").value):
                    self.need_search = True
            self.last_stamp = t
            pose, bad = self.pose, self.bad
        if odom is not None:
            if self.corr is not None:
                self.publish_corr(msg.header.stamp)
                self.publish_pose_msg(pose, msg.header.stamp)
        else:
            self.publish(pose, msg.header.stamp)
        self.status_pub.publish(String(data=json.dumps({
            "ok": bool(res.ok), "inlier_ratio": round(res.inlier_ratio, 3),
            "rmse_cm": round(res.rmse * 100, 2), "points": res.n_points, "leveled": res.leveled,
            "bad_in_row": bad, "searching": search, "stamp": round(t, 4),
            "skipped_no_odom": self.skipped,
            "x": round(pose.x, 3), "y": round(pose.y, 3), "yaw_deg": round(math.degrees(pose.yaw), 2)})))

    # ------------------------------------------------------------------ outputs
    def apply_corr(self, odom: Pose) -> Pose:
        cx, cy, cyaw = self.corr
        c, s = math.cos(cyaw), math.sin(cyaw)
        return Pose(cx + c * odom.x - s * odom.y, cy + s * odom.x + c * odom.y,
                    wrap(cyaw + odom.yaw), odom.z, odom.roll, odom.pitch)

    def publish_corr(self, stamp):
        cx, cy, cyaw = self.corr
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, "map", self.odom_frame
        tf.transform.translation.x, tf.transform.translation.y = cx, cy
        tf.transform.rotation.z, tf.transform.rotation.w = math.sin(cyaw / 2), math.cos(cyaw / 2)
        self.tf_pub.sendTransform(tf)

    def publish_pose_msg(self, pose: Pose, stamp):
        T = pose.matrix()
        qx, qy, qz, qw = mat_to_quat(T[:3, :3])
        m = PoseWithCovarianceStamped()
        m.header.stamp, m.header.frame_id = stamp, "map"
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = T[:3, 3]
        o = m.pose.pose.orientation
        o.x, o.y, o.z, o.w = qx, qy, qz, qw
        self.pose_pub.publish(m)

    def publish(self, pose: Pose, stamp):
        T = pose.matrix()
        qx, qy, qz, qw = mat_to_quat(T[:3, :3])
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, "map", self.child
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = T[:3, 3]
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf_pub.sendTransform(tf)
        m = PoseWithCovarianceStamped()
        m.header = tf.header
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = T[:3, 3]
        o = m.pose.pose.orientation
        o.x, o.y, o.z, o.w = qx, qy, qz, qw
        self.pose_pub.publish(m)


def main():
    rclpy.init()
    node = IcpLocalizer()
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
