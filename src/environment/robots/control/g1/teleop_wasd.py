"""WASD keyboard teleop for the G1 bridge."""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from typing import Dict, Optional, Sequence, Tuple

HELP = """
  w/s forward/back   a/d turn in place   q/e strafe   space stop
  -/= speed          _/+ turn rate           Ctrl-C quit
"""

AXES: Dict[str, Tuple[float, float, float]] = {
    "w": (1.0, 0.0, 0.0),
    "s": (-1.0, 0.0, 0.0),
    "a": (0.0, 0.0, 1.0),
    "d": (0.0, 0.0, -1.0),
    "q": (0.0, 1.0, 0.0),
    "e": (0.0, -1.0, 0.0),
}


def read_key(timeout: float) -> Optional[str]:
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1) or None
    return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.parameter import Parameter
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topic", default="/cmd_vel")
    ap.add_argument("--rate", type=float, default=20.0,
                    help="publish rate; must stay well inside the bridge's "
                         "--cmd-vel-timeout")
    ap.add_argument("--speed", type=float, default=0.30,
                    help="linear speed cap. The policy tracks 0.3 well and the "
                         "bridge clamps at --max-vel regardless")
    ap.add_argument("--turn", type=float, default=0.40, help="yaw rate cap")
    ap.add_argument("--strafe", type=float, default=0.15,
                    help="lateral speed cap; the policy is least reliable sideways")
    ap.add_argument("--latch", action="store_true",
                    help="latch the command instead of holding it: a key sets a "
                         "speed that persists until another key changes it, and "
                         "space is the brake. The behaviour before hold-to-move")
    ap.add_argument("--arcs", action="store_true",
                    help="--latch only: latch each axis independently, so a turn "
                         "key added to a forward speed drives an arc. Off by "
                         "default: with it on, 'a' after 'w' holds vx and traces a "
                         "0.75 m circle instead of turning on the spot, which is "
                         "not what anyone means")
    ap.add_argument("--ramp", type=float, default=1.2,
                    help="seconds of holding to go from --ramp-start of the cap to "
                         "all of it. 0 starts at the cap")
    ap.add_argument("--ramp-start", type=float, default=0.35,
                    help="fraction of the cap the first instant of a press gets")
    ap.add_argument("--release", type=float, default=0.12,
                    help="silence that ends a hold once the terminal is repeating. "
                         "Must stay above the repeat interval -- 30 ms at the 33 Hz "
                         "in `xset q` -- or a held key stutters")
    ap.add_argument("--press-grace", type=float, default=0.60,
                    help="silence that ends a hold before the first repeat has "
                         "arrived. Must stay above the terminal's auto-repeat delay "
                         "(500 ms here) or every press stutters once; it is also "
                         "how long a single tap moves the robot")
    args = ap.parse_args(argv)

    if not 0.0 <= args.ramp_start <= 1.0:
        ap.error("--ramp-start is a fraction of the cap, so 0..1")

    rclpy.init()
    node = rclpy.create_node("teleop_wasd")
    node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    pub = node.create_publisher(
        Twist, args.topic,
        QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                   history=HistoryPolicy.KEEP_LAST))

    speed, turn, strafe = args.speed, args.turn, args.strafe
    vx = vy = wz = 0.0
    held = ""
    t_press = t_last = 0.0
    chars = 0
    frac = 0.0
    msg = Twist()
    old = termios.tcgetattr(sys.stdin)
    print(__doc__.split("\n\n")[0])
    print(HELP)
    print("  latched" if args.latch else
          f"  hold to move, ramping to the cap over {args.ramp:.1f}s")
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            k = read_key(1.0 / args.rate)
            now = time.monotonic()

            if k is None:
                break
            if k == "\x03":
                break
            if k == "-":
                speed = max(0.05, speed - 0.05)
                strafe = max(0.05, strafe - 0.05)
            elif k == "=":
                speed += 0.05
                strafe += 0.05
            elif k == "_":
                turn = max(0.05, turn - 0.05)
            elif k == "+":
                turn += 0.05
            elif k == " ":
                vx = vy = wz = 0.0
                held, chars = "", 0
            elif k and k.lower() in AXES:
                key = k.lower()
                if args.latch:
                    fx, fy, fz = AXES[key]
                    if args.arcs:
                        vx = fx * speed if fx else vx
                        vy = fy * strafe if fy else vy
                        wz = fz * turn if fz else wz
                    else:
                        vx, vy, wz = fx * speed, fy * strafe, fz * turn
                else:
                    if key != held:
                        held, t_press, chars = key, now, 0
                    chars += 1
                    t_last = now

            if held:
                # TTY has no key-up event; allow a longer gap before the first key repeat.
                gap = args.press_grace if chars < 2 else args.release
                if now - t_last > gap:
                    held, chars = "", 0

            if not args.latch:
                if held:
                    if args.ramp > 0.0:
                        t = (now - t_press) / args.ramp
                        frac = args.ramp_start + (1.0 - args.ramp_start) * min(1.0, t)
                    else:
                        frac = 1.0
                    fx, fy, fz = AXES[held]
                    vx, vy, wz = fx * speed * frac, fy * strafe * frac, fz * turn * frac
                else:
                    vx = vy = wz = frac = 0.0

            msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, wz
            pub.publish(msg)
            state = f"[{held or '-'} {frac * 100:3.0f}%]" if not args.latch else "[latched]"
            print(f"\r  vx {vx:+5.2f}  vy {vy:+5.2f}  wz {wz:+5.2f}  {state}   "
                  f"caps: speed {speed:.2f}  turn {turn:.2f}  strafe {strafe:.2f}   ",
                  end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        msg.linear.x = msg.linear.y = msg.angular.z = 0.0
        for _ in range(5):
            pub.publish(msg)
        print("\nstopped")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
