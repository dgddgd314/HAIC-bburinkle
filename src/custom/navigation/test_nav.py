"""ROS-free regression tests for the navigation controller."""
import math
import unittest
from unittest.mock import Mock, patch

from custom.navigation.nav import GridMap, Navigator


class NavigationTests(unittest.TestCase):
    def setUp(self):
        """Record velocity and stop commands without a running simulator."""
        self.mission = Mock()
        self.nav = Navigator(self.mission, grid=Mock())

    def test_unready_localization_with_existing_pose(self):
        """A stale TF must not authorize walking after the readiness timeout."""
        self.mission.pose.return_value = (0, 0, 0)
        self.nav.grid.plan.return_value = [(0, 0), (2, 0)]
        with patch.object(self.nav, "wait_localized", return_value=False):
            self.assertFalse(self.nav.go_to(2, 0, timeout=0.01))
        self.mission.stop.assert_called()
        self.mission.cmd_vel.assert_not_called()
        self.nav.grid.plan.assert_not_called()

    def test_deviation_near_later_segment(self):
        """Drifting across a switchback must not select its return leg."""
        path = [(0, 0), (4, 0), (4, 2), (0, 2)]
        self.mission.pose.return_value = (1, 1.8, math.pi)
        self.assertFalse(self.nav.follow(path, tol=0.15, timeout=0.1))
        self.mission.stop.assert_called()
        self.mission.cmd_vel.assert_not_called()

    def test_straight_corner_and_short_final_segment(self):
        """Track the current leg, turn at the corner, and reach a nearby endpoint."""
        path = [(0, 0), (4, 0), (4, 2)]
        self.assertEqual(self.nav._lookahead(path, 0, 0, 0), (0.6, 0))
        self.assertEqual(self.nav._lookahead(path, 1, 4, 0), (4, 0.6))
        self.assertEqual(self.nav._lookahead(path, 1, 4, 1.8), path[-1])

    def test_off_path_with_no_intersection(self):
        """An empty lookahead circle must not fall back to the final goal."""
        self.assertIsNone(self.nav._lookahead([(0, 0), (4, 0), (4, 2)], 0, 1, -1))

    def test_turns_keep_commanded_trajectory_in_free_cells(self):
        """Replay the failing corner route with ideal body-frame velocities at 20 Hz."""
        grid = GridMap()
        nav = Navigator(self.mission, grid)
        x, y, yaw = 12.5, 20.1, 0.0
        goal = (6.05, 18.8)
        path = grid.plan((x, y), goal)

        def step(vx, vy, wz):
            """Integrate one controller tick and check the traversed map cell."""
            nonlocal x, y, yaw
            dt = 0.05
            x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
            y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
            yaw += wz * dt
            self.assertTrue(grid.free(*grid.to_cell(x, y)), (x, y))

        self.mission.pose.side_effect = lambda timeout: (x, y, yaw)
        self.mission.cmd_vel.side_effect = step
        with patch("custom.navigation.nav.time.sleep"):
            self.assertTrue(nav.follow(path, tol=0.15, timeout=8))
        self.assertLess(math.hypot(x - goal[0], y - goal[1]), 0.15)
