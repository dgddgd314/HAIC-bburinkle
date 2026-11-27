"""ROS-free regression tests for the navigation controller."""
import math
import unittest
from unittest.mock import Mock, patch

from custom.navigation.nav import Navigator


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
