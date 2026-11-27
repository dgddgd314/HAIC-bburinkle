"""Offline checks for color selection and depth projection."""
import itertools
import unittest

import cv2
import numpy as np

from custom.mission_2.cone_vision import cone_points


class ConeVisionTests(unittest.TestCase):
    k = (100, 0, 60, 0, 100, 40, 0, 0, 1)
    colors = {"blue": (26, 89, 199), "green": (33, 153, 71), "orange": (242, 89, 13)}

    def test_all_color_orders_and_lighting(self):
        """Find each requested cone independently of ordering and image brightness."""
        for order in itertools.permutations(self.colors):
            for light in (1, 0.5):
                rgb = np.full((80, 120, 3), 90, np.uint8)
                depth = np.full((80, 120), 3.0, np.float32)
                for i, name in enumerate(order):
                    x = 20 + 40 * i
                    triangle = np.array([(x, 25), (x - 7, 50), (x + 7, 50)])
                    cv2.fillConvexPoly(rgb, triangle, tuple(int(c * light) for c in self.colors[name]))
                for i, name in enumerate(order):
                    with self.subTest(order=order, name=name, light=light):
                        point, = cone_points(rgb, depth, self.k, name)
                        self.assertAlmostEqual(point[0], (20 + 40 * i - 60) * 3 / 100)
                        self.assertEqual(point[2], 3)

    def test_missing_color_and_invalid_depth(self):
        """A gray scene and a blue component without depth must produce no target."""
        rgb = np.full((80, 120, 3), 90, np.uint8)
        depth = np.ones((80, 120), np.float32)
        self.assertFalse(list(cone_points(rgb, depth, self.k, "blue")))
        rgb[30:50, 25:35] = self.colors["blue"]
        depth[30:50, 25:35] = np.nan
        self.assertFalse(list(cone_points(rgb, depth, self.k, "blue")))
