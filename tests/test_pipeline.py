import itertools
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from data_utils.file_filters import is_single_person, is_multi_person
from data_utils.RadarFallDataLoader import pc_normalize
from debug_tools.track_debugger import best_assignment, refine_tracks
from predict import ROOT_DIR, load_model, prepare_points
from target_inference_tools.run_single_model_on_multiperson import sample_window_points


def brute_force_assignment(costs, gate):
    for count in range(min(costs.shape), 0, -1):
        totals = []
        for rows in itertools.combinations(range(costs.shape[0]), count):
            for cols in itertools.permutations(range(costs.shape[1]), count):
                pairs = list(zip(rows, cols))
                if all(costs[t, d] <= gate for t, d in pairs):
                    totals.append(sum(float(costs[t, d]) for t, d in pairs))
        if totals:
            return count, min(totals)
    return 0, 0.0


class PipelineTests(unittest.TestCase):
    def test_person_count_filter_ignores_action_description(self):
        name = '10.三人前后进，一人坠床_frames.mat'
        self.assertTrue(is_multi_person(name))
        self.assertFalse(is_single_person(name))
        self.assertTrue(is_single_person('1.一人负类_frames.mat'))

    def test_assignment_preserves_gated_optimum(self):
        rng = np.random.default_rng(42)
        for rows in range(1, 5):
            for cols in range(1, 5):
                for _ in range(5):
                    costs = rng.uniform(0, 2, (rows, cols))
                    pairs = best_assignment(costs, 0.9)
                    count, total = brute_force_assignment(costs, 0.9)
                    self.assertEqual(len(pairs), count)
                    self.assertAlmostEqual(sum(costs[t, d] for t, d in pairs), total)
        self.assertEqual(best_assignment(np.empty((0, 3)), 0.9), [])

    def test_split_track_ids_remain_unique(self):
        tracks = []
        # These parent IDs collided with the old decimal-concatenation IDs.
        for track_id, count in [(1, 11), (11, 1)]:
            history = [{'frame_idx': frame, 'center_2d': np.array([2.0 * frame, 0.0]),
                        'num_points': 6} for frame in range(count)]
            tracks.append(SimpleNamespace(track_id=track_id, history=history))
        refined = refine_tracks(tracks, max_step_dist=1.1, min_track_len=1)
        self.assertEqual(len(refined), 12)
        self.assertEqual(len({t.track_id for t in refined}), 12)

    def test_empty_and_constant_points_are_finite(self):
        self.assertTrue(np.isfinite(pc_normalize(np.ones((192, 3)))).all())
        empty = sample_window_points([{'points': np.empty((0, 5))}], 192)
        self.assertEqual(empty.shape, (192, 3))
        self.assertTrue(np.isfinite(empty).all())
        with self.assertRaises(ValueError):
            prepare_points(np.empty((0, 3)))

    def test_reference_checkpoints_run_on_cpu(self):
        torch.set_num_threads(2)
        points = torch.randn(1, 3, 192)
        for name in ['fall_ssg_reference.pth', 'multi_person_ssg_smoke.pth']:
            model, _ = load_model(ROOT_DIR / 'weights' / name, 'cpu')
            with torch.no_grad():
                scores, _ = model(points)
            self.assertEqual(tuple(scores.shape), (1, 2))
            self.assertTrue(torch.isfinite(scores).all())
            torch.testing.assert_close(scores.exp().sum(1), torch.ones(1))


if __name__ == '__main__':
    unittest.main()
