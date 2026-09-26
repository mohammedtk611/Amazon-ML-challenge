import unittest
import pandas as pd
from src.threshold_optimization import calculate_entity_metrics, evaluate_predictions, apply_threshold

class TestPhase6(unittest.TestCase):
    def test_calculate_entity_metrics(self):
        # 1. Empty actual, empty predicted
        p, r, f = calculate_entity_metrics(set(), set())
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(f, 1.0)
        
        # 2. Empty actual, non-empty predicted (False Match)
        p, r, f = calculate_entity_metrics({"S2-1"}, set())
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f, 0.0)
        
        # 3. Non-empty actual, empty predicted (Missed Match)
        p, r, f = calculate_entity_metrics(set(), {"S2-1"})
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f, 0.0)
        
        # 4. Partial overlap
        pred = {"S2-1", "S2-2"}
        actual = {"S2-1", "S3-1"}
        p, r, f = calculate_entity_metrics(pred, actual)
        # TP = 1 (S2-1), FP = 1 (S2-2), FN = 1 (S3-1)
        self.assertEqual(p, 0.5)
        self.assertEqual(r, 0.5)
        # F0.5 = (1.25 * 0.25) / (0.125 + 0.5) = 0.3125 / 0.625 = 0.5
        self.assertEqual(f, 0.5)

    def test_evaluate_predictions(self):
        predictions = {
            "S1-1": {"S2-1"},
            "S1-2": {"S2-2"},
            "S1-3": set()
        }
        ground_truth = {
            "S1-1": {"S2-1"}, # TP
            "S1-2": {"S3-1"}, # FP and FN
            "S1-3": set()     # True Negative (p=1, r=1, f=1)
        }
        
        metrics = evaluate_predictions(predictions, ground_truth)
        # S1-1: p=1, r=1, f=1
        # S1-2: p=0, r=0, f=0
        # S1-3: p=1, r=1, f=1
        
        self.assertAlmostEqual(metrics['precision'], 2.0 / 3)
        self.assertAlmostEqual(metrics['recall'], 2.0 / 3)
        self.assertAlmostEqual(metrics['f05'], 2.0 / 3)

    def test_apply_threshold(self):
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2", "S2-3"],
            "match_probability": [0.9, 0.4, 0.6]
        })
        
        # threshold 0.5
        preds = apply_threshold(df, 0.5)
        self.assertIn("S1-1", preds)
        self.assertIn("S1-2", preds)
        self.assertEqual(preds["S1-1"], {"S2-1"})
        self.assertEqual(preds["S1-2"], {"S2-3"})
        
        # threshold 0.95
        preds2 = apply_threshold(df, 0.95)
        self.assertEqual(preds2["S1-1"], set())
        self.assertEqual(preds2["S1-2"], set())

if __name__ == '__main__':
    unittest.main()
