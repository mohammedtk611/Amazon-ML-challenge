import unittest
import pandas as pd
import numpy as np

from src.training import (
    create_entity_split, build_ground_truth_index, label_candidate_pairs,
    sample_negative_pairs
)

class TestPhase5(unittest.TestCase):
    def setUp(self):
        self.s1 = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2", "S1-3", "S1-4", "S1-5", "S1-6", "S1-7", "S1-8", "S1-9", "S1-10"],
            "business_name_normalized": ["apple"] * 10
        })
        self.s2 = pd.DataFrame({"entity_id": ["S2-1", "S2-2"], "business_name_normalized": ["apple", "banana"]})
        self.s3 = pd.DataFrame({"entity_id": ["S3-1"], "business_name_normalized": ["cherry"]})
        
        self.gt = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "matched_entity_ids": ["S2-1,S3-1", "S2-2"]
        })

    def test_entity_split(self):
        train, val = create_entity_split(self.s1, test_size=0.2, random_state=42)
        
        # Check size
        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 2)
        
        # Check no overlap
        overlap = set(train['entity_id']).intersection(set(val['entity_id']))
        self.assertEqual(len(overlap), 0)

    def test_label_candidates(self):
        gt_index = build_ground_truth_index(self.gt)
        
        cands = {
            "S1-1": (["S2-1", "S2-2"], {}),
            "S1-2": (["S3-1"], {}) # missed S2-2 entirely
        }
        
        pos, gt_count, missing = label_candidate_pairs(cands, gt_index)
        
        # S1-1 matched S2-1 (pos), missed S3-1 (missing)
        # S1-2 missed S2-2 (missing)
        
        self.assertEqual(gt_count, 3) # S1-1 has 2 matches, S1-2 has 1
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos[0]["s1_id"], "S1-1")
        self.assertEqual(pos[0]["cand_id"], "S2-1")
        self.assertEqual(missing, 2) # S3-1 for S1-1, and S2-2 for S1-2

    def test_negative_sampling(self):
        gt_index = build_ground_truth_index(self.gt)
        
        cands = {
            "S1-1": (["S2-1", "S2-2", "S3-1"], {}) # S2-1 and S3-1 are True Matches
        }
        
        negs = sample_negative_pairs(cands, gt_index, self.s1, self.s2, self.s3, num_random=2)
        
        # Only S2-2 is a negative
        self.assertTrue(all(n['label'] == 0 for n in negs))
        
        # S2-1 and S3-1 must NOT be in negs
        cand_ids = [n['cand_id'] for n in negs]
        self.assertIn("S2-2", cand_ids)
        self.assertNotIn("S2-1", cand_ids)
        self.assertNotIn("S3-1", cand_ids)
        
        # Test uniqueness
        self.assertEqual(len(negs), len(set(cand_ids)))

if __name__ == '__main__':
    unittest.main()
