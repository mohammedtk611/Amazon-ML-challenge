import unittest
import pandas as pd
from src.blocking import (
    ExactNameBlock, PostalBlock, combine_candidates
)

class TestBlocking(unittest.TestCase):
    
    def setUp(self):
        self.s2 = pd.DataFrame({
            "entity_id": ["S2-1", "S2-2", "S2-3"],
            "business_name_normalized": ["apple inc", "banana ltd", "apple inc"],
            "business_name_core": ["apple", "banana", "apple"],
            "business_address_postal": ["10001", "20002", ""]
        })
        self.s3 = pd.DataFrame({
            "entity_id": ["S3-1", "S3-2"],
            "business_name_normalized": ["apple inc", "cherry"],
            "business_name_core": ["apple", "cherry"],
            "business_address_postal": ["10001", "30003"]
        })
        self.s1_row = pd.Series({
            "entity_id": "S1-1",
            "business_name_normalized": "apple inc",
            "business_name_core": "apple",
            "business_address_postal": "10001"
        })
        self.s1_empty = pd.Series({
            "entity_id": "S1-2",
            "business_name_normalized": "",
            "business_name_core": "",
            "business_address_postal": ""
        })

    def test_exact_name_block(self):
        block = ExactNameBlock()
        block.build_index(self.s2, self.s3)
        cands = block.get_candidates(self.s1_row)
        
        self.assertIn("S2-1", cands)
        self.assertIn("S2-3", cands)
        self.assertIn("S3-1", cands)
        self.assertNotIn("S2-2", cands)
        
        # Test empty row
        empty_cands = block.get_candidates(self.s1_empty)
        self.assertEqual(len(empty_cands), 0)

    def test_postal_block(self):
        block = PostalBlock()
        block.build_index(self.s2, self.s3)
        cands = block.get_candidates(self.s1_row)
        
        self.assertIn("S2-1", cands)
        self.assertIn("S3-1", cands)
        self.assertNotIn("S2-3", cands) # postal is empty

    def test_combine_candidates(self):
        cands_dict = {
            "exact_name": ["S2-1", "S3-1", "S2-3"],
            "postal": ["S2-1", "S3-1", "S2-99"]
        }
        
        final_cands, prov = combine_candidates(cands_dict)
        
        # Test unique
        self.assertEqual(len(final_cands), 4)
        
        # Test determinism
        self.assertEqual(final_cands, ["S2-1", "S3-1", "S2-3", "S2-99"])
        
        # Test provenance
        self.assertEqual(set(prov["S2-1"]), {"exact_name", "postal"})
        self.assertEqual(prov["S2-3"], ["exact_name"])
        
    def test_max_candidates_cap(self):
        import src.blocking
        src.blocking.MAX_CANDIDATES_PER_SOURCE = 2
        
        cands_dict = {
            "exact_name": ["S2-1", "S3-1", "S2-3"],
            "postal": ["S2-1", "S3-1", "S2-99"]
        }
        
        final_cands, prov = combine_candidates(cands_dict)
        self.assertEqual(len(final_cands), 2)
        # S2-1 and S3-1 have higher combined weight
        self.assertIn("S2-1", final_cands)
        self.assertIn("S3-1", final_cands)

if __name__ == "__main__":
    unittest.main()
