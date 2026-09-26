import unittest
import pandas as pd
from src.features import (
    name_features, address_features, postal_features, 
    numeric_features, country_features, build_pair_features
)

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.s1_base = pd.Series({
            "business_name_normalized": "amazon com inc",
            "business_name_core": "amazon",
            "business_name_tokens": ["amazon", "com", "inc"],
            "business_address_normalized": "123 main st seattle",
            "business_address_tokens": ["123", "main", "st", "seattle"],
            "business_address_postal": "98101",
            "business_address_numeric_tokens": ["123", "98101"],
            "country_normalized": "us"
        })
        
        self.s2_identical = self.s1_base.copy()
        
        self.s3_diff = pd.Series({
            "business_name_normalized": "google llc",
            "business_name_core": "google",
            "business_name_tokens": ["google", "llc"],
            "business_address_normalized": "1600 amphitheatre pkwy",
            "business_address_tokens": ["1600", "amphitheatre", "pkwy"],
            "business_address_postal": "94043",
            "business_address_numeric_tokens": ["1600", "94043"],
            "country_normalized": "us"
        })
        
        self.s2_missing = pd.Series({
            "business_name_normalized": "",
            "business_name_core": "",
            "business_name_tokens": [],
            "business_address_normalized": "",
            "business_address_tokens": [],
            "business_address_postal": "",
            "business_address_numeric_tokens": [],
            "country_normalized": ""
        })
        
        self.s2_typo = pd.Series({
            "business_name_normalized": "amzon com inc",
            "business_name_core": "amzon",
            "business_name_tokens": ["amzon", "com", "inc"],
            "business_address_normalized": "123 main street seattle",
            "business_address_tokens": ["123", "main", "street", "seattle"],
            "business_address_postal": "98101",
            "business_address_numeric_tokens": ["123", "98101"],
            "country_normalized": "us"
        })

    def test_identical_records(self):
        feats = build_pair_features(self.s1_base, self.s2_identical, "s2", ["exact_name"])
        
        self.assertEqual(feats['name_exact'], 1)
        self.assertEqual(feats['name_core_exact'], 1)
        self.assertAlmostEqual(feats['name_jaro_winkler'], 1.0)
        self.assertEqual(feats['name_token_jaccard'], 1.0)
        self.assertEqual(feats['address_exact'], 1)
        self.assertAlmostEqual(feats['address_levenshtein'], 1.0)
        self.assertEqual(feats['postal_exact_match'], 1)
        self.assertEqual(feats['postal_mismatch'], 0)
        self.assertEqual(feats['country_exact_match'], 1)
        
    def test_different_records(self):
        feats = build_pair_features(self.s1_base, self.s3_diff, "s3", [])
        
        self.assertEqual(feats['name_exact'], 0)
        self.assertLess(feats['name_jaro_winkler'], 0.5)
        self.assertEqual(feats['name_token_jaccard'], 0.0)
        self.assertEqual(feats['address_exact'], 0)
        self.assertEqual(feats['postal_exact_match'], 0)
        self.assertEqual(feats['postal_mismatch'], 1)

    def test_missing_values(self):
        feats = build_pair_features(self.s1_base, self.s2_missing, "s2", [])
        
        # Missing should NOT be a mismatch
        self.assertEqual(feats['postal_mismatch'], 0)
        self.assertEqual(feats['postal_exact_match'], 0)
        self.assertEqual(feats['postal_both_present'], 0)
        
        self.assertEqual(feats['name_missing_candidate'], 1)
        self.assertEqual(feats['name_exact'], 0)
        self.assertEqual(feats['name_jaro_winkler'], 0.0)

    def test_small_typo(self):
        feats = build_pair_features(self.s1_base, self.s2_typo, "s2", [])
        
        self.assertEqual(feats['name_exact'], 0)
        self.assertGreater(feats['name_jaro_winkler'], 0.8) # Should be high
        self.assertGreater(feats['address_levenshtein'], 0.7) # Should be high
        
        # Jaccard for tokens: ["amazon", "com", "inc"] vs ["amzon", "com", "inc"]
        # Intersection = {"com", "inc"} (2)
        # Union = {"amazon", "amzon", "com", "inc"} (4)
        self.assertEqual(feats['name_token_intersection_count'], 2)
        self.assertEqual(feats['name_token_jaccard'], 0.5)
        
        self.assertEqual(feats['postal_exact_match'], 1)

if __name__ == '__main__':
    unittest.main()
