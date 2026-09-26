import unittest
import pandas as pd
from src.preprocessing import (
    normalize_text,
    normalize_business_name_advanced,
    get_business_name_core,
    normalize_business_address,
    extract_postal_code,
    extract_numeric_tokens,
    tokenize_text,
    apply_preprocessing
)

class TestPreprocessing(unittest.TestCase):
    
    def test_normalize_text(self):
        # Empty and missing
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text(float('nan')), "")
        
        # Unicode, Case, Whitespace, Punctuation
        self.assertEqual(normalize_text("  CaFé  "), "cafe")
        self.assertEqual(normalize_text("Hello, World!"), "hello world")
        
    def test_normalize_business_name_advanced(self):
        # Ampersand
        self.assertEqual(normalize_business_name_advanced("A & B Corp"), "a and b corp")
        self.assertEqual(normalize_business_name_advanced("A&B"), "a and b")
        
    def test_get_business_name_core(self):
        self.assertEqual(get_business_name_core("abc technologies ltd"), "abc technologies")
        self.assertEqual(get_business_name_core("abc technologies pvt ltd"), "abc technologies")
        self.assertEqual(get_business_name_core("abc inc corp limited"), "abc")
        self.assertEqual(get_business_name_core("apple"), "apple")
        
    def test_normalize_business_address(self):
        self.assertEqual(normalize_business_address("123 Main St., Apt 4B"), "123 main street apt 4b")
        self.assertEqual(normalize_business_address("Park Ave"), "park avenue")
        
    def test_extract_postal_code(self):
        self.assertEqual(extract_postal_code("Pune 411001 India"), "411001")
        self.assertEqual(extract_postal_code("NY 10001-1234"), "10001")
        self.assertEqual(extract_postal_code("No postal code here"), "")
        
    def test_extract_numeric_tokens(self):
        self.assertEqual(extract_numeric_tokens("12 Main St, 411001"), ["12", "411001"])
        self.assertEqual(extract_numeric_tokens("No numbers"), [])
        
    def test_tokenize_text(self):
        self.assertEqual(tokenize_text("hello world"), ["hello", "world"])
        
    def test_idempotency(self):
        # Test idempotency
        name = "A & B Technologies Pvt. Ltd."
        norm1 = normalize_business_name_advanced(name)
        norm2 = normalize_business_name_advanced(norm1)
        self.assertEqual(norm1, norm2)
        
        core1 = get_business_name_core(norm1)
        core2 = get_business_name_core(core1)
        self.assertEqual(core1, core2)
        
        addr = "123 Main St."
        addr1 = normalize_business_address(addr)
        addr2 = normalize_business_address(addr1)
        self.assertEqual(addr1, addr2)

    def test_apply_preprocessing_preserves_raw(self):
        df = pd.DataFrame({
            "entity_id": ["S1-01"],
            "business_name": ["Apple Inc."],
            "business_address": ["1 Infinite Loop, 95014"],
            "country": ["USA"]
        })
        out_df = apply_preprocessing(df)
        
        # Raw fields must remain untouched
        self.assertEqual(out_df["business_name"].iloc[0], "Apple Inc.")
        self.assertEqual(out_df["business_address"].iloc[0], "1 Infinite Loop, 95014")
        self.assertEqual(out_df["country"].iloc[0], "USA")
        
        # Check derived
        self.assertEqual(out_df["business_name_normalized"].iloc[0], "apple inc")
        self.assertEqual(out_df["business_name_core"].iloc[0], "apple")
        self.assertEqual(out_df["business_address_normalized"].iloc[0], "1 infinite loop 95014")
        self.assertEqual(out_df["business_address_postal"].iloc[0], "95014")

if __name__ == "__main__":
    unittest.main()
