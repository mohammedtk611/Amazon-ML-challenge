import re
import unicodedata
import pandas as pd
from typing import List

def normalize_text(text: str) -> str:
    """Generic text normalization: unicode, lowercase, punctuation, whitespace."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    # Unicode normalization to remove accents
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    # Lowercase
    text = text.lower()
    # Basic punctuation to space (conservative)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_country(country: str) -> str:
    """Normalize country open-set string."""
    return normalize_text(country)

def normalize_business_name(name: str) -> str:
    """Conservative business name normalization."""
    # Using the generic normalize_text handles lowercase, punctuation, and whitespace
    name = normalize_text(name)
    # Expand "&" if we hadn't stripped it, but we stripped punctuation.
    # Wait, & is punctuation. If we want to keep "&" -> "and", we should do it before stripping punctuation.
    return name

def normalize_business_name_advanced(name: str) -> str:
    """Preserve specific characters before general text normalization."""
    if pd.isna(name) or name is None:
        return ""
    name = str(name).lower()
    name = name.replace('&', ' and ')
    return normalize_text(name)

def get_business_name_core(normalized_name: str) -> str:
    """Remove common legal suffixes for a core representation."""
    suffixes = r'\b(inc|llc|corp|corporation|ltd|limited|pvt|private|co|company)\b'
    core = normalized_name
    prev = ""
    # Strip suffixes iteratively from the end
    while core != prev:
        prev = core
        core = re.sub(suffixes + r'$', '', core).strip()
    return core

def normalize_business_address(address: str) -> str:
    """Normalize addresses, expanding common conservative abbreviations."""
    addr = normalize_text(address)
    # Conservative substitutions based on word boundaries
    addr = re.sub(r'\brd\b', 'road', addr)
    addr = re.sub(r'\bst\b', 'street', addr)
    addr = re.sub(r'\bave\b', 'avenue', addr)
    return addr

def extract_postal_code(address: str) -> str:
    """Extract 5 or 6 digit postal/PIN-like codes."""
    if pd.isna(address) or address is None:
        return ""
    address = str(address)
    # Look for 5 to 6 consecutive digits bounded by non-digits/string ends
    matches = re.findall(r'\b\d{5,6}\b', address)
    if matches:
        return matches[-1] # Usually postal code is towards the end of an address
    return ""

def extract_numeric_tokens(address: str) -> List[str]:
    """Extract all numeric tokens from an address."""
    if pd.isna(address) or address is None:
        return []
    address = str(address)
    return re.findall(r'\b\d+\b', address)

def tokenize_text(text: str) -> List[str]:
    """Basic whitespace tokenization."""
    if not text:
        return []
    return text.split()

def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all normalization functions to generate derived columns without altering originals."""
    df = df.copy()
    
    # Name
    df['business_name_normalized'] = df['business_name'].apply(normalize_business_name_advanced)
    df['business_name_core'] = df['business_name_normalized'].apply(get_business_name_core)
    df['business_name_tokens'] = df['business_name_normalized'].apply(tokenize_text)
    
    # Address
    df['business_address_normalized'] = df['business_address'].apply(normalize_business_address)
    df['business_address_postal'] = df['business_address'].apply(extract_postal_code)
    df['business_address_numeric_tokens'] = df['business_address'].apply(extract_numeric_tokens)
    df['business_address_tokens'] = df['business_address_normalized'].apply(tokenize_text)
    
    # Country
    df['country_normalized'] = df['country'].apply(normalize_country)
    
    return df
