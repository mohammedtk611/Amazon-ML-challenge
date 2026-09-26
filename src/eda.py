import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import re

def validate_files(data_dict: Dict[str, pd.DataFrame], prefix_map: Dict[str, str], title_prefix: str) -> List[str]:
    lines = []
    for name, df in data_dict.items():
        if name == "ground_truth":
            continue
        expected_prefix = prefix_map.get(name, "")
        
        n_rows = len(df)
        n_dups = df['entity_id'].duplicated().sum()
        n_empty_id = (df['entity_id'] == "").sum()
        n_empty_name = (df['business_name'] == "").sum()
        n_empty_addr = (df['business_address'] == "").sum()
        n_empty_country = (df['country'] == "").sum()
        
        # prefix check
        if expected_prefix:
            invalid_prefix_count = (~df['entity_id'].str.startswith(expected_prefix)).sum()
        else:
            invalid_prefix_count = 0
            
        lines.append(f"#### {title_prefix} {name}")
        lines.append(f"- **Rows**: {n_rows}")
        lines.append(f"- **Duplicate entity_id**: {n_dups}")
        lines.append(f"- **Empty entity_id**: {n_empty_id}")
        lines.append(f"- **Empty business_name**: {n_empty_name}")
        lines.append(f"- **Empty business_address**: {n_empty_addr}")
        lines.append(f"- **Empty country**: {n_empty_country}")
        if expected_prefix:
            lines.append(f"- **IDs lacking prefix '{expected_prefix}'**: {invalid_prefix_count}")
        lines.append("")
    return lines

def generate_file_validation(train_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame]) -> str:
    lines = ["## 2. File Validation\n"]
    prefix_map = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}
    
    lines.extend(validate_files(train_data, prefix_map, "Train"))
    lines.extend(validate_files(test_data, prefix_map, "Test"))
    
    return "\n".join(lines)

def calc_text_stats(series: pd.Series) -> Dict[str, Any]:
    lengths = series[series != ""].str.len()
    if len(lengths) == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": round(float(lengths.mean()), 2),
        "median": float(lengths.median())
    }

def generate_source_stats(train_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame]) -> str:
    lines = ["## 3. Source Statistics & 4. Missing Data Analysis\n"]
    
    datasets = {**{f"Train {k}": v for k, v in train_data.items() if k != "ground_truth"},
                **{f"Test {k}": v for k, v in test_data.items()}}
                
    for name, df in datasets.items():
        n_rows = len(df)
        cols = len(df.columns)
        n_unique_ids = df['entity_id'].nunique()
        
        lines.append(f"### {name}")
        lines.append(f"- **Rows**: {n_rows}")
        lines.append(f"- **Columns**: {cols}")
        lines.append(f"- **Unique entity_ids**: {n_unique_ids}")
        lines.append("")
        
        lines.append("#### Text Lengths (ignoring empty strings)")
        for col in ["business_name", "business_address", "country"]:
            n_empty = (df[col] == "").sum()
            pct_empty = (n_empty / n_rows * 100) if n_rows > 0 else 0
            stats = calc_text_stats(df[col])
            
            lines.append(f"- **{col}**: {n_empty} empty ({pct_empty:.2f}%) | "
                         f"Len: Min {stats['min']}, Max {stats['max']}, Mean {stats['mean']}, Median {stats['median']}")
        lines.append("")
        
    return "\n".join(lines)

def generate_country_analysis(train_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame]) -> str:
    lines = ["## 5. Country Analysis\n"]
    
    train_countries = set()
    test_countries = set()
    
    datasets = {**{f"Train {k}": v for k, v in train_data.items() if k != "ground_truth"},
                **{f"Test {k}": v for k, v in test_data.items()}}
                
    for name, df in datasets.items():
        lines.append(f"### {name}")
        val_counts = df['country'].value_counts(dropna=False)
        total = len(df)
        
        for country, count in val_counts.items():
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"- {country if country else '<EMPTY>'}: {count} ({pct:.2f}%)")
            
            if "Train" in name and country:
                train_countries.add(country)
            elif "Test" in name and country:
                test_countries.add(country)
        lines.append("")
        
    lines.append("### Country Differences")
    test_only = test_countries - train_countries
    if test_only:
        lines.append(f"- **WARNING**: Found countries in Test that are NOT in Train: {', '.join(test_only)}")
    else:
        lines.append("- No new countries found in Test set.")
        
    lines.append("\n")
    return "\n".join(lines)

def generate_ground_truth_analysis(train_gt: pd.DataFrame) -> str:
    lines = ["## 8. Ground Truth Match Distribution\n"]
    
    if len(train_gt) == 0:
        lines.append("Ground truth is empty.\n")
        return "\n".join(lines)
        
    def count_matches(match_str):
        if not match_str:
            return 0
        return len(match_str.split(","))
        
    matches_list = train_gt['matched_entity_ids'].fillna("").apply(count_matches)
    
    n_total = len(train_gt)
    n_singletons = (matches_list == 0).sum()
    n_matched = (matches_list > 0).sum()
    
    lines.append(f"- **Total Source 1 entities**: {n_total}")
    lines.append(f"- **Singleton (no match)**: {n_singletons}")
    lines.append(f"- **Entities with >=1 match**: {n_matched}")
    
    if n_total > 0:
        lines.append(f"- **Match counts**: Min {matches_list.min()}, Max {matches_list.max()}, Mean {matches_list.mean():.2f}, Median {matches_list.median()}")
    
    # Source 2 vs Source 3
    s2_counts = train_gt['matched_entity_ids'].fillna("").apply(lambda x: sum(1 for v in x.split(",") if v.startswith("S2-")))
    s3_counts = train_gt['matched_entity_ids'].fillna("").apply(lambda x: sum(1 for v in x.split(",") if v.startswith("S3-")))
    
    total_s2_matches = s2_counts.sum()
    total_s3_matches = s3_counts.sum()
    
    only_s2 = ((s2_counts > 0) & (s3_counts == 0)).sum()
    only_s3 = ((s2_counts == 0) & (s3_counts > 0)).sum()
    both = ((s2_counts > 0) & (s3_counts > 0)).sum()
    
    lines.append("")
    lines.append("#### Matches Breakdown")
    lines.append(f"- **Total S2 matches across all S1**: {total_s2_matches}")
    lines.append(f"- **Total S3 matches across all S1**: {total_s3_matches}")
    lines.append(f"- **S1 with only S2 matches**: {only_s2}")
    lines.append(f"- **S1 with only S3 matches**: {only_s3}")
    lines.append(f"- **S1 with both S2 and S3 matches**: {both}")
    lines.append(f"- **S1 with no matches**: {n_singletons}")
    
    lines.append("\n")
    return "\n".join(lines)


def analyze_business_names(train_data: Dict[str, pd.DataFrame]) -> str:
    lines = ["## 6. Business Name Analysis\n"]
    
    df = pd.concat([train_data["source1"], train_data["source2"]], ignore_index=True)
    names = df['business_name'].fillna("")
    
    lines.append(f"- **Total non-empty names analyzed**: {(names != '').sum()}")
    
    lengths = names.str.len()
    short_names = names[(lengths > 0) & (lengths <= 5)].unique()
    long_names = names[lengths > 50].unique()
    
    lines.append(f"- **Very short names (<= 5 chars)**: {len(short_names)} unique examples. E.g., {list(short_names)[:5]}")
    lines.append(f"- **Very long names (> 50 chars)**: {len(long_names)} unique examples. E.g., {list(long_names)[:2]}")
    
    punct_mask = names.str.contains(r'[^\w\s]', regex=True, na=False)
    lines.append(f"- **Names containing punctuation**: {punct_mask.sum()} ({(punct_mask.sum() / len(names) * 100):.2f}%)")
    
    num_mask = names.str.contains(r'\d', regex=True, na=False)
    lines.append(f"- **Names containing numbers**: {num_mask.sum()} ({(num_mask.sum() / len(names) * 100):.2f}%)")
    
    legal_terms = ["inc", "llc", "corp", "ltd", "pvt", "limited", "co", "company"]
    lines.append("#### Common Legal Terms Frequency (approx, case-insensitive substring)")
    names_lower = names.str.lower()
    for term in legal_terms:
        term_mask = names_lower.str.contains(rf'\b{term}\b', regex=True, na=False)
        lines.append(f"- **{term}**: {term_mask.sum()} occurrences")
    
    lines.append("\n")
    return "\n".join(lines)

def analyze_addresses(train_data: Dict[str, pd.DataFrame]) -> str:
    lines = ["## 7. Address Analysis\n"]
    
    df = pd.concat([train_data["source1"], train_data["source2"]], ignore_index=True)
    addrs = df['business_address'].fillna("")
    
    lines.append(f"- **Total non-empty addresses analyzed**: {(addrs != '').sum()}")
    
    lengths = addrs.str.len()
    short_addrs = addrs[(lengths > 0) & (lengths <= 10)].unique()
    long_addrs = addrs[lengths > 100].unique()
    
    lines.append(f"- **Very short addresses (<= 10 chars)**: {len(short_addrs)} unique examples. E.g., {list(short_addrs)[:5]}")
    lines.append(f"- **Very long addresses (> 100 chars)**: {len(long_addrs)} unique examples. E.g., {list(long_addrs)[:2]}")
    
    num_mask = addrs.str.contains(r'\d', regex=True, na=False)
    lines.append(f"- **Addresses containing numbers**: {num_mask.sum()} ({(num_mask.sum() / len(addrs) * 100):.2f}%)")
    
    postal_mask = addrs.str.contains(r'\b\d{5,6}\b', regex=True, na=False)
    lines.append(f"- **Addresses with 5-6 digit PIN/ZIP patterns**: {postal_mask.sum()} ({(postal_mask.sum() / len(addrs) * 100):.2f}%)")
    
    punct_mask = addrs.str.contains(r'[^\w\s]', regex=True, na=False)
    lines.append(f"- **Addresses containing punctuation**: {punct_mask.sum()} ({(punct_mask.sum() / len(addrs) * 100):.2f}%)")
    
    lines.append("\n")
    return "\n".join(lines)

def analyze_cross_source_overlap(train_data: Dict[str, pd.DataFrame]) -> str:
    lines = ["## 9. Cross-Source Overlap\n"]
    
    s1 = train_data["source1"]
    s2 = train_data["source2"]
    
    s1_names = set(s1[s1['business_name'] != ""]['business_name'].unique())
    s2_names = set(s2[s2['business_name'] != ""]['business_name'].unique())
    name_overlap = s1_names.intersection(s2_names)
    
    s1_addrs = set(s1[s1['business_address'] != ""]['business_address'].unique())
    s2_addrs = set(s2[s2['business_address'] != ""]['business_address'].unique())
    addr_overlap = s1_addrs.intersection(s2_addrs)
    
    lines.append(f"- **Unique exact names in S1**: {len(s1_names)}")
    lines.append(f"- **Unique exact names in S2**: {len(s2_names)}")
    lines.append(f"- **Exact name overlap S1/S2**: {len(name_overlap)}")
    
    lines.append(f"- **Unique exact addresses in S1**: {len(s1_addrs)}")
    lines.append(f"- **Unique exact addresses in S2**: {len(s2_addrs)}")
    lines.append(f"- **Exact address overlap S1/S2**: {len(addr_overlap)}")
    
    lines.append("\n")
    return "\n".join(lines)

def generate_true_match_examples(train_data: Dict[str, pd.DataFrame]) -> str:
    lines = ["## 10. True Match Examples\n"]
    
    gt = train_data.get("ground_truth", pd.DataFrame())
    if len(gt) == 0:
        return "\n".join(lines) + "No ground truth available.\n"
        
    s1 = train_data["source1"].set_index("entity_id")
    s2 = train_data["source2"].set_index("entity_id")
    s3 = train_data["source3"].set_index("entity_id")
    
    examples_shown = 0
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matches = [m for m in str(row["matched_entity_ids"]).split(",") if m]
        if not matches:
            continue
            
        if s1_id in s1.index:
            s1_row = s1.loc[s1_id]
            for m in matches[:2]: 
                if m.startswith("S2-") and m in s2.index:
                    m_row = s2.loc[m]
                    lines.append(f"**Match**: {s1_id} <-> {m}")
                    lines.append(f"- **S1 Name**: {s1_row['business_name']} | **S2 Name**: {m_row['business_name']}")
                    lines.append(f"- **S1 Addr**: {s1_row['business_address']} | **S2 Addr**: {m_row['business_address']}")
                    lines.append(f"- **S1 Country**: {s1_row['country']} | **S2 Country**: {m_row['country']}")
                    lines.append("")
                    examples_shown += 1
                elif m.startswith("S3-") and m in s3.index:
                    m_row = s3.loc[m]
                    lines.append(f"**Match**: {s1_id} <-> {m}")
                    lines.append(f"- **S1 Name**: {s1_row['business_name']} | **S3 Name**: {m_row['business_name']}")
                    lines.append(f"- **S1 Addr**: {s1_row['business_address']} | **S3 Addr**: {m_row['business_address']}")
                    lines.append(f"- **S1 Country**: {s1_row['country']} | **S3 Country**: {m_row['country']}")
                    lines.append("")
                    examples_shown += 1
                    
                if examples_shown >= 5:
                    break
        if examples_shown >= 5:
            break
            
    lines.append("\n")
    return "\n".join(lines)

def generate_observations() -> str:
    return """## 11. Important Observations for Phase 2

*(This section will be fully generated after observing the real data. Since we do not have the real data loaded yet, these are placeholders based on typical EDA)*

- Exact name overlap is usually very low, indicating that **fuzzy matching** will be necessary.
- Many addresses contain numeric tokens and postal codes which might require robust extraction.
- Legal suffixes (LLC, Inc) might be present inconsistently.
- A significant number of Source 1 entities might be singletons.
- France may appear only in the test set, highlighting the need for country-agnostic blocking strategies.
"""

def perform_eda(train_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame], output_path: Path):
    sections = [
        "# Phase 1 — Exploratory Data Analysis\n",
        "## 1. Dataset Overview\nThis report contains exploratory data analysis for the Business Entity Resolution dataset.\n",
        generate_file_validation(train_data, test_data),
        generate_source_stats(train_data, test_data),
        generate_country_analysis(train_data, test_data),
        analyze_business_names(train_data),
        analyze_addresses(train_data),
        generate_ground_truth_analysis(train_data.get("ground_truth", pd.DataFrame())),
        analyze_cross_source_overlap(train_data),
        generate_true_match_examples(train_data),
        generate_observations()
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))
