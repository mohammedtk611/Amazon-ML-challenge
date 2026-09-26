# Business Entity Resolution — Complete Explanation

## 1. Problem
Business Entity Resolution is the task of identifying whether two or more distinct textual records refer to the same real-world business entity. In this challenge, Source 1 (S1) contains unique reference records. Source 2 (S2) and Source 3 (S3) contain potentially duplicate, noisy records. The records are difficult to match due to severe OCR noise, transliteration variations across languages, missing fields, and structural inconsistencies in address formatting.

## 2. Overall Idea
Raw Records -> Normalization -> Blocking -> Candidate Pairs -> Feature Engineering -> LightGBM -> Probability -> Threshold -> Entity-Level Matching -> Final Output

## 3. Why We Cannot Compare Everything
An $O(N^2)$ cross-join across millions of records is computationally infeasible. Blocking restricts the search space by only generating Candidate Pairs for records that share a high-confidence exact match on cheap-to-compute structural keys (like normalized country, postal codes, or numeric structural elements).

## 4. Normalization
- **Business names**: Lowercased, stripped of punctuation, and basic legal suffixes standardized.
- **Addresses**: Lowercased, stripped of common punctuation.
- **Countries**: ISO mappings applied where possible.
- **Postal codes / Numerics**: Extracted explicitly via regex into independent columns.
Raw values are preserved so that features can calculate string distances without losing native signal.

## 5. Blocking
- Token blocking (matching significant unigrams)
- Postal blocking
- Numeric-address blocking
These are combined using a logical UNION. If any strategy hits, the pair enters the candidate set.

## 6. Features
- Name similarity (RapidFuzz Jaro-Winkler)
- Address similarity (RapidFuzz Levenshtein)
- Numeric token containment
- Exact column matches

## 7. LightGBM
Input: A 1D vector of numerical similarity and categorical boolean features.
Output: A probability $P(Match=1)$.
It is a binary classification problem because any given S1-candidate pair is either fundamentally the same entity (1) or different (0).

## 8. Threshold
The probability must be binarized into a strict decision. A threshold of 0.5 is mathematically arbitrary; threshold optimization calibrates this cutoff to specifically maximize the F0.5 metric, balancing the challenge's high penalty on false positives (precision).

## 9. Entity-Level Matching
One S1 record can map to 0, 1, or N S2/S3 records. 

## 10. Evaluation
- Precision: How many predicted matches were true?
- Recall: How many true matches were predicted?
- F0.5: Harmonic mean of precision and recall, with precision weighted twice as much. F0.5 is the official challenge metric.

## 11. Final Outputs
- `candidate_pairs.tsv`: The unioned output of the blocking stage.
- `matching_results.tsv`: The final threshold-filtered subset of the candidate pairs.