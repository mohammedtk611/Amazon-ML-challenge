# Model Card

## Model
LightGBM binary classifier

## Task
Business entity pair classification

## Input
Pair-level numerical features (RapidFuzz distances)

## Output
Match probability

## Training
Trained on synthetic negative pairs sampled from the blocking candidate universe.

## Validation
Strict Source-1 entity-level split.

## Threshold
Grid search maximized for Entity-Level F0.5.

## Known Limitations
- Bounded entirely by Blocking Recall ceiling.
- Extremely noise-sensitive to transliteration that bypasses string distance metrics.