# Technical Deep-Dive

1. **Why blocking was necessary**: Scale.
2. **Why multiple blocks**: No single noisy key is reliable enough.
3. **Blocking recall**: The model cannot match what the blocker drops.
4. **Leakage prevention**: S1 GroupKFold splitting.
5. **Why LightGBM**: Fast, natively handles missing values.