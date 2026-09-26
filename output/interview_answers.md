# Interview Answers

- **Why blocking?**: Because N^2 cross joins crash on millions of records.
- **Why UNION?**: Intersections limit recall too heavily on noisy data.
- **Entity Leakage?**: We split strictly by `source1_entity_id`, never row-level.
- **Zero matches?**: We explicitly allow S1 entities to have empty sets to maximize precision if probabilities fall below the threshold.
- **LLM Usage?**: That was not implemented in the final system.