# Holdout error analysis

Categories are diagnostic heuristics, not verified causal explanations.
Held-out labels are used for reporting only; model and retrieval choices use the tuning split. No model was retuned on these errors.

| Type | Category | Count |
|---|---|---:|
| false_negative | blocking_miss | 1120 |
| false_negative | missing_address | 133 |
| false_negative | numeric_conflict | 221 |
| false_negative | script_variation | 44 |
| false_negative | similar_address_different_name | 40 |
| false_negative | similar_name_different_address | 24 |
| false_negative | text_variation_or_ambiguous | 271 |
| false_positive | missing_address | 5 |
| false_positive | numeric_conflict | 9 |
| false_positive | script_variation | 19 |
| false_positive | similar_address_different_name | 3 |
| false_positive | similar_name_different_address | 2 |
| false_positive | text_variation_or_ambiguous | 84 |

## True-pair name script transitions

```json
{
  "Latin -> Latin": 7559,
  "Latin -> Devanagari": 317,
  "Latin -> Gujarati": 43,
  "Latin -> Gujarati+Latin": 4,
  "Latin -> Devanagari+Latin": 23,
  "Latin -> Kannada+Latin": 6,
  "Latin -> Kannada": 42,
  "Latin -> Latin+Tamil": 6,
  "Latin -> Telugu": 41,
  "Latin -> Latin+Telugu": 3,
  "Latin -> Bengali": 30,
  "Latin -> Bengali+Latin": 2,
  "Latin -> Tamil": 36,
  "Latin -> Oriya": 6,
  "Latin -> Malayalam": 6,
  "Latin -> Gurmukhi": 6,
  "Latin -> Other": 1,
  "Latin -> Latin+Malayalam": 2
}
```

Full error pairs, original Unicode text and probabilities are in experiments/error_analysis.csv.
Cross-script truth pairs demonstrate script variation; local transliteration is an auxiliary representation and can create collisions.
First/last numeric features are heuristics, not country-specific house/postcode parsers. No external address data is used.