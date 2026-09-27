# Full dataset EDA

All six source TSV files were scanned; original files were not modified.
Script names describe Unicode writing systems, not inferred spoken languages. Mixed scripts contain `+`.

| File | Rows | Unique IDs | Duplicate IDs | Duplicate rows |
|---|---:|---:|---:|---:|
| train_source1 | 2,206,821 | 2,206,821 | 0 | 0 |
| train_source2 | 5,034,616 | 5,034,616 | 0 | 0 |
| train_source3 | 5,285,603 | 5,285,603 | 0 | 0 |
| test_source1 | 1,732,544 | 1,732,544 | 0 | 0 |
| test_source2 | 4,887,273 | 4,887,273 | 0 | 0 |
| test_source3 | 5,082,316 | 5,082,316 | 0 | 0 |

## train_source1

```json
{
  "rows": 2206821,
  "unique_entity_ids": 2206821,
  "duplicate_entity_ids": 0,
  "duplicate_full_rows": 0,
  "missing": {
    "entity_id": 0,
    "business_name": 0,
    "business_address": 0,
    "country": 0
  },
  "countries": {
    "US": 1323633,
    "India": 883188
  },
  "name_scripts": {
    "Latin": 2206821
  },
  "address_scripts": {
    "Latin": 2206821
  },
  "name_lengths": {
    "min": 3,
    "max": 105,
    "mean": 24.03440333402664,
    "median": 24,
    "p95": 37
  },
  "address_lengths": {
    "min": 11,
    "max": 256,
    "mean": 52.06621289175697,
    "median": 41,
    "p95": 103
  },
  "name_script_presence": {
    "Latin": 2206821,
    "Devanagari": 0,
    "Arabic": 0,
    "Cyrillic": 0,
    "CJK": 0
  },
  "mixed_name_records": 0,
  "address_script_presence": {
    "Latin": 2206821,
    "Devanagari": 0,
    "Arabic": 0,
    "Cyrillic": 0,
    "CJK": 0
  },
  "mixed_address_records": 0
}
```

## train_source2

```json
{
  "rows": 5034616,
  "unique_entity_ids": 5034616,
  "duplicate_entity_ids": 0,
  "duplicate_full_rows": 0,
  "missing": {
    "entity_id": 0,
    "business_name": 0,
    "business_address": 168967,
    "country": 0
  },
  "countries": {
    "India": 2017799,
    "US": 3016817
  },
  "name_scripts": {
    "Devanagari": 258863,
    "Latin": 4560077,
    "Tamil": 32440,
    "Gujarati": 29700,
    "Kannada": 35787,
    "Bengali": 29497,
    "Telugu": 37799,
    "Malayalam": 18062,
    "Devanagari+Latin": 10561,
    "Oriya": 7194,
    "Gurmukhi+Latin": 271,
    "Gurmukhi": 6417,
    "Gujarati+Latin": 1229,
    "Latin+Malayalam": 711,
    "Latin+Tamil": 1341,
    "Other": 194,
    "Kannada+Latin": 1424,
    "Latin+Telugu": 1524,
    "Bengali+Latin": 1226,
    "Latin+Oriya": 299
  },
  "address_scripts": {
    "Latin": 4388128,
    "Devanagari+Latin": 277523,
    "Other": 168967,
    "Bengali+Latin": 31770,
    "Gujarati+Latin": 31543,
    "Latin+Tamil": 34777,
    "Kannada+Latin": 38560,
    "Latin+Telugu": 34382,
    "Latin+Malayalam": 15680,
    "Latin+Oriya": 6157,
    "Gurmukhi+Latin": 7129
  },
  "name_lengths": {
    "min": 2,
    "max": 104,
    "mean": 25.103574930044317,
    "median": 25,
    "p95": 40
  },
  "address_lengths": {
    "min": 0,
    "max": 249,
    "mean": 46.22590858965212,
    "median": 37,
    "p95": 96
  },
  "name_script_presence": {
    "Devanagari": 269424,
    "Latin": 4578663,
    "Tamil": 33781,
    "Gujarati": 30929,
    "Kannada": 37211,
    "Bengali": 30723,
    "Telugu": 39323,
    "Malayalam": 18773,
    "Oriya": 7493,
    "Gurmukhi": 6688,
    "Other": 194,
    "Arabic": 0,
    "Cyrillic": 0,
    "CJK": 0
  },
  "mixed_name_records": 18586,
  "address_script_presence": {
    "Latin": 4865649,
    "Devanagari": 277523,
    "Other": 168967,
    "Bengali": 31770,
    "Gujarati": 31543,
    "Tamil": 34777,
    "Kannada": 38560,
    "Telugu": 34382,
    "Malayalam": 15680,
    "Oriya": 6157,
    "Gurmukhi": 7129,
    "Arabic": 0,
    "Cyrillic": 0,
    "CJK": 0
  },
  "mixed_address_records": 477521
}
```

## train_source3

```json
{
  "rows": 5285603,
  "unique_entity_ids": 5285603,
  "duplicate_entity_ids": 0,
  "duplicate_full_rows": 0,
  "missing": {
    "entity_id": 0,
    "business_name": 0,
    "business_address": 175916,
    "country": 0
  },
  "countries": {
    "US": 3170056,
    "India": 2115547
  },
  "name_scripts": {
    "Latin": 5006656,
    "Latin+Tamil": 2444,
    "Malayalam": 9759,
    "Devanagari": 138187,
    "Latin+Telugu": 2945,
    "Gujarati": 15746,
    "Kannada": 19254,
    "Devanagari+Latin": 19816,
    "Tamil": 17346,
    "Kannada+Latin": 2741,
    "Latin+Oriya": 536,
    "Bengali": 15863,
    "Oriya": 3781,
    "Telugu": 20088,
    "Gurmukhi": 3606,
    "Gurmukhi+Latin": 500,
    "Gujarati+Latin": 2274,
    "Bengali+Latin": 2281,
    "Latin+Malayalam": 1357,
    "Other": 423
  },
  "address_scripts": {
    "Latin": 4633909,
    "Other": 175916,
    "Kannada+Latin": 38965,
    "Devanagari+Latin": 275815,
    "Gujarati+Latin": 31508,
    "Latin+Telugu": 34319,
    "Latin+Tamil": 34966,
    "Bengali+Latin": 31447,
    "Latin+Malayalam": 15375,
    "Gurmukhi+Latin": 7195,
    "Latin+Oriya": 6101,
    "Malayalam": 4,
    "Gujarati": 9,
    "Devanagari": 46,
    "Tamil": 12,
    "Kannada": 9,
    "Telugu": 5,
    "Oriya": 1,
    "Bengali": 1
  },
  "name_lengths": {
    "min": 2,
    "max": 123,
    "mean": 25.202143823514554,
    "median": 25,
    "p95": 42
  },
  "address_lengths": {
    "min": 0,
    "max": 240,
    "mean": 46.71443731207206,
    "median": 42,
    "p95": 91
  },
  "name_script_presence": {
    "Latin": 5041550,
    "Tamil": 19790,
    "Malayalam": 11116,
    "Devanagari": 158003,
    "Telugu": 23033,
    "Gujarati": 18020,
    "Kannada": 21995,
    "Oriya": 4317,
    "Bengali": 18144,
    "Gurmukhi": 4106,
    "Other": 423,
    "Arabic": 0,
    "Cyrillic": 0,
    "CJK": 0
  },
  "mixed_name_records": 34894,
  "address_script_presence": {
    "Latin": 5109600,
    "Other": 175916,
    "Kannada": 38974,
    "Devanagari": 275861,
    "Gujarati": 31517,
    "Telugu": 34324,
    "Tamil": 34978,
    "Bengali": 31448,
    "Malayalam": 15379,
    "Gurmukhi": 7195,
    "Oriya": 6102,
    "Arabic": 0,
    "Cyrillic": 0,
    "CJK": 0
  },
  "mixed_address_records": 475691
}
```

## test_source1

```json
{
  "rows": 1732544,
  "unique_entity_ids": 1732544,
  "duplicate_entity_ids": 0,
  "duplicate_full_rows": 0,
  "missing": {
    "entity_id": 0,
    "business_name": 0,
    "business_address": 0,
    "country": 0
  },
  "countries": {
    "US": 663106,
    "France": 259452,
    "India": 809986
  },
  "name_scripts": {
    "Latin": 1732544
  },
  "address_scripts": {
    "Latin": 1732544
  },
  "name_lengths": {
    "min": 3,
    "max": 92,
    "mean": 23.83642608788002,
    "median": 24,
    "p95": 36
  },
  "address_lengths": {
    "min": 11,
    "max": 268,
    "mean": 57.213179578700455,
    "median": 50,
    "p95": 105
  },
  "name_script_presence": {
    "Latin": 1732544,
    "Devanagari": 0,
    "Arabic": 0,
    "Cyrillic": 0,
    "CJK": 0
  },
  "mixed_name_records": 0,
  "address_script_presence": {
    "Latin": 1732544,
    "Devanagari": 0,
    "Arabic": 0,
    "Cyrillic": 0,
    "CJK": 0
  },
  "mixed_address_records": 0
}
```

## test_source2

```json
{
  "rows": 4887273,
  "unique_entity_ids": 4887273,
  "duplicate_entity_ids": 0,
  "duplicate_full_rows": 0,
  "missing": {
    "entity_id": 0,
    "business_name": 0,
    "business_address": 129408,
    "country": 0
  },
  "countries": {
    "India": 2312565,
    "France": 703378,
    "US": 1871330
  },
  "name_scripts": {
    "Latin": 4340556,
    "Telugu": 43403,
    "Gujarati+Latin": 1303,
    "Gurmukhi": 7618,
    "Bengali": 34207,
    "Malayalam": 21154,
    "Devanagari": 297604,
    "Kannada": 42256,
    "Devanagari+Latin": 11499,
    "Tamil": 37458,
    "Gujarati": 34018,
    "Oriya": 8332,
    "Bengali+Latin": 1339,
    "Latin+Telugu": 1768,
    "Kannada+Latin": 1650,
    "Gurmukhi+Latin": 285,
    "Latin+Malayalam": 879,
    "Latin+Tamil": 1495,
    "Latin+Oriya": 338,
    "Other": 111
  },
  "address_scripts": {
    "Latin": 4195411,
    "Gujarati+Latin": 36709,
    "Devanagari+Latin": 318369,
    "Gurmukhi+Latin": 8195,
    "Latin+Oriya": 7088,
    "Latin+Tamil": 40621,
    "Latin+Malayalam": 18129,
    "Other": 129408,
    "Bengali+Latin": 36832,
    "Kannada+Latin": 45006,
    "Latin+Masculine": 11881,
    "Latin+Telugu": 39624
  },
  "name_lengths": {
    "min": 2,
    "max": 102,
    "mean": 25.703646184692364,
    "median": 25,
    "p95": 42
  },
  "address_lengths": {
    "min": 0,
    "max": 269,
    "mean": 50.413405594490015,
    "median": 43,
    "p95": 99
  },
  "name_script_presence": {
    "Latin": 4361112,
    "Telugu": 45171,
    "Gujarati": 35321,
    "Gurmukhi": 7903,
    "Bengali": 35546,
    "Malayalam": 22033,
    "Devanagari": 309103,
    "Kannada": 43906,
    "Tamil": 38953,
    "Oriya": 8670,
    "Other": 111,
    "Arabic": 0,
    "Cyrillic": 0,
    "CJK": 0
  },
  "mixed_name_records": 20556,
  "address_script_presence": {
    "Latin": 4757865,
    "Gujarati": 36709,
    "Devanagari": 318369,
    "Gurmukhi": 8195,
    "Oriya": 7088,
    "Tamil": 40621,
    "Malayalam": 18129,
    "Other": 129408,
    "Bengali": 36832,
    "Kannada": 45006,
    "Masculine": 11881,
    "Telugu": 39624,
    "Arabic": 0,
    "Cyrillic": 0,
    "CJK": 0
  },
  "mixed_address_records": 562454
}
```

## test_source3

```json
{
  "rows": 5082316,
  "unique_entity_ids": 5082316,
  "duplicate_entity_ids": 0,
  "duplicate_full_rows": 0,
  "missing": {
    "entity_id": 0,
    "business_name": 0,
    "business_address": 136098,
    "country": 0
  },
  "countries": {
    "India": 2405000,
    "France": 731615,
    "US": 1945701
  },
  "name_scripts": {
    "Devanagari": 158605,
    "Latin": 4761439,
    "Tamil": 19994,
    "Devanagari+Latin": 22463,
    "Kannada": 22706,
    "Gujarati+Latin": 2649,
    "Gurmukhi": 4061,
    "Bengali": 18297,
    "Malayalam": 11471,
    "Telugu": 23268,
    "Latin+Tamil": 2814,
    "Gurmukhi+Latin": 573,
    "Oriya": 4456,
    "Gujarati": 17921,
    "Latin+Telugu": 3430,
    "Latin+Oriya": 606,
    "Kannada+Latin": 3126,
    "Bengali+Latin": 2575,
    "Latin+Malayalam": 1624,
    "Other": 238
  },
  "address_scripts": {
    "Latin": 4383926,
    "Devanagari+Latin": 319016,
    "Kannada+Latin": 45153,
    "Bengali+Latin": 36425,
    "Latin+Tamil": 40538,
    "Other": 136098,
    "Gujarati+Latin": 36727,
    "Gurmukhi+Latin": 7990,
    "Latin+Masculine": 11809,
    "Latin+Telugu": 39420,
    "Latin+Malayalam": 17928,
    "Latin+Oriya": 7187,
    "Devanagari": 57,
    "Gujarati": 5,
    "Tamil": 6,
    "Malayalam": 5,
    "Bengali": 2,
    "Telugu": 10,
    "Kannada": 11,
    "Gurmukhi": 2,
    "Oriya": 1
  },
  "name_lengths": {
    "min": 2,
    "max": 103,
    "mean": 25.65519440349636,
    "median": 25,
    "p95": 42
  },
  "address_lengths": {
    "min": 0,
    "max": 267,
    "mean": 48.74047087981149,
    "median": 43,
    "p95": 94
  },
  "name_script_presence": {
    "Devanagari": 181068,
    "Latin": 4801299,
    "Tamil": 22808,
    "Kannada": 25832,
    "Gujarati": 20570,
    "Gurmukhi": 4634,
    "Bengali": 20872,
    "Malayalam": 13095,
    "Telugu": 26698,
    "Oriya": 5062,
    "Other": 238,
    "Arabic": 0,
    "Cyrillic": 0,
    "CJK": 0
  },
  "mixed_name_records": 39860,
  "address_script_presence": {
    "Latin": 4946119,
    "Devanagari": 319073,
    "Kannada": 45164,
    "Bengali": 36427,
    "Tamil": 40544,
    "Other": 136098,
    "Gujarati": 36732,
    "Gurmukhi": 7992,
    "Masculine": 11809,
    "Telugu": 39430,
    "Malayalam": 17933,
    "Oriya": 7188,
    "Arabic": 0,
    "Cyrillic": 0,
    "CJK": 0
  },
  "mixed_address_records": 562193
}
```

## ground_truth

```json
{
  "rows": 2206821,
  "match_count_distribution": {
    "5": 321957,
    "4": 484115,
    "3": 530841,
    "6": 164868,
    "2": 375212,
    "7": 63968,
    "0": 123247,
    "1": 119157,
    "8": 18680,
    "9": 4205,
    "10": 534,
    "11": 37
  },
  "target_source_pairs": {
    "S2": 3693619,
    "S3": 3944746
  },
  "zero_matches": 123247,
  "one_match": 119157,
  "multiple_matches": 1964417
}
```

Paired script variation and error examples are measured separately on the held-out sample in error_analysis.md.