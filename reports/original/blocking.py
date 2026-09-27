import re
import pandas as pd


# ============================================================
# BASIC TOKEN HELPERS
# ============================================================

def get_tokens(text):
    if not isinstance(text, str) or not text:
        return []

    return [token for token in text.split() if token]


# ============================================================
# NAME BLOCKING KEYS
# ============================================================

def make_name_key(text):
    """
    Existing conservative key:
    first token + last token
    """
    tokens = get_tokens(text)

    if not tokens:
        return ""

    if len(tokens) == 1:
        return tokens[0]

    return f"{tokens[0]}_{tokens[-1]}"


def make_name_first_key(text):
    """
    First token of normalized business name.
    """
    tokens = get_tokens(text)

    if not tokens:
        return ""

    return tokens[0]


def make_name_last_key(text):
    """
    Last token of normalized business name.
    """
    tokens = get_tokens(text)

    if not tokens:
        return ""

    return tokens[-1]


def make_name_length_key(text):
    """
    Number of tokens in the business name.
    Used together with country and another key.
    """
    tokens = get_tokens(text)

    if not tokens:
        return ""

    return str(len(tokens))


# ============================================================
# ADDRESS BLOCKING KEYS
# ============================================================

def make_address_key(text):
    """
    Existing conservative address key:
    first 3 tokens.
    """
    tokens = get_tokens(text)

    if not tokens:
        return ""

    return "_".join(tokens[:3])


def make_address_first_key(text):
    """
    First token of normalized address.
    """
    tokens = get_tokens(text)

    if not tokens:
        return ""

    return tokens[0]


def make_address_last_key(text):
    """
    Last token of normalized address.
    """
    tokens = get_tokens(text)

    if not tokens:
        return ""

    return tokens[-1]


# ============================================================
# ADD ALL BLOCKING KEYS
# ============================================================

def add_blocking_keys(df):
    """
    Add multiple blocking keys to dataframe.
    """

    result = df.copy()

    # -------------------------
    # NAME KEYS
    # -------------------------

    result["name_block_key"] = (
        result["business_name_normalized"]
        .map(make_name_key)
    )

    result["name_first_key"] = (
        result["business_name_normalized"]
        .map(make_name_first_key)
    )

    result["name_last_key"] = (
        result["business_name_normalized"]
        .map(make_name_last_key)
    )

    result["name_length_key"] = (
        result["business_name_normalized"]
        .map(make_name_length_key)
    )

    # -------------------------
    # ADDRESS KEYS
    # -------------------------

    result["address_block_key"] = (
        result["business_address_normalized"]
        .map(make_address_key)
    )

    result["address_first_key"] = (
        result["business_address_normalized"]
        .map(make_address_first_key)
    )

    result["address_last_key"] = (
        result["business_address_normalized"]
        .map(make_address_last_key)
    )

    # -------------------------
    # COUNTRY
    # -------------------------

    result["country_block_key"] = (
        result["country_normalized"]
    )

    return result


# ============================================================
# GENERIC CANDIDATE GENERATOR
# ============================================================

def generate_candidates(
    source1,
    source2,
    blocking_columns,
):
    """
    Generate candidate pairs using arbitrary blocking columns.
    """

    left_columns = [
        "entity_id",
        *blocking_columns,
    ]

    right_columns = [
        "entity_id",
        *blocking_columns,
    ]

    left = source1[left_columns].copy()
    right = source2[right_columns].copy()

    # Remove empty blocking keys.
    for column in blocking_columns:
        left = left[left[column] != ""]
        right = right[right[column] != ""]

    candidates = left.merge(
        right,
        on=blocking_columns,
        suffixes=("_s1", "_s2"),
        how="inner",
    )

    return candidates[
        [
            "entity_id_s1",
            "entity_id_s2",
        ]
    ].drop_duplicates()


# ============================================================
# MULTI-STRATEGY BLOCKING
# ============================================================

def generate_all_candidates(source1, source2):
    """
    Generate candidates using multiple blocking strategies.

    The final candidate set is the UNION of all strategies.
    """

    strategies = [
        ["country_block_key", "name_block_key"],

        ["country_block_key", "name_first_key"],

        ["country_block_key", "name_last_key"],

        ["country_block_key", "address_block_key"],

        ["country_block_key", "address_first_key"],

        ["country_block_key", "address_last_key"],
    ]

    all_candidates = []

    for strategy in strategies:

        candidates = generate_candidates(
            source1,
            source2,
            strategy,
        )

        all_candidates.append(candidates)

    combined = pd.concat(
        all_candidates,
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=[
            "entity_id_s1",
            "entity_id_s2",
        ]
    )

    return combined.reset_index(drop=True)


# ============================================================
# BACKWARD-COMPATIBLE FUNCTIONS
# ============================================================

def generate_candidates_by_name(source1, source2):
    return generate_candidates(
        source1,
        source2,
        [
            "country_block_key",
            "name_block_key",
        ],
    )


def generate_candidates_by_address(source1, source2):
    return generate_candidates(
        source1,
        source2,
        [
            "country_block_key",
            "address_block_key",
        ],
    )


def combine_candidates(
    name_candidates,
    address_candidates,
):
    candidates = pd.concat(
        [
            name_candidates,
            address_candidates,
        ],
        ignore_index=True,
    )

    candidates = candidates.drop_duplicates(
        subset=[
            "entity_id_s1",
            "entity_id_s2",
        ],
    )

    return candidates.reset_index(drop=True)