import re
import unicodedata
import pandas as pd



# Text normalization

def normalize_unicode(text):
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def normalize_case(text):
    return text.casefold()


def normalize_punctuation(text):
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return normalize_whitespace(text)


def normalize_text(text):
    """
    Order:
        Unicode normalization
        ↓
        Case normalization
        ↓
        Punctuation normalization
        ↓
        Whitespace normalization
    """

    text = normalize_unicode(text)
    text = normalize_case(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)

    return text

# field specific
# Business name
def normalize_business_name(text):
    text = normalize_text(text)
    return text


# Business address

def normalize_business_address(text):
    text = normalize_text(text)
    return text

# Country

def normalize_country(text):
    if not isinstance(text, str):
        return ""

    return normalize_whitespace(
        normalize_case(
            normalize_unicode(text)
        )
    )

# normalize entire dataframe
def normalize_dataframe(df):
    result = df.copy()

    if "business_name" in result.columns:
        result["business_name_normalized"] = (
            result["business_name"]
            .map(normalize_business_name)
        )

    if "business_address" in result.columns:
        result["business_address_normalized"] = (
            result["business_address"]
            .map(normalize_business_address)
        )

    if "country" in result.columns:
        result["country_normalized"] = (
            result["country"]
            .map(normalize_country)
        )

    return result

# Script detection

SCRIPT_PATTERNS = {
    "Devanagari": re.compile(r"[\u0900-\u097F]"),
    "Tamil": re.compile(r"[\u0B80-\u0BFF]"),
    "Telugu": re.compile(r"[\u0C00-\u0C7F]"),
    "Kannada": re.compile(r"[\u0C80-\u0CFF]"),
    "Malayalam": re.compile(r"[\u0D00-\u0D7F]"),
    "Bengali": re.compile(r"[\u0980-\u09FF]"),
    "Gujarati": re.compile(r"[\u0A80-\u0AFF]"),
    "Gurmukhi": re.compile(r"[\u0A00-\u0A7F]"),
    "Arabic": re.compile(r"[\u0600-\u06FF]"),
    "Latin": re.compile(r"[A-Za-z]"),
}


def detect_script(text):
    if not isinstance(text, str):
        return "Unknown"

    for script, pattern in SCRIPT_PATTERNS.items():

        if pattern.search(text):
            return script

    return "Other"