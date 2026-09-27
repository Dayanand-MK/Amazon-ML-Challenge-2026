"""Pairwise signals; empty values never count as exact matches."""
import re
from functools import lru_cache
import numpy as np
from rapidfuzz import fuzz
from .normalization import detect_script

TEXT_NAMES = ["exact", "ratio", "partial", "token_sort", "token_set", "jaccard", "length_ratio", "bigram_dice", "missing"]
FEATURE_NAMES = [f"{prefix}_{f}" for prefix in ("name", "address") for f in TEXT_NAMES] + [
    "translit_name_ratio", "translit_address_ratio", "country_equal", "country_missing",
    "script_equal", "source3", "numeric_jaccard", "numeric_conflict", "first_number_equal",
    "last_long_number_equal", "name_address_product", "name_numeric_jaccard"]


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.


@lru_cache(maxsize=6000)
def text_parts(text):
    return frozenset(text.split()), frozenset(text[i:i+2] for i in range(len(text)-1))


@lru_cache(maxsize=6000)
def script(text):
    return detect_script(text)


def text_features(a, b):
    if not a or not b:
        return [0.]*8 + [1.]
    x, ga = text_parts(a)
    y, gb = text_parts(b)
    return [float(a == b), fuzz.ratio(a,b)/100, fuzz.partial_ratio(a,b)/100,
            fuzz.token_sort_ratio(a,b)/100, fuzz.token_set_ratio(a,b)/100,
            jaccard(x,y), min(len(a),len(b))/max(len(a),len(b)),
            2*len(ga & gb)/(len(ga)+len(gb)) if ga and gb else 0., 0.]


def pair_features(left, right):
    _, n, a, c, nt, at = left
    eid, m, b, d, mt, bt = right
    nf, af = text_features(n,m), text_features(a,b)
    an, bn = re.findall(r"\b\d+\b", at), re.findall(r"\b\d+\b", bt)
    ap, bp = [x for x in an if len(x)>=4], [x for x in bn if len(x)>=4]
    return nf + af + [fuzz.ratio(nt,mt)/100 if nt and mt else 0.,
        fuzz.ratio(at,bt)/100 if at and bt else 0., float(bool(c and d) and c==d), float(not c or not d),
        float(bool(n and m) and script(n)==script(m)), float(eid.startswith("S3-")),
        jaccard(set(an),set(bn)), float(bool(an and bn) and not set(an)&set(bn)),
        float(bool(an and bn) and an[0]==bn[0]), float(bool(ap and bp) and ap[-1]==bp[-1]),
        nf[1]*af[1], jaccard(set(re.findall(r"\d+",n)),set(re.findall(r"\d+",m)))]


def feature_matrix(left, rights):
    return np.asarray([pair_features(left,r) for r in rights], dtype=np.float32).reshape(-1,len(FEATURE_NAMES))
