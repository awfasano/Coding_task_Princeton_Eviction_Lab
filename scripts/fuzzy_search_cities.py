import pandas as pd
from collections import defaultdict
from typing import List
from helpers import ProposedChange, UnionFind
from rapidfuzz.distance import Levenshtein as _lev


def _dist(a: str, b: str) -> int:
    return _lev.distance(a, b)

def propose_correct_city_names_by_zip(
    df_view: pd.DataFrame,
    threshold: float = 0.10,
) -> List[ProposedChange]:
    """
    Rule 4: Within each valid 5-digit ZIP, group city spellings that
    differ by < `threshold` edit-distance ratio, choose the majority spelling,
    and add propose objects to list to fix
    """

    need = ["AID", "zip_c", "city_c"]
    if miss := [c for c in need if c not in df_view.columns]:
        raise ValueError(f"Missing columns for Rule 4: {miss}")

    df = (
        df_view.loc[:, need]         # keep only needed cols
               .dropna(subset=["zip_c", "city_c"])
               .copy()
    )

    # --- keep rows whose ZIP already matches the _12345 pattern -------
    z = df["zip_c"].astype(str).str.strip()
    valid_zip_mask = z.str.match(r"^_\d{5}$")
    df = df.loc[valid_zip_mask]
    if df.empty:
        return []

    # table to break ties, so we use the spelling used most often
    freq = (
        df.groupby(["zip_c", "city_c"], observed=True)
          .size()
          .to_dict()
    )

    # strip whitespace and lowercase for grouping
    df["city_norm"] = (
        df["city_c"]
          .astype("string")
          .str.strip()
          .str.lower()
    )

    canon_map: dict[tuple[str, str], str] = {}  # (zip, city_norm) → canonical or "correct" city_c

    for zip_code, sub in df.groupby("zip_c", observed=True):
        variants = sub["city_norm"].unique().tolist()
        if len(variants) < 2:
            continue

        variants.sort(key=len)
        uf = UnionFind(variants)

        for i, s1 in enumerate(variants):
            len1 = len(s1)
            for s2 in variants[i + 1:]:
                len2   = len(s2)
                longer = max(len1, len2)
                if abs(len1 - len2) > 2 and abs(len1 - len2) / longer >= threshold:
                    continue  # fast prune
                if _dist(s1, s2) / longer < threshold:
                    uf.union(s1, s2)

        for cluster in uf.clusters().values():
            if len(cluster) == 1:
                continue

            sub_rows   = sub[sub["city_norm"].isin(cluster)]
            originals  = sub_rows["city_c"].unique().tolist()
            best_city  = max(
                originals,
                key=lambda c: (freq.get((zip_code, c), 0), c)
            )
            for norm in cluster:
                canon_map[(zip_code, norm)] = best_city

    # create column with canonical suggestion (or NaN if none)
    df["canon_city"] = [
        canon_map.get((zipc, norm))
        for zipc, norm in zip(df["zip_c"], df["city_norm"])
    ]
    mask = df["canon_city"].notna() & (df["canon_city"] != df["city_c"])
    rows_to_fix = df.loc[mask, ["AID", "zip_c", "city_c", "canon_city"]]

    # build Proposal list
    proposals: List[ProposedChange] = [
        {
            "original_AID": int(aid),
            "EID_context": None,
            "column_to_change": "city_c",
            "original_value": orig,
            "proposed_value": new,
            "rule_name": "Rule 4: Fuzzy city by ZIP",
        }
        for aid, _, orig, new in rows_to_fix.itertuples(index=False, name=None)
    ]
    return proposals
