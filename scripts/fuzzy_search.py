import pandas as pd
from collections import defaultdict
from typing import List, TypedDict, Optional, Any
from helpers import ProposedChange, UnionFind
from rapidfuzz.distance import Levenshtein as _lev


#func for string distance
def _distance(a: str, b: str) -> int:
    return _lev.distance(a, b)

def propose_street_name_corrections(
    df_merged_view: pd.DataFrame,
    threshold: float = 0.10,
) -> List[ProposedChange]:
    """
    Rule 2 – Majority-vote street-name correction inside each (EID_1, num1_c) bucket.
    Returns a list of ProposedChange dicts; does NOT mutate the input frame.
    """
    proposals: List[ProposedChange] = []

    need = ["AID", "EID_1", "num1_c", "streetName_c"]
    if missing := [c for c in need if c not in df_merged_view.columns]:
        raise ValueError(f"Missing columns for street-name rule: {missing}")

    #to save some runtime, we can just use the needed columns for our rule
    df = (
        df_merged_view
        .loc[:, need]
        .dropna()
        .assign(
            streetName_c=lambda d: d["streetName_c"].astype("string")
        )
        .copy()
    )
    if df.empty:
        return proposals

    # Pre-compute frequency of ORIGINAL spellings within each (EID,num)
    freq = (
        df.groupby(["EID_1", "num1_c", "streetName_c"], observed=True)
          .size()
          .to_dict()
    )

    # strip whitespace and lowercase
    df["street_norm"] = (
        df["streetName_c"]
        .str.strip()
        .str.lower()
        .astype("string")
    )

    for (eid, num), g in df.groupby(["EID_1", "num1_c"], observed=True):
        norms = g["street_norm"].unique().tolist()
        if len(norms) < 2:
            continue

        norms.sort(key=len)
        uf = UnionFind(norms)

        for i, s1 in enumerate(norms):
            len1 = len(s1)
            for s2 in norms[i + 1:]:
                len2 = len(s2)
                # do some filtering of the strings so we don't run the comparison needlessly
                diff = abs(len1 - len2)
                longer = max(len1, len2)
                if diff and diff / longer >= threshold and diff > 2:
                    continue
                if _distance(s1, s2) / longer < threshold:
                    uf.union(s1, s2)

        # Proposed changes per group
        for cluster_norms in uf.clusters().values():
            rows = g[g["street_norm"].isin(cluster_norms)]
            originals = rows["streetName_c"].unique().tolist()
            if len(originals) <= 1:
                continue

            # pick "correct" spelling by frequency
            best = max(originals,
                       key=lambda v: (freq.get((eid, num, v), 0), v))

            needs_fix = rows[rows["streetName_c"] != best]
            if needs_fix.empty:
                continue

            for aid, cur in zip(needs_fix["AID"], needs_fix["streetName_c"]):
                proposals.append({
                    "original_AID": int(aid),
                    "EID_context": str(eid),
                    "column_to_change": "streetName_c",
                    "original_value": cur,
                    "proposed_value": best,
                    "rule_name": "Rule 2: Street-name majority vote"
                })

    return proposals
