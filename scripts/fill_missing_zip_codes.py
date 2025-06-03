import pandas as pd
import re
from typing import List, Optional, Any, TypedDict
from helpers import ProposedChange

ZIP_RE = re.compile(r'^_\d{5}$')

# Rule 1
def propose_fill_missing_zips_keep(df_merged_view: pd.DataFrame) -> List[ProposedChange]:
    """
    Rule 1: Within each (EID_1, num1_c, street_norm) group, if exactly one f
    valid ZIP exists, fill every blank ZIP in that group with that ZIP.

    Returns a list[ProposedChange]
    """
    need = ["AID", "EID_1", "num1_c", "streetName_c", "zip_c"]
    if missing := [c for c in need if c not in df_merged_view.columns]:
        raise ValueError(f"Missing cols for Rule 1: {missing}")

    df = df_merged_view.loc[:, need].copy()

    # make them all lowercases because case changes are not as interesting to use and strip of whitespace
    df["street_norm"] = (
        df["streetName_c"]
          .fillna("")
          .astype("string")
          .str.strip()
          .str.lower()
    )

    z = df["zip_c"].fillna("").astype(str).str.strip()
    df["_valid"] = z.str.match(r"^_\d{5}$")
    df["_blank"] = z.eq("")

    grp = ["EID_1", "num1_c", "street_norm"]

    # For each (EID, house-number, street) group,
    #         see if all rows share the SAME valid ZIP.
    #         If so, treat that ZIP as the “official” one.

    canon = (
        df.loc[df["_valid"], grp + ["zip_c"]]
          .drop_duplicates()
          .groupby(grp, sort=False)["zip_c"]
          .agg(list)
          .reset_index(name="zip_lst")
    )
    canon = canon[canon["zip_lst"].str.len().eq(1)]
    canon["canon_zip"] = canon["zip_lst"].str[0]
    canon = canon.drop(columns="zip_lst")

    # Merge (“join”) the official ZIP back onto the full table,
    # then keep only the rows where the ZIP is still blank—
    # those are the ones we need to fill.

    df = df.merge(canon, on=grp, how="left")
    targets = df.loc[df["_blank"] & df["canon_zip"].notna(),
                     ["AID", "EID_1", "zip_c", "canon_zip"]]

    #Creating proposal list
    proposals: List[ProposedChange] = [
        {
            "original_AID": int(aid),
            "EID_context": str(eid),
            "column_to_change": "zip_c",
            "original_value": orig,          # "" or NaN
            "proposed_value": new,           # canonical underscore-prefixed ZIP
            "rule_name": "Rule 1: Fill Missing ZIPs (Keep)",
        }
        for aid, eid, orig, new in targets.itertuples(index=False, name=None)
    ]
    return proposals

#Fast vectorised version of Rule 3a
def propose_replace_invalid_zips(df_merged_view: pd.DataFrame) -> List[ProposedChange]:
    """
    Rule 3a (fast): For each (EID_1, num1_c, street_norm) group, if exactly one
    valid ZIP appears, replace every invalid non-blank ZIP in that trio with
    that official ZIP.

    Returns a list[ProposedChange]
    """
    need = ["AID", "EID_1", "num1_c", "streetName_c", "zip_c"]
    if missing := [c for c in need if c not in df_merged_view.columns]:
        raise ValueError(f"Missing cols for Rule 3a: {missing}")

    df = df_merged_view.loc[:, need].copy()

    # make them all lowercases because case changes are not as interesting to use and strip of whitespace
    df["street_norm"] = (
        df["streetName_c"]
          .fillna("")
          .astype("string")
          .str.strip()
          .str.lower()
    )
    z = df["zip_c"].fillna("").astype(str).str.strip()
    df["_valid"]   = z.str.match(r"^_\d{5}$")
    df["_blank"]   = z.eq("")
    df["_invalid"] = (~df["_valid"]) & (~df["_blank"])

    grp = ["EID_1", "num1_c", "street_norm"]

    # For each (EID, house-number, street) group,
    # see if all rows share the SAME valid ZIP.
    # If so, treat that ZIP as the “official” one.

    canon = (
        df.loc[df["_valid"], grp + ["zip_c"]]
          .drop_duplicates()
          .groupby(grp, sort=False)["zip_c"]
          .agg(list)
          .reset_index(name="zip_lst")
    )
    canon = canon[canon["zip_lst"].str.len().eq(1)]         # unique ZIP only
    canon["canon_zip"] = canon["zip_lst"].str[0]
    canon = canon.drop(columns="zip_lst")

    # Merge (“join”) the official ZIP back onto the full table,
    # then keep only the rows where the ZIP is still blank—
    # those are the ones we need to fill.

    df = df.merge(canon, on=grp, how="left")
    targets = df.loc[df["_invalid"] & df["canon_zip"].notna(),
                     ["AID", "EID_1", "zip_c", "canon_zip"]]

    #Creating proposal list
    proposals: List[ProposedChange] = [
        {
            "original_AID": int(aid),
            "EID_context": str(eid),
            "column_to_change": "zip_c",
            "original_value": orig,
            "proposed_value": new,
            "rule_name": "Rule 3a: Replace Invalid ZIPs",
        }
        for aid, eid, orig, new in targets.itertuples(index=False, name=None)
    ]
    return proposals


#3b
def propose_fill_missing_zips_by_address(
    df_view: pd.DataFrame
) -> List[ProposedChange]:
    """
    Rule 3b (fast): For every (state, city, street_norm, num1_c) group —regardless
    of EID—if *exactly one* valid ZIP appears among non-blank rows, fill all blank
    ZIPs in that combo with that official ZIP.

    Returns list[ProposedChange]
    """
    need = ["AID", "state_c", "city_c", "streetName_c", "num1_c", "zip_c"]
    if miss := [c for c in need if c not in df_view.columns]:
        raise ValueError(f"Missing cols for 3b: {miss}")

    df = (
        df_view.loc[:, need]        # keep only needed cols
              .copy()
    )

    # make them all lowercases because case changes are not as interesting to use and strip of whitespace
    df["state_norm"]  = df["state_c"].fillna("").str.strip().str.upper()
    df["city_norm"]   = df["city_c"].fillna("").str.strip().str.lower()
    df["street_norm"] = df["streetName_c"].fillna("").str.strip().str.lower()

    z = df["zip_c"].fillna("").astype(str).str.strip()
    df["_valid"] = z.str.match(r"^_\d{5}$")
    df["_blank"] = z.eq("")

    grp = ["state_norm", "city_norm", "street_norm", "num1_c"]

    # For each (EID, house-number, street) group,
    # see if all rows share the SAME valid ZIP.
    # If so, treat that ZIP as the “official” one.

    valid = df.loc[df["_valid"], grp + ["zip_c"]]

    nunique = (
        valid.groupby(grp, sort=False)["zip_c"]
             .nunique()
             .rename("n_zip")
    )
    # keep combos where exactly one unique ZIP
    canon = (
        valid.merge(nunique.reset_index(), on=grp)
             .loc[lambda d: d["n_zip"].eq(1)]
             .drop_duplicates(subset=grp)
             .rename(columns={"zip_c": "canon_zip"})
             .drop(columns="n_zip")
    )

    # Merge (“join”) the official ZIP back onto the full table,
    # then keep only the rows where the ZIP is still blank—
    # those are the ones we need to fill.
    df = df.merge(canon, on=grp, how="left", copy=False)

    targets = df.loc[df["_blank"] & df["canon_zip"].notna(),
                     ["AID", "zip_c", "canon_zip"]]

    #Creating proposal list
    return [
        {
            "original_AID": int(aid),
            "EID_context": None,
            "column_to_change": "zip_c",
            "original_value": orig,      # "" or NaN
            "proposed_value": new,
            "rule_name": "Rule 3b: Fill Missing ZIPs by Address",
        }
        for aid, orig, new in targets.itertuples(index=False, name=None)
    ]

