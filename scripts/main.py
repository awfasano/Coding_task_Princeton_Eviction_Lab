"""
Full script to
  • load fe.csv, fa.csv and the relationships csv
  • Find all possible changes we want to update and put them into an array of proposed changes
  • Any updates we are making on an AID need to not conflict if we find
    a conflict like two different updates.  We should resolve this conflict, splitting the AID

Rule's functions can be found in the following files:
  * fill_missing_zip_codes.py - Rule 1, 3a, 3b
  * fill_street_name.py - Rule 2
  * fuzzy_search.py - Rule 4
  * fuzzy_search_cities.py
  * helpers.py - classes: ProposedChange and SplitEvent
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Callable, List, Tuple, Any

from helpers import ProposedChange, SplitEvent

#Import rule functions engines
from fuzzy_search import propose_street_name_corrections           # Rule 2
from fill_missing_zip_codes import (
    propose_fill_missing_zips_keep,      # Rule 1
    propose_replace_invalid_zips,        # Rule 3a
    propose_fill_missing_zips_by_address # Rule 3b
)
from fuzzy_search_cities import propose_correct_city_names_by_zip  # Rule 4
from pandas import DataFrame

#  Search thresholds for easy modulation
STREET_THRESHOLD = 0.10
CITY_THRESHOLD   = 0.10

# Path helpers
DATA_DIR = Path(__file__).parent.parent / "data"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df_fa      = pd.read_csv(DATA_DIR / "fa.csv")
    df_fe      = pd.read_csv(DATA_DIR / "fe.csv")
    df_r_fe_fa = pd.read_csv(DATA_DIR / "r_fe_fa.csv")

    #ensure type consistency
    df_fa["AID"]   = df_fa["AID"].astype(int)
    df_fe["EID"]    = df_fe["EID"].astype(str)
    df_r_fe_fa["AID_2"] = df_r_fe_fa["AID_2"].astype(int)
    df_r_fe_fa["EID_1"] = df_r_fe_fa["EID_1"].astype(str)

    print(
        f"Loaded   df_fa={df_fa.shape}  df_fe={df_fe.shape}  df_r_fe_fa={df_r_fe_fa.shape}"
    )
    return df_fa, df_r_fe_fa, df_fe

def reconstruct_full_address(row: pd.Series) -> str:
    parts = [
        str(row["num1_c"])           if pd.notna(row.get("num1_c"))           else None,
        str(row["streetName_c"])      if pd.notna(row.get("streetName_c"))      else None,
        str(row["city_c"])            if pd.notna(row.get("city_c"))            else None,
        str(row["state_c"])           if pd.notna(row.get("state_c"))           else None,
        str(row["zip_c"])             if pd.notna(row.get("zip_c"))             else None,
    ]
    return " ".join([p for p in parts if p]).strip()

# Post‑split consistency pass
def split_conflicting_addresses(
    df_fa: pd.DataFrame,
    df_r: pd.DataFrame,
    signature_cols: Tuple[str, ...] = (
        "num1_c",
        "streetName_c",
        "streetSuffix_c",
        "unit_c",
        "zip_c",
        "city_c",
        "state_c",
    ),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    make sure that each AID maps to one physical location.
    """
    df_r = df_r.copy()
    df_fa_sub = df_fa[["AID", *signature_cols]]
    df_r = df_r.merge(df_fa_sub, left_on="AID_2", right_on="AID", how="left")

    # Entity‑aware signature: (EID, all address cols)
    df_r["signature"] = (
        df_r[["EID_1", *signature_cols]]
        .astype(str)
        .agg("|".join, axis=1)
    )

    # AIDs that map to >1 signature
    bad_aids = (
        df_r.groupby("AID_2")["signature"].nunique().loc[lambda s: s.gt(1)].index
    )
    if bad_aids.empty:
        return df_fa, df_r[["EID_1", "AID_2", "relationshipType", "number"]]

    next_aid = df_fa["AID"].max() + 1
    remap: dict[tuple[int, str], int] = {}

    for aid in bad_aids:
        sigs = df_r.loc[df_r["AID_2"] == aid, "signature"].unique()
        for sig in sigs:
            remap[(aid, sig)] = next_aid
            next_aid += 1

    df_r["AID_2"] = df_r.apply(
        lambda r: remap.get((r["AID_2"], r["signature"]), r["AID_2"]), axis=1
    )

    # Clone df_fa rows
    additions = []
    for (old_aid, sig), new_aid in remap.items():
        base = df_fa.loc[df_fa["AID"] == old_aid].iloc[0].copy()
        base["AID"] = new_aid
        additions.append(base)

    df_fa_out = pd.concat([df_fa, pd.DataFrame(additions)], ignore_index=True)
    df_r_out  = df_r[["EID_1", "AID_2", "relationshipType", "number"]]
    print(f"🔀 Split pass: {len(additions)} new AIDs minted due to conflicting signatures")
    return df_fa_out, df_r_out


#resolve & apply proposals
def resolve_and_apply_changes(
    df_fa_in: pd.DataFrame,
    df_r_in: pd.DataFrame,
    proposals: List[ProposedChange],
    current_max_aid: int,
    reconstructor: Callable[[pd.Series], str],
) -> tuple[DataFrame, DataFrame, int | Any, list[SplitEvent]]:
    """Apply proposed fixes; split on conflict (vectorised)."""
    df_fa = df_fa_in.copy()
    df_r  = df_r_in.copy()

    splits: list[SplitEvent] = []      #  ← NEW

    # Bucket proposals by (AID, column)
    bucket: dict[tuple[int, str], list[ProposedChange]] = defaultdict(list)
    for p in proposals:
        bucket[(p["original_AID"], p["column_to_change"])].append(p)

    for (aid, col), items in bucket.items():
        # Map proposed_value to list[EID]
        val2eids: defaultdict = defaultdict(list)
        for p in items:
            if p["EID_context"] not in val2eids[p["proposed_value"]]:
                val2eids[p["proposed_value"]].append(p["EID_context"])

        # Fetch current value quickly
        idx = df_fa.index[df_fa["AID"] == aid]
        if idx.empty:
            continue  # stale AID from earlier split
        cur_val = df_fa.at[idx[0], col]

        #NO CONFLICT
        if len(val2eids) == 1:
            new_val = next(iter(val2eids))
            if (pd.isna(cur_val) and pd.isna(new_val)) or (cur_val == new_val):
                continue  # nothing to do
            df_fa.at[idx[0], col] = new_val
            if "fullAddress_c" in df_fa.columns:
                df_fa.at[idx[0], "fullAddress_c"] = reconstructor(df_fa.loc[idx[0]])
            continue

        # IF CONFLICT, pick majority, split others
        # Sort by supporter count desc, then value for determinism
        majority_val = max(val2eids.items(), key=lambda x: (len(x[1]), str(x[0])))[0]
        if cur_val != majority_val:
            df_fa.at[idx[0], col] = majority_val
            if "fullAddress_c" in df_fa.columns:
                df_fa.at[idx[0], "fullAddress_c"] = reconstructor(df_fa.loc[idx[0]])

        # Handle minority variants
        for variant_val, supporters in val2eids.items():
            if variant_val == majority_val:
                continue
            current_max_aid += 1
            new_aid = current_max_aid

            splits.append(
                SplitEvent(old_aid=aid,
                           new_aid=new_aid,
                           column=col,
                           new_value=str(variant_val))
            )

            new_row = df_fa.loc[idx[0]].copy()
            new_row["AID"] = new_aid
            new_row[col] = variant_val
            if "fullAddress_c" in df_fa.columns:
                new_row["fullAddress_c"] = reconstructor(new_row)
            df_fa = pd.concat([df_fa, pd.DataFrame([new_row])], ignore_index=True)

            supporter_set = set([s for s in supporters if s is not None])
            mask = (df_r["AID_2"] == aid) & (df_r["EID_1"].isin(supporter_set))
            df_r.loc[mask, "AID_2"] = new_aid

    return df_fa, df_r, current_max_aid, splits


#merge helpers
def create_merged_view(df_fa: pd.DataFrame, df_r: pd.DataFrame, df_fe: pd.DataFrame) -> pd.DataFrame:
    return (
        df_r.merge(df_fe, left_on="EID_1", right_on="EID", how="left")
            .merge(df_fa, left_on="AID_2", right_on="AID", how="left")
    )

def _show_samples(rule_name: str, props: list[ProposedChange], k: int = 5) -> None:
    print(f"    {rule_name}: {len(props):,} proposals generated")
    if not props:
        return
    for p in props[:k]:
        short = {k: p[k] for k in ("original_AID", "column_to_change",
                                   "original_value", "proposed_value")}
        print(f"       • {short}")
    if len(props) > k:
        print(f"       … (+{len(props)-k:,} more)")

#entry point
def main() -> None:
    df_fa, df_r, df_fe = load_data()
    max_aid = df_fa["AID"].max()

    print("\nBuilding merged view …")
    merged = create_merged_view(df_fa, df_r, df_fe)

    proposals: list[ProposedChange] = []

    # Rule 1
    r1 = propose_fill_missing_zips_keep(merged)
    _show_samples("Rule 1  (fill missing ZIPs – keep)", r1)
    proposals += r1

    #Rule 2
    r2 = propose_street_name_corrections(merged, threshold=STREET_THRESHOLD)
    _show_samples("Rule 2  (fuzzy search street-name)", r2)
    proposals += r2

    #Rule 3a
    r3a = propose_replace_invalid_zips(merged)
    _show_samples("Rule 3a (replace invalid ZIPs)", r3a)
    proposals += r3a

    #Rule 3b
    r3b = propose_fill_missing_zips_by_address(merged)
    _show_samples("Rule 3b (fill missing ZIPs by addr)", r3b)
    proposals += r3b

    #Rule 4
    r4 = propose_correct_city_names_by_zip(merged, threshold=CITY_THRESHOLD)
    _show_samples("Rule 4  (fuzzy city by ZIP)", r4)
    proposals += r4

    print(f"\n Collected {len(proposals):,} total proposals → resolving …")
    df_fa, df_r, max_aid, split_log = resolve_and_apply_changes(
        df_fa, df_r, proposals, max_aid, reconstruct_full_address
    )

    if split_log:
        print(f"\n Total splits created: {len(split_log):,}")
        print("    (showing first 5):")
        for ev in split_log[:5]:
            print(f"     • AID {ev.old_aid} → new AID {ev.new_aid} "
                  f"(column {ev.column}, value='{ev.new_value}')")
    else:
        print("\nNo AID splits were necessary during conflict resolution.")

    #save outputs
    OUT_DIR = Path(__file__).parent.parent / "data_cleaned_split_all_rules"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_fa.to_csv(OUT_DIR / "fa_cleaned.csv", index=False)
    df_r.to_csv(OUT_DIR / "r_fe_fa_cleaned.csv", index=False)
    print(f"\nCleaned data written to {OUT_DIR if OUT_DIR.is_absolute() else OUT_DIR.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()