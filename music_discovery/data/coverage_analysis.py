from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from music_discovery.data.normalization import normalize_name


def _diff_tokens(s1: str, s2: str) -> list[str]:
    t1, t2 = set(s1.split()), set(s2.split())
    return sorted(t1 - t2)


def analyze_coverage(
    unmatched_path: str | Path,
    catalog_df: pd.DataFrame,
    output_path: str | Path,
    near_miss_threshold: int = 70,
    top_n_pairs: int = 200,
) -> dict:
    """
    Analyze unmatched events to surface candidate normalization rules.

    Reads unmatched_events.parquet, fuzzy-searches the catalog at a low
    threshold, extracts diff tokens from near-miss pairs, and ranks them
    by event frequency → candidate strip_patterns for normalization_rules.yaml.
    """
    miss_df = pd.read_parquet(unmatched_path)
    miss_df["artist_norm"] = miss_df["artist_raw"].apply(normalize_name)
    miss_df["track_norm"] = miss_df["track_raw"].apply(normalize_name)

    # first-artist-token → [(track_norm, idx)]
    catalog_index: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for i, row in catalog_df.iterrows():
        token = row.artist_norm.split()[0] if row.artist_norm.split() else row.artist_norm
        catalog_index[token].append((row.track_norm, i))

    token_counts: dict[str, int] = defaultdict(int)
    near_miss_pairs: list[dict] = []

    unique_misses = miss_df.drop_duplicates(subset=["artist_norm", "track_norm"])

    for _, row in unique_misses.iterrows():
        artist_n, track_n = row["artist_norm"], row["track_norm"]
        event_count = int(
            miss_df[
                (miss_df["artist_norm"] == artist_n) & (miss_df["track_norm"] == track_n)
            ]["count"].sum()
        )

        token = artist_n.split()[0] if artist_n.split() else artist_n
        candidates = catalog_index.get(token, [])

        best_score, best_track = 0.0, ""
        for cand_track, _ in candidates:
            score = fuzz.token_set_ratio(track_n, cand_track)
            if score > best_score:
                best_score, best_track = score, cand_track

        if near_miss_threshold <= best_score < 85:
            diff = _diff_tokens(track_n, best_track)
            for tok in diff:
                token_counts[tok] += event_count
            near_miss_pairs.append({
                "lastfm": f"{row['artist_raw']} — {row['track_raw']}",
                "kaggle": best_track,
                "score": round(best_score, 1),
                "diff_tokens": diff,
                "event_count": event_count,
            })

    near_miss_pairs.sort(key=lambda x: x["score"], reverse=True)

    candidate_rules = sorted(
        [
            {
                "token": tok,
                "suggested_pattern": re.escape(tok),
                "events_unlocked": cnt,
            }
            for tok, cnt in token_counts.items()
        ],
        key=lambda x: x["events_unlocked"],
        reverse=True,
    )

    automaton_state_count = 5  # S0 S1_base S2 S3 S4
    rules_yaml = Path(__file__).parents[2] / "configs" / "normalization_rules.yaml"
    if rules_yaml.exists():
        import yaml
        with open(rules_yaml) as f:
            cfg = yaml.safe_load(f) or {}
        automaton_state_count += len(cfg.get("strip_patterns", [])) + (
            1 if cfg.get("artist_aliases") else 0
        )

    report = {
        "total_unmatched_events": int(miss_df["count"].sum()),
        "unique_unmatched_pairs": len(unique_misses),
        "near_miss_pairs_found": len(near_miss_pairs),
        "automaton_state_count": automaton_state_count,
        "candidate_rules": candidate_rules[:50],
        "near_miss_pairs": near_miss_pairs[:top_n_pairs],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[coverage] {report['total_unmatched_events']:,} unmatched events")
    print(f"[coverage] {len(near_miss_pairs):,} near-miss pairs (score {near_miss_threshold}–84)")
    print("[coverage] Top candidate rules:")
    for r in candidate_rules[:10]:
        print(f"  +{r['events_unlocked']:,} events  token='{r['token']}'  pattern='{r['suggested_pattern']}'")
    print(f"[coverage] Report saved → {output_path}")

    return report
