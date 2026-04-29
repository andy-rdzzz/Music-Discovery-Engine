"""Benchmark and score Last.fm -> Spotify matching quality."""

from __future__ import annotations

import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from music_discovery.data.kaggle_spotify import load_kaggle_tracks, _normalize_name as normalize_kaggle_name


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _single_typo(text: str, rng: random.Random) -> str:
    tokens = [t for t in re.split(r"(\s+)", text) if t]
    word_positions = [i for i, token in enumerate(tokens) if not token.isspace() and len(token) >= 5]
    if not word_positions:
        return text
    idx = rng.choice(word_positions)
    token = tokens[idx]
    if len(token) >= 6 and rng.random() < 0.5:
        pos = rng.randrange(1, len(token) - 2)
        token = token[:pos] + token[pos + 1] + token[pos] + token[pos + 2 :]
    else:
        pos = rng.randrange(1, len(token) - 1)
        token = token[:pos] + token[pos + 1 :]
    tokens[idx] = token
    return "".join(tokens)


def _append_version_suffix(track: str, rng: random.Random) -> str:
    suffix = rng.choice([" - Remastered", " - Live", " - Radio Edit", " - Mono Version"])
    return f"{track}{suffix}"


def _swap_and_symbol(text: str) -> str | None:
    lowered = text.lower()
    if "&" in text:
        return text.replace("&", "and")
    if " and " in lowered:
        return re.sub(r"\band\b", "&", text, count=1, flags=re.IGNORECASE)
    return None


def _punctuation_variant(text: str) -> str | None:
    new_text = text.replace("'", "").replace("’", "").replace("–", "-").replace("—", "-")
    new_text = re.sub(r"[,:;.!?]+", "", new_text)
    new_text = re.sub(r"\s+", " ", new_text).strip()
    return new_text if new_text != text else None


def _make_query_variant(
    artist: str,
    track: str,
    perturbation: str,
    rng: random.Random,
) -> tuple[str, str]:
    if perturbation == "raw":
        return artist, track
    if perturbation == "artist_and_symbol":
        return _swap_and_symbol(artist) or artist, track
    if perturbation == "artist_punctuation":
        return _punctuation_variant(artist) or _strip_accents(artist), track
    if perturbation == "track_punctuation":
        return artist, _punctuation_variant(track) or _strip_accents(track)
    if perturbation == "track_version_suffix":
        return artist, _append_version_suffix(track, rng)
    if perturbation == "track_typo":
        return artist, _single_typo(track, rng)
    if perturbation == "artist_typo":
        return _single_typo(artist, rng), track
    raise ValueError(f"Unknown perturbation: {perturbation}")


def _difficulty_for(perturbation: str) -> str:
    if perturbation in {"raw", "artist_punctuation", "track_punctuation"}:
        return "easy"
    if perturbation in {"artist_and_symbol", "track_version_suffix"}:
        return "medium"
    return "hard"


def _build_exact_matcher(catalog_df: pd.DataFrame) -> Callable[[str, str], dict[str, object]]:
    lookup = {
        (row.artist_norm, row.track_norm): {
            "predicted_track_idx": int(row.track_idx),
            "predicted_artist_norm": row.artist_norm,
            "predicted_track_norm": row.track_norm,
            "score": 1.0,
            "accepted": True,
        }
        for row in catalog_df.itertuples()
    }

    def match(artist: str, track: str) -> dict[str, object]:
        key = (normalize_kaggle_name(artist), normalize_kaggle_name(track))
        return lookup.get(
            key,
            {
                "predicted_track_idx": None,
                "predicted_artist_norm": None,
                "predicted_track_norm": None,
                "score": 0.0,
                "accepted": False,
            },
        )

    return match


@dataclass
class BenchmarkArtifacts:
    benchmark_df: pd.DataFrame
    baseline_predictions_df: pd.DataFrame
    summary: dict[str, object]
    breakdown_df: pd.DataFrame


def build_benchmark(
    kaggle_csv: str,
    lastfm_tsv: str,
    output_csv: str | Path,
    n_records: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    if not 100 <= n_records <= 300:
        raise ValueError("n_records must be between 100 and 300")

    rng = random.Random(seed)
    catalog_df = load_kaggle_tracks(kaggle_csv).reset_index(drop=True)
    catalog_df["track_idx"] = range(len(catalog_df))
    candidate_df = catalog_df[
        catalog_df["track_name"].astype(str).str.len() >= 3
    ].copy()
    if len(candidate_df) < n_records:
        raise ValueError(f"Only found {len(candidate_df)} candidate tracks, need at least {n_records}")

    weights = {
        "raw": 0.20,
        "artist_punctuation": 0.15,
        "track_punctuation": 0.15,
        "artist_and_symbol": 0.10,
        "track_version_suffix": 0.20,
        "track_typo": 0.10,
        "artist_typo": 0.10,
    }
    perturbation_plan: list[str] = []
    for perturbation, weight in weights.items():
        perturbation_plan.extend([perturbation] * int(round(n_records * weight)))
    while len(perturbation_plan) < n_records:
        perturbation_plan.append(rng.choice(list(weights)))
    if len(perturbation_plan) > n_records:
        perturbation_plan = perturbation_plan[:n_records]
    rng.shuffle(perturbation_plan)

    same_artist_counts = catalog_df["artist_norm"].value_counts()
    same_track_counts = catalog_df["track_norm"].value_counts()

    sampled = candidate_df.sample(n=n_records, random_state=seed).reset_index(drop=True)
    records: list[dict[str, object]] = []

    for i, row in sampled.iterrows():
        perturbation = perturbation_plan[i]
        query_artist, query_track = _make_query_variant(
            row["artists"],
            row["track_name"],
            perturbation,
            rng,
        )
        records.append(
            {
                "benchmark_id": i,
                "difficulty": _difficulty_for(perturbation),
                "perturbation": perturbation,
                "query_artist": query_artist,
                "query_track": query_track,
                "query_artist_norm": normalize_kaggle_name(query_artist),
                "query_track_norm": normalize_kaggle_name(query_track),
                "target_artist_norm": row["artist_norm"],
                "target_track_norm": row["track_norm"],
                "target_track_idx": int(row["track_idx"]),
                "canonical_artist": row["artists"],
                "canonical_track": row["track_name"],
                "n_matched_events": None,
                "same_artist_candidates": int(same_artist_counts[row["artist_norm"]]),
                "same_track_candidates": int(same_track_counts[row["track_norm"]]),
            }
        )

    benchmark_df = pd.DataFrame(records)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(output_path, index=False)
    return benchmark_df


def score_predictions(
    benchmark_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    scored = benchmark_df.merge(predictions_df, on="benchmark_id", how="left")
    scored["accepted"] = scored["accepted"].fillna(False).astype(bool)
    scored["correct"] = (
        scored["accepted"]
        & (
            (scored["predicted_track_idx"].notna() & (scored["predicted_track_idx"] == scored["target_track_idx"]))
            | (
                scored["predicted_track_idx"].isna()
                & scored["predicted_artist_norm"].fillna("").eq(scored["target_artist_norm"])
                & scored["predicted_track_norm"].fillna("").eq(scored["target_track_norm"])
            )
        )
    )

    total = len(scored)
    accepted = int(scored["accepted"].sum())
    correct = int(scored["correct"].sum())
    summary = {
        "n_queries": total,
        "accepted": accepted,
        "correct": correct,
        "success_at_1": correct / total if total else 0.0,
        "precision_at_accepted": correct / accepted if accepted else 0.0,
        "coverage_at_threshold": accepted / total if total else 0.0,
    }

    breakdown = (
        scored.groupby(["difficulty", "perturbation"], dropna=False)
        .agg(
            n_queries=("benchmark_id", "size"),
            accepted=("accepted", "sum"),
            correct=("correct", "sum"),
        )
        .reset_index()
    )
    breakdown["success_at_1"] = breakdown["correct"] / breakdown["n_queries"]
    breakdown["precision_at_accepted"] = breakdown["correct"] / breakdown["accepted"].where(
        breakdown["accepted"] > 0, 1
    )
    breakdown["coverage_at_threshold"] = breakdown["accepted"] / breakdown["n_queries"]
    breakdown["precision_at_accepted"] = breakdown["precision_at_accepted"].where(
        breakdown["accepted"] > 0, 0.0
    )
    return summary, breakdown


def run(
    kaggle_csv: str,
    lastfm_tsv: str,
    benchmark_csv: str,
    output_dir: str,
    n_records: int = 200,
    seed: int = 42,
    predictions_csv: str | None = None,
    regenerate: bool = False,
) -> BenchmarkArtifacts:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    benchmark_path = Path(benchmark_csv)

    if regenerate or not benchmark_path.exists():
        benchmark_df = build_benchmark(kaggle_csv, lastfm_tsv, benchmark_path, n_records=n_records, seed=seed)
    else:
        benchmark_df = pd.read_csv(benchmark_path)

    catalog_df = load_kaggle_tracks(kaggle_csv).reset_index(drop=True)
    catalog_df["track_idx"] = range(len(catalog_df))

    if predictions_csv is None:
        matcher = _build_exact_matcher(catalog_df)
        baseline_predictions_df = pd.DataFrame(
            [
                {"benchmark_id": int(row.benchmark_id), **matcher(row.query_artist, row.query_track)}
                for row in benchmark_df.itertuples()
            ]
        )
    else:
        baseline_predictions_df = pd.read_csv(predictions_csv)

    summary, breakdown_df = score_predictions(benchmark_df, baseline_predictions_df)

    benchmark_df.to_csv(output / "matching_benchmark.csv", index=False)
    baseline_predictions_df.to_csv(output / "matching_predictions_scored.csv", index=False)
    breakdown_df.to_csv(output / "matching_quality_breakdown.csv", index=False)
    with open(output / "matching_quality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[matching_quality] Benchmark rows:   {summary['n_queries']}")
    print(f"[matching_quality] Accepted:         {summary['accepted']}")
    print(f"[matching_quality] Correct:          {summary['correct']}")
    print(f"[matching_quality] Success@1:        {summary['success_at_1']:.3f}")
    print(f"[matching_quality] Precision@accept: {summary['precision_at_accepted']:.3f}")
    print(f"[matching_quality] Coverage@τ:       {summary['coverage_at_threshold']:.3f}")
    print(f"[matching_quality] Saved artifacts to {output}")

    return BenchmarkArtifacts(
        benchmark_df=benchmark_df,
        baseline_predictions_df=baseline_predictions_df,
        summary=summary,
        breakdown_df=breakdown_df,
    )
