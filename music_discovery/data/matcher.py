from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz

from music_discovery.data.normalization import normalize_name, load_rule_normalizers

_DEFAULT_RULES = Path(__file__).parents[2] / "configs" / "normalization_rules.yaml"
TRAIN_MATCH_TYPES = {"exact", "normalized_exact", "fuzzy_high"}

_ARTICLE_STOPS = {"the", "a", "an", "of", "my", "la", "le", "los", "las", "el", "de"}

@dataclass
class MatchResult:
    track_idx: Optional[int]
    match_type: Optional[str]
    match_score: Optional[float]
    is_ambiguous: bool = False
    
class TrackMatcher:
    def __init__(
        self,
        catalog_df: pd.DataFrame,
        rules_path: str | Path = _DEFAULT_RULES
    ) -> None:
        
        # s0: index
        self._exact: dict[tuple[str, str], int] = {
            (row.artist_norm, row.track_norm): i
            for i, row in enumerate(catalog_df.itertuples(index=False))
        }
        
        #s2: block: artist norm → (artist_norm, track_norm, idx)
        self.artist_blocks: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        #s2.5: block: artist_norm stripped of leading "the " → entries
        self._no_the_block: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        #s3: block: first artist token → entries
        self._token_block: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        #s3.5: block: last artist token (multi-word artists only) → entries
        self._last_token_block: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        for i, row in enumerate(catalog_df.itertuples(index=False)):
            a, t = row.artist_norm, row.track_norm
            self.artist_blocks[a].append((a, t, i))
            parts = a.split()
            token = parts[0] if parts else a
            self._token_block[token].append((a, t, i))
            if a.startswith("the "):
                self._no_the_block[a[4:]].append((a, t, i))
            if len(parts) > 1:
                self._last_token_block[parts[-1]].append((a, t, i))
            
        self._extra: list[callable[[str], str]] = load_rule_normalizers(rules_path) 
        
        
    def match(self, artist_raw: str, track_raw: str) -> MatchResult | None:
        artist_n = normalize_name(artist_raw)
        track_n = normalize_name(track_raw)
        return self.match_normalized(artist_n, track_n)

    def match_normalized(self, artist_n: str, track_n: str) -> MatchResult | None:

        #S0: exact
        idx = self._exact.get((artist_n, track_n))
        if idx is not None:
            return MatchResult(idx, "exact", 100.0)

        #S1: extra rules
        for rule_fn in self._extra:
            a2, t2 = rule_fn(artist_n), rule_fn(track_n)
            idx = self._exact.get((a2, t2))
            if idx is not None:
                return MatchResult(idx, "normalized_exact", 100.0)

        #S2: blocked fuzzy (exact artist_norm, high confidence)
        result = self._score_candidates(self.artist_blocks.get(artist_n, ()), artist_n, track_n, exact_artist=True, min_score=92)
        if result:
            return result

        #S2.5: "the"-article block — handles "beatles" ↔ "the beatles"
        the_key = artist_n[4:] if artist_n.startswith("the ") else artist_n
        result = self._score_candidates(self._no_the_block.get(the_key, ()), artist_n, track_n, exact_artist=False, min_score=90)
        if result:
            return result

        #S3: first-token block (medium confidence)
        parts = artist_n.split()
        token = parts[0] if parts else artist_n
        result = self._score_candidates(self._token_block.get(token, ()), artist_n, track_n, exact_artist=False, min_score=85)
        if result:
            return result

        #S3.5: last-token block for stop-word-led artists (e.g., "the rolling stones" → "stones")
        if parts and parts[0] in _ARTICLE_STOPS and len(parts) > 1:
            result = self._score_candidates(self._last_token_block.get(parts[-1], ()), artist_n, track_n, exact_artist=False, min_score=88)
            if result and not result.is_ambiguous:
                return result

        #S4: reject
        return None

    def _score_candidates(
        self,
        candidates: list[tuple[str, str, int]],
        artist_n: str,
        track_n: str,
        exact_artist: bool,
        min_score: int,
    ) -> MatchResult | None:
        if not candidates:
            return None

        best_score, best_idx, second_score = 0.0, -1, 0.0
        for cand_artist, cand_track, idx in candidates:
            a_score = 100.0 if exact_artist else fuzz.token_sort_ratio(artist_n, cand_artist)
            t_score = fuzz.token_set_ratio(track_n, cand_track)
            len_pen = max(0, abs(len(track_n) - len(cand_track)) - 5) * 2
            combined = 0.4 * a_score + 0.6 * t_score - len_pen
            if combined > best_score:
                second_score = best_score
                best_score, best_idx = combined, idx
            elif combined > second_score:
                second_score = combined

        if best_idx == -1 or best_score < min_score:
            return None

        match_type = "fuzzy_high" if best_score >= 92 else "fuzzy_medium"
        is_ambiguous = (best_score - second_score) <= 3.0
        return MatchResult(best_idx, match_type, round(best_score, 2), is_ambiguous)
