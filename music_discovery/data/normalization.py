
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Callable


_INLINE_NOISE = re.compile(
    r"\s*-\s*(?:remaster(?:ed)?|radio\s*edit|mono|stereo|live|explicit|clean|single\s*version).*",
    re.IGNORECASE,
)

_FEAT = re.compile(r"\s+(?:feat\.?|ft\.?|featuring)\s+.*", re.IGNORECASE)
_AMP = re.compile(r"\s*&\s*")
_APOS = re.compile(r"[''`]")
_DASH = re.compile(r"[–—]")
_PUNCT = re.compile(r"[^\w\s\-']")
_SPACE = re.compile(r"\s+")
_PAREN = re.compile(r"\s*[\(\[].*?[\)\]]")

def normalize_name(s: object) -> str:
    if not isinstance(s, str): return ""
    
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = _PAREN.sub("", s)          
    s = _INLINE_NOISE.sub("", s)
    s = _FEAT.sub("", s)
    s = _AMP.sub(" and ", s)
    s = _APOS.sub("", s)
    s = _DASH.sub("-", s)
    s = _PUNCT.sub("", s)
    return _SPACE.sub(" ", s).strip()

def load_rule_normalizers(rules_path: str | Path) -> list[Callable[[str], str]]:
    path = Path(rules_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalization rules YAML not found: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load normalization rules. Install pyyaml to use rule-based matching."
        ) from exc

    with open(path, "r") as f:
        rules = yaml.safe_load(f) or {}

    fns: list[Callable[[str], str]] = []
    
    _DASH_ARTIFACT = re.compile(r"^\s*-\s*|\s*-\s*$")

    for pattern in rules.get("strip_patterns", []):
        compiled = re.compile(pattern, re.IGNORECASE)

        def make_strip(rx: re.Pattern) -> Callable[[str], str]:
            def strip_fn(s: str) -> str:
                cleaned = _SPACE.sub(" ", rx.sub("", s)).strip()
                return _DASH_ARTIFACT.sub("", cleaned).strip()
            return strip_fn
        
        fns.append(make_strip(compiled))
    
    aliases: list[tuple[str, str]] = [
        (a[0], a[1]) for a in rules.get("artist_aliases", []) if len(a) == 2
    ]
    
    if aliases:
        def alias_fn(s: str) -> str:
            for a, b in aliases:
                if s == a:
                    return b
            return s
        
        fns.append(alias_fn)
    
    return fns
