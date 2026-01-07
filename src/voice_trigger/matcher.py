from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from rapidfuzz import fuzz, process

from .utils import normalize_text


@dataclass
class MatchResult:
    phrase: str
    command: str
    score: int


class PhraseMatcher:
    def __init__(
        self,
        triggers: Dict[str, str],
        threshold: int = 85,
        cooldown_sec: float = 4.0,
        require_all_tokens: bool = True,
        min_chars: int = 3,
    ):
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec
        self.require_all_tokens = require_all_tokens
        self.min_chars = min_chars

        self.triggers: Dict[str, str] = {}
        self.tokens: Dict[str, set[str]] = {}
        for phrase, cmd in triggers.items():
            p = normalize_text(phrase)
            self.triggers[p] = cmd
            self.tokens[p] = set(p.split())

        self._cooldowns: Dict[str, float] = {}

    def match(self, text: str) -> Tuple[Optional[MatchResult], int]:
        txt = normalize_text(text)
        if not txt or len(txt) < self.min_chars:
            return None, 0

        txt_tokens = set(txt.split())

        candidates = []
        for phrase in self.triggers.keys():
            phrase_tokens = self.tokens[phrase]
            if self.require_all_tokens and len(phrase_tokens) > 1:
                if not phrase_tokens.issubset(txt_tokens):
                    continue
            candidates.append(phrase)

        if not candidates:
            return None, 0

        best = process.extractOne(txt, candidates, scorer=fuzz.ratio)
        if not best:
            return None, 0

        match_phrase, score, _ = best
        score_i = int(score)

        if score_i >= self.threshold:
            now = time.time()
            last = self._cooldowns.get(match_phrase, 0.0)
            if (now - last) >= self.cooldown_sec:
                self._cooldowns[match_phrase] = now
                return MatchResult(match_phrase, self.triggers[match_phrase], score_i), score_i

        return None, score_i
