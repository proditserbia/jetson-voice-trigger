from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


DEFAULT_TRIGGERS: Dict[str, str] = {
    "open browser": "xdg-open https://www.wikipedia.org",
    "turn off screen": "bash -lc 'xset dpms force off'",
    "say hello": "bash -lc 'notify-send \"Trigger activated\" \"Hello from voice trigger\"'",
}


def load_triggers_json(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return dict(DEFAULT_TRIGGERS)
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Triggers JSON must be an object: {\"phrase\": \"command\", ...}")
    out: Dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("Triggers JSON must map string -> string")
        out[k] = v
    return out
