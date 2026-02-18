from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from core.state import build_initial_state


def _load_payload(path: str | Path):
    content = Path(path).read_text(encoding="utf-8")
    return json.loads(content)


def replay_transcripts(input_path: str | Path, output_path: str | Path = "outputs/replay_diagnostics.json") -> None:
    from graph import app as graph_app

    payload = _load_payload(input_path)
    if not isinstance(payload, list):
        raise ValueError("Replay payload must be a list of persona conversation objects")

    diagnostics: List[Dict] = []
    for entry in payload:
        persona_id = str(entry.get("LLM", "unknown"))
        conversation = entry.get("conversation", [])
        if not isinstance(conversation, list):
            continue

        state = build_initial_state(persona_id=persona_id)
        timeline: List[Dict] = []
        for turn in conversation:
            role = str(turn.get("role", ""))
            message = str(turn.get("message", ""))
            if role == "assistant":
                state["messages"].append({"role": "assistant", "content": message})
                state = graph_app.invoke(state)
                timeline.append(
                    {
                        "turn": state.get("turn_index", 0),
                        "route_debug": state.get("route_debug", ""),
                        "specialist_debug": state.get("specialist_debug", ""),
                        "stop_debug": state.get("stop_debug", ""),
                        "predicted_label": state.get("predicted_label"),
                        "predicted_bdi_score": state.get("predicted_bdi_score"),
                    }
                )
                if state.get("should_stop"):
                    break

        diagnostics.append(
            {
                "LLM": persona_id,
                "timeline": timeline,
                "final_prediction": {
                    "predicted_label": state.get("predicted_label"),
                    "predicted_bdi_score": state.get("predicted_bdi_score"),
                    "global_confidence": state.get("global_confidence", 0.0),
                },
            }
        )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
