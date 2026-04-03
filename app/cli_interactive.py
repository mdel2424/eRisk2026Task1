from __future__ import annotations

from typing import Any, Dict, List

from core.llm import LLMBudgetExceeded, get_llm_usage, reset_llm_usage, set_llm_call_budget
from core.state import BDI_ITEM_NAMES, build_initial_state
from persona import PersonaProfile, generate_persona_pool, create_persona

from app.cli_runtime import _build_probe_intent
from app.cli_runtime_helpers import _assert_detector_backend_ready, _print_backend_info

PIPELINE_ORDER = (
    "ingest_turn -> judgment_agent -> bayes_state_update -> diagnosis_agent -> "
    "stop_controller -> navigation_agent -> question_agent"
)


def _all_profiles(persona_count: int, seed: int) -> List[PersonaProfile]:
    rows = generate_persona_pool(count=persona_count, seed=seed)

    def _sort_key(profile: PersonaProfile) -> tuple[int, str]:
        try:
            return int(profile.persona_id), profile.persona_id
        except Exception:
            return 10**9, profile.persona_id

    return sorted(rows, key=_sort_key)


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _short_list(values: List[Any], limit: int = 5) -> str:
    if not values:
        return "-"
    if len(values) <= limit:
        return ",".join(str(v) for v in values)
    head = ",".join(str(v) for v in values[:limit])
    return f"{head},+{len(values)-limit}"


def _evidence_preview(state: Dict[str, Any], limit: int = 4) -> str:
    rows = list(state.get("latest_turn_evidence", []))
    if not rows:
        return "-"
    out: List[str] = []
    for row in rows[:limit]:
        data = _as_dict(row)
        item_id = int(data.get("item_id", 0) or 0)
        symptom = BDI_ITEM_NAMES.get(item_id, f"item{item_id}")
        conf = float(data.get("confidence", 0.0) or 0.0)
        intensity = float(data.get("intensity", 0.0) or 0.0)
        method = str(data.get("method", "") or "")
        out.append(f"{item_id}:{symptom} i={intensity:.1f} c={conf:.2f} {method}".strip())
    return " | ".join(out)


def _usage_line() -> str:
    usage = get_llm_usage()
    calls = int(usage.get("calls_total", 0))
    max_calls = usage.get("max_calls")
    errs = int(usage.get("errors_total", 0))
    if max_calls is None:
        return f"usage calls={calls}/inf errors={errs}"
    return f"usage calls={calls}/{int(max_calls)} errors={errs}"


def _next_action_value(state: Dict[str, Any], key: str, default: Any) -> Any:
    action = state.get("next_action")
    if isinstance(action, dict):
        return action.get(key, default)
    return getattr(action, key, default)


def _print_detector_step(step: int, state: Dict[str, Any]) -> None:
    trace = _as_dict(state.get("turn_trace", {}))
    ingest = _as_dict(trace.get("ingest_turn"))
    judgment = _as_dict(trace.get("judgment_agent"))
    bayes = _as_dict(trace.get("bayes_state_update"))
    diagnosis = _as_dict(trace.get("diagnosis_agent"))
    navigation = _as_dict(trace.get("navigation_agent"))
    stop = _as_dict(trace.get("stop_controller"))

    detector_message = ""
    messages = list(state.get("messages", []))
    if messages and messages[-1].get("role") == "user":
        detector_message = str(messages[-1].get("content", "")).strip()

    print(f"\n[Detector step {step}]")
    print(
        "  ingest: "
        f"turn={int(ingest.get('turn', state.get('turn_index', 0)))} "
        f"new_persona={bool(ingest.get('has_new_persona_input', False))}"
    )
    print(
        "  judgment: "
        f"cluster={str(judgment.get('active_cluster', '-'))} "
        f"bound_pos={int(judgment.get('bound_positive_assertion_count', 0) or 0)} "
        f"binding={float(judgment.get('evidence_binding_coverage', 1.0) or 1.0):.2f}"
    )
    print(
        "  bayes: "
        f"expected_bdi={float(bayes.get('total_expected_bdi', 0.0) or 0.0):.2f} "
        f"top_uncertain={_short_list(list(bayes.get('top_uncertain_items', []) or []), limit=4)} "
        f"risk={float(state.get('risk_prob', 0.0)):.2f}"
    )
    print(f"  evidence: {_evidence_preview(state)}")
    print(
        "  diagnosis: "
        f"label={str(diagnosis.get('predicted_label', state.get('predicted_label', '-')))} "
        f"bdi={int(diagnosis.get('total_bdi', state.get('predicted_bdi_score', 0)) or 0)} "
        f"conf={float(diagnosis.get('confidence', state.get('global_confidence', 0.0)) or 0.0):.2f}"
    )
    print(
        "  navigation: "
        f"route={str(navigation.get('route', state.get('next_node', '-')))} "
        f"target={int(_next_action_value(state, 'target_item_id', 0) or 0)} "
        f"question_kind={str(navigation.get('question_kind', '-'))} "
        f"urgency={str(navigation.get('urgency_mode', '-'))}"
    )
    print(
        "  stop: "
        f"should_stop={bool(stop.get('should_stop', state.get('should_stop', False)))} "
        f"reason={str(stop.get('reason', '-'))}"
    )
    if detector_message:
        print(f"  detector> {detector_message}")
    print(f"  {_usage_line()}")


def _print_persona_step(
    step: int,
    state: Dict[str, Any],
    persona_reply: str,
    probe_intent: Dict[str, Any],
    style_stats: Dict[str, Any],
) -> None:
    print(f"\n[Persona step {step}]")
    print(
        "  handoff: "
        f"route={probe_intent['route']} "
        f"target_item={probe_intent['target_item_id']} "
        f"style={probe_intent['style']} "
        f"mode={probe_intent['mode']} "
        f"directness={probe_intent['directness']} "
        f"priority={float(probe_intent['priority']):.2f}"
    )
    print(f"  persona> {persona_reply}")
    if style_stats:
        print(
            "  style: "
            f"n={int(style_stats.get('responses_total', 0))} "
            f"hedged={int(style_stats.get('hedged_response_count', 0))} "
            f"deflect={int(style_stats.get('deflect_response_count', 0))} "
            f"context={int(style_stats.get('context_anchor_count', 0))} "
            f"avg_words={float(style_stats.get('avg_response_words', 0.0)):.1f}"
        )
    print(f"  {_usage_line()}")


def run_interactive(
    *,
    persona_count: int,
    seed: int,
    persona_index: int,
    show_ground_truth: bool,
    max_api_calls: int,
) -> None:
    from graph import app as graph_app

    _assert_detector_backend_ready()
    set_llm_call_budget(max_api_calls if max_api_calls > 0 else None)
    reset_llm_usage()
    _print_backend_info(max_api_calls=max_api_calls if max_api_calls > 0 else None, trace_level="compact")

    profiles = _all_profiles(persona_count=persona_count, seed=seed)
    if not profiles:
        raise ValueError("No personas available for interactive mode.")
    if persona_index < 0 or persona_index >= len(profiles):
        raise ValueError(
            f"interactive_persona_index out of range: {persona_index}. Available range: 0..{len(profiles)-1}"
        )

    profile = profiles[persona_index]
    persona = create_persona(profile)
    state = build_initial_state(persona_id=profile.persona_id)

    print(
        "\nInteractive stepper controls: "
        "press Enter to advance one step (detector/persona alternates), 'q' to quit."
    )
    print(f"Pipeline: {PIPELINE_ORDER}")
    print(
        f"Selected persona: id={profile.persona_id} split={profile.split} family={profile.family}"
    )
    if show_ground_truth:
        print(
            f"Ground truth: depressed={profile.depressed} bdi_total={profile.bdi_total} "
            f"risk_signal={profile.has_risk_signal} key_symptoms={profile.key_symptoms}"
        )

    next_actor = "detector"
    step = 0

    while True:
        if state.get("should_stop"):
            print(
                "\nConversation stopped by policy: "
                f"label={state.get('predicted_label')} "
                f"bdi={int(state.get('predicted_bdi_score') or 0)} "
                f"risk={bool(state.get('risk_flag', False))}"
            )
            print(f"Top symptoms: {list(state.get('predicted_key_symptoms', []))[:4]}")
            print(_usage_line())
            return

        raw = input(f"\n[{next_actor}] Enter to advance ('q' to quit): ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            print("Exiting interactive mode.")
            return
        if raw:
            print("Unknown command. Use Enter to advance or 'q' to quit.")
            continue

        step += 1
        if next_actor == "detector":
            try:
                state = graph_app.invoke(state)
            except LLMBudgetExceeded as exc:
                print(f"\nBudget exceeded during detector step: {exc}")
                print(_usage_line())
                return

            _print_detector_step(step, state)
            next_actor = "persona"
            continue

        probe_intent = _build_probe_intent(state)
        try:
            persona_reply = persona.reply(state["messages"], probe_intent)
        except LLMBudgetExceeded as exc:
            print(f"\nBudget exceeded during persona step: {exc}")
            print(_usage_line())
            return

        state["messages"].append({"role": "assistant", "content": persona_reply})
        style_stats = {}
        if hasattr(persona, "style_stats"):
            try:
                maybe_stats = persona.style_stats()
                if isinstance(maybe_stats, dict):
                    style_stats = maybe_stats
            except Exception:
                style_stats = {}

        _print_persona_step(step, state, persona_reply, probe_intent, style_stats)
        next_actor = "detector"
