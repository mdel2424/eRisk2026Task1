from __future__ import annotations

from typing import Dict, List

from core.state import AgentState, BeliefState, ItemBelief, coerce_item_belief



def _normalize(values: List[float]) -> List[float]:
    clipped = [max(1e-8, float(v)) for v in values]
    total = sum(clipped)
    if total <= 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [value / total for value in clipped]



def _posterior_stats(posterior: List[float]) -> tuple[float, float]:
    import math

    expected = sum(idx * prob for idx, prob in enumerate(posterior))
    entropy = 0.0
    for prob in posterior:
        p = max(1e-12, min(1.0, float(prob)))
        entropy -= p * math.log2(p)
    return max(0.0, min(3.0, expected)), max(0.0, min(2.0, entropy))



def _coerce_belief(item_id: int, value) -> ItemBelief:
    return coerce_item_belief(item_id, value)



def update_beliefs(state: AgentState) -> Dict:
    turn = int(state.get("turn_index", 0))
    latest_likelihoods = list(state.get("latest_turn_likelihoods", []))
    prior_beliefs = state.get("item_beliefs", {})

    beliefs: Dict[int, ItemBelief] = {}
    for item_id in range(1, 22):
        beliefs[item_id] = _coerce_belief(item_id, prior_beliefs.get(item_id))

    evidence_count_by_item: Dict[int, int] = {}
    combined_likelihood_by_item: Dict[int, List[float]] = {}
    for row in latest_likelihoods:
        item_id = int(getattr(row, "item_id", 0) or 0)
        if item_id < 1 or item_id > 21:
            continue
        values = [float(v) for v in list(getattr(row, "likelihood", [1.0, 1.0, 1.0, 1.0]))[:4]]
        if len(values) < 4:
            values.extend([1.0] * (4 - len(values)))

        if item_id not in combined_likelihood_by_item:
            combined_likelihood_by_item[item_id] = [1.0, 1.0, 1.0, 1.0]
        combined_likelihood_by_item[item_id] = [
            combined_likelihood_by_item[item_id][idx] * max(1e-8, values[idx])
            for idx in range(4)
        ]
        evidence_count_by_item[item_id] = int(evidence_count_by_item.get(item_id, 0)) + 1

    updated_item_ids: List[int] = []
    for item_id, combined in combined_likelihood_by_item.items():
        prior = beliefs[item_id]
        prior_posterior = [float(v) for v in list(prior.posterior)[:4]]
        if len(prior_posterior) < 4:
            prior_posterior.extend([0.25] * (4 - len(prior_posterior)))

        posterior = _normalize(
            [prior_posterior[idx] * max(1e-8, combined[idx]) for idx in range(4)]
        )
        expected_score, entropy = _posterior_stats(posterior)
        beliefs[item_id] = ItemBelief(
            item_id=item_id,
            posterior=posterior,
            expected_score=expected_score,
            entropy=entropy,
            support_count=int(prior.support_count) + int(evidence_count_by_item.get(item_id, 0)),
            last_update_turn=max(0, turn),
        )
        updated_item_ids.append(item_id)

    turn_trace = dict(state.get("turn_trace", {}))
    window_size = 4
    new_items_this_turn = len(updated_item_ids)
    nonempty_this_turn = 1 if len(latest_likelihoods) > 0 else 0
    recent_new_items = list(state.get("recent_new_items_window", [])) + [new_items_this_turn]
    recent_nonempty = list(state.get("recent_nonempty_window", [])) + [nonempty_this_turn]
    if len(recent_new_items) > window_size:
        recent_new_items = recent_new_items[-window_size:]
    if len(recent_nonempty) > window_size:
        recent_nonempty = recent_nonempty[-window_size:]

    turn_trace["belief_update"] = {
        "turn": turn,
        "updated_item_ids": sorted(updated_item_ids),
        "likelihood_rows": len(latest_likelihoods),
        "new_items_this_turn": new_items_this_turn,
        "recent_new_items_window": recent_new_items,
        "recent_nonempty_window": recent_nonempty,
    }

    return {
        "beliefs": BeliefState(items=beliefs),
        "item_beliefs": beliefs,
        "new_items_this_turn": new_items_this_turn,
        "recent_new_items_window": recent_new_items,
        "recent_nonempty_window": recent_nonempty,
        "turn_trace": turn_trace,
    }
