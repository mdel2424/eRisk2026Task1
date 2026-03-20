from __future__ import annotations

from persona.simulated_persona import SimulatedPersona
from persona.profiles import PersonaProfile, PersonaResponder


def create_persona(profile: PersonaProfile) -> PersonaResponder:
    return SimulatedPersona(
        persona_id=profile.persona_id,
        bdi_scores=profile.bdi_scores,
        evasive=True,
        family=profile.family,
        split=profile.split,
        context_tag=profile.context_tag,
        style_tag=profile.style_tag,
        behavior_params=dict(profile.behavior_params),
        template_bank=profile.template_bank,
    )
