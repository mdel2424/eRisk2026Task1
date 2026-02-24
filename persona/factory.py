from __future__ import annotations

from core.runtime_policy import resolve_persona_backend
from persona.openrouter_persona import OpenRouterSimPersona
from persona.profiles import PersonaProfile, PersonaResponder


def create_persona(profile: PersonaProfile) -> PersonaResponder:
    backend = resolve_persona_backend()
    if backend != "openrouter_sim":
        raise ValueError(
            f"Persona backend '{backend}' disabled: deterministic simulator is the only supported persona backend in this build."
        )
    return OpenRouterSimPersona(
        persona_id=profile.persona_id,
        bdi_scores=profile.bdi_scores,
        evasive=True,
        family=profile.family,
        split=profile.split,
        behavior_params=dict(profile.behavior_params),
        template_bank=profile.template_bank,
        generator_version=profile.generator_version,
    )
