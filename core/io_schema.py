from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    message: str


class PersonaConversation(BaseModel):
    LLM: str
    conversation: List[Turn]


class PersonaResult(BaseModel):
    LLM: str
    bdi_score: int = Field(ge=0, le=63)
    key_symptoms: List[str] = Field(default_factory=list, max_length=4)
    item_scores: Dict[str, int] = Field(default_factory=dict)

    def to_erisk_dict(self) -> dict:
        return {
            "LLM": self.LLM,
            "bdi-score": self.bdi_score,
            "key-symptoms": self.key_symptoms,
            "item-scores": self.item_scores,
        }
