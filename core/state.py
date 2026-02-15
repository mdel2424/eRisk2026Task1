from typing import Annotated, List, TypedDict, Optional
import operator
from pydantic import BaseModel

class SymptomFinder(BaseModel):
    item_id: int
    score: int
    confidence: float

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    findings: Annotated[List[SymptomFinder], operator.add]
    next_node: str