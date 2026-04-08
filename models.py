from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any


class Ticket(BaseModel):
    ticket_id: str
    customer_name: str
    issue: str
    product: str
    history: str


class Observation(BaseModel):
    task_id: str
    ticket: Ticket


class Action(BaseModel):
    category: str = ""
    priority: str = ""
    action: str = ""
    response: str = ""


class Reward(BaseModel):
    score: float

    
    @field_validator("score")
    def validate_score(cls, v):
        try:
            v = float(v)
        except:
            return 0.02

        if v >= 1.0:
            return 0.99
        if v <= 0.0:
            return 0.01
        return v


class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: Reward
    done: bool
    info: Dict[str, Any]
