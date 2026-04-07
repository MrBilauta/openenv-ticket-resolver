from pydantic import BaseModel
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
    category: str
    priority: str
    action: str
    response: str


class Reward(BaseModel):
    score: float


class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: Reward
    done: bool
    info: Dict[str, Any]
