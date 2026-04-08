import random
from typing import Dict, Any, Tuple, Optional

from models import Observation, Ticket, Reward
from tasks import EASY_TASKS, MEDIUM_TASKS, HARD_TASKS
from graders import (
    grade_category,
    grade_priority,
    grade_action,
    grade_response
)


class CustomerSupportEnv:
    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id
        self.current: Optional[Dict[str, Any]] = None
        self.done: bool = False

    def reset(self) -> Dict[str, Any]:
        self.done = False

        if self.task_id == "easy":
            self.current = random.choice(EASY_TASKS)
        elif self.task_id == "medium":
            self.current = random.choice(MEDIUM_TASKS)
        else:
            self.current = random.choice(HARD_TASKS)

        ticket = Ticket(
            ticket_id=self.current["ticket_id"],
            customer_name=self.current["customer_name"],
            issue=self.current["issue"],
            product=self.current["product"],
            history=self.current["history"]
        )

        observation = Observation(
            task_id=self.task_id,
            ticket=ticket
        )

        return observation.dict()

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "done": self.done,
            "has_ticket": self.current is not None
        }

    def step(self, action: Dict[str, Any]) -> Tuple[None, Dict[str, Any], bool, Dict[str, Any]]:

        if self.done:
            return None, {"score": 0.02}, True, {"error": "episode_already_done"}

        if self.current is None:
            return None, {"score": 0.02}, True, {"error": "no_active_ticket"}

        expected = self.current["expected"]

        category = action.get("category", "")
        priority = action.get("priority", "")
        chosen_action = action.get("action", "")
        response = action.get("response", "")

        
        def safe_score(x):
            if x >= 1.0:
                return 0.98
            elif x <= 0.0:
                return 0.02
            return x

        category_score = safe_score(grade_category(category, expected["category"]))
        priority_score = safe_score(grade_priority(priority, expected["priority"]))
        action_score = safe_score(grade_action(chosen_action, expected["action"]))
        response_score = safe_score(grade_response(response))

        
        reward_value = (
            0.15 * category_score +
            0.15 * priority_score +
            0.2 * action_score +
            0.5 * response_score
        )

        
        
        reward_value = max(min(reward_value, 0.98), 0.02)

        
        reward_value = float(f"{reward_value:.2f}")

        
        if reward_value >= 1.0:
            reward_value = 0.98
        if reward_value <= 0.0:
            reward_value = 0.02

        self.done = True

        reward = Reward(score=reward_value)

        info = {
            "expected": expected,
            "received": {
                "category": category,
                "priority": priority,
                "action": chosen_action
            },
            "breakdown": {
                "category": category_score,
                "priority": priority_score,
                "action": action_score,
                "response": response_score
            }
        }

        return None, reward.dict(), True, info
