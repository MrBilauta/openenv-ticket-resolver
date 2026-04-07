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
        """Reset environment and return initial observation"""
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
        """Return current environment state"""
        return {
            "task_id": self.task_id,
            "done": self.done,
            "has_ticket": self.current is not None
        }

    def step(self, action: Dict[str, Any]) -> Tuple[None, Dict[str, Any], bool, Dict[str, Any]]:
        """
        Perform one step in environment
        Returns: observation, reward, done, info
        """

        if self.done:
            return None, {"score": 0.0}, True, {"error": "episode_already_done"}

        if self.current is None:
            return None, {"score": 0.0}, True, {"error": "no_active_ticket"}

        expected = self.current["expected"]

        # Extract agent outputs safely
        category = action.get("category", "")
        priority = action.get("priority", "")
        chosen_action = action.get("action", "")
        response = action.get("response", "")

        # Grading (each returns 0.0 or 1.0 or partial)
        category_score = grade_category(category, expected["category"])
        priority_score = grade_priority(priority, expected["priority"])
        action_score = grade_action(chosen_action, expected["action"])
        response_score = grade_response(response)

        # Weighted reward (total = 1.0)
        reward_value = (
            0.2 * category_score +
            0.2 * priority_score +
            0.2 * action_score +
            0.4 * response_score
        )

        reward_value = round(min(max(reward_value, 0.0), 1.0), 2)

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
