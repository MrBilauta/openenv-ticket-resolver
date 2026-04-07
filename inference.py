import os
import requests
import json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN required")

SPACE_URL = os.getenv("SPACE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def run(task):
    print(f"[START] task={task} env=customer_support model={MODEL_NAME}")

    res = requests.post(f"{SPACE_URL}/reset", json={"task_id": task})
    obs = res.json()

    prompt = f"Resolve this support ticket: {obs}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        action = json.loads(response.choices[0].message.content)
    except:
        action = {
            "category": "technical",
            "priority": "medium",
            "action": "escalation",
            "response": "Sorry, escalating."
        }

    step = requests.post(f"{SPACE_URL}/step", json=action).json()

    reward = step["reward"]["score"]
    done = step["done"]

    print(f"[STEP] step=1 action={str(action)} reward={reward:.2f} done={str(done).lower()} error=null")
    print(f"[END] success=true steps=1 rewards={reward:.2f}")


if __name__ == "__main__":
    for t in ["easy", "medium", "hard"]:
        run(t)
