from fastapi import FastAPI, Body, Request
import uvicorn
from models import Action
from environment import CustomerSupportEnv

app = FastAPI()
env = CustomerSupportEnv()


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/reset")
def reset(data: dict = Body(default={})):
    task = data.get("task_id", "easy")
    global env
    env = CustomerSupportEnv(task)
    return env.reset()


@app.get("/state")
def state():
    return env.state()



@app.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
    except:
        body = {}

    action = Action(
        category=body.get("category", ""),
        priority=body.get("priority", ""),
        action=body.get("action", ""),
        response=body.get("response", "")
    )

    try:
        obs, reward, done, info = env.step(action.dict())
    except:
        return {
            "observation": None,
            "reward": {"score": 0.02},
            "done": True,
            "info": {"error": "fallback"}
        }

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
