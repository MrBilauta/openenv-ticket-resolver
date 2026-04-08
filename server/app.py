from fastapi import FastAPI, Body
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
def step(action: Action = Body(default={})):
    obs, reward, done, info = env.step(action.dict())
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
