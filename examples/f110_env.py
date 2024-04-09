import time
from typing import Tuple

import gymnasium as gym
import numpy as np



env = gym.make(
    "f1tenth_gym:f1tenth-v0",
    config={
        "map": "Spielberg",
        "num_agents": 1,
        "timestep": 0.01,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "model": "st",
        "observation_config": {"type": "original"},
        "params": {"mu": 1.0, "num_beams": 1080, "fov": 4.7},
        "reset_config": {"type": "rl_random_static"},
    },
    render_mode="human",
)

obs, info = env.reset()
done = False
env.render()

laptime = 0.0
start = time.time()

while not done:
    action = env.action_space.sample()
    obs, step_reward, done, truncated, info = env.step(action)
    print(obs["scans"].shape)
    exit()
    laptime += step_reward
    frame = env.render()
