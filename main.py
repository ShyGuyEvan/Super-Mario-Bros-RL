import torch

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from agent import Agent

from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers

import os
import csv
from datetime import datetime

from utils import *

# Define the directory to save log files
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Generate timestamp for the filename
timestamp =  get_current_date_time_string()
filename = f"episode_logs_{timestamp}.csv"
filepath = os.path.join(log_dir, filename)

# Define the directory to save model checkpoints
model_checkpoint_dir = "model_checkpoints"
os.makedirs(model_checkpoint_dir, exist_ok=True)

# Function to save the model checkpoint
def save_model_checkpoint(agent, episode):
    checkpoint_path = os.path.join(model_checkpoint_dir, f"model_episode_{episode}.pt")
    torch.save(agent.state_dict(), checkpoint_path)

# Function to load the model checkpoint
def load_model_checkpoint(agent, episode):
    checkpoint_path = os.path.join(model_checkpoint_dir, f"model_episode_{episode}.pt")
    agent.load_state_dict(torch.load(checkpoint_path))
    
# Define the path to save the episode logs CSV file
episode_logs_path = "episode_logs.csv"

# Define the fieldnames for the CSV file
fieldnames = ["Episode", "Total Reward", "Epsilon", "Replay Buffer Size", "Learn Step Counter"]

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 5000
NUM_OF_EPISODES = 50_000

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

if not SHOULD_TRAIN:
    folder_name = ""
    ckpt_name = ""
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)


for i in range(NUM_OF_EPISODES):
    if i == 0:  # Write the header only once before writing episode logs
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        a = agent.choose_action(state)
        new_state, reward, done, truncated, info  = env.step(a)
        total_reward += reward

        if SHOULD_TRAIN:
            agent.store_in_memory(state, a, reward, new_state, done)
            agent.learn()

        state = new_state

    print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)

    # Open the CSV file in append mode and write the episode logs
    with open(filepath, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            "Episode": i,
            "Total Reward": total_reward,
            "Epsilon": agent.epsilon,
            "Replay Buffer Size": len(agent.replay_buffer),
            "Learn Step Counter": agent.learn_step_counter
        })

    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

    print("Total reward:", total_reward)
env.close()
