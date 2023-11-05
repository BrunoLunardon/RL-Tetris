import argparse
import os
import shutil
from random import random, randint, sample
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from dqn import DQN
from tetris import Tetris
from collections import deque
import matplotlib.pyplot as plt

WIDTH = 10
HEIGHT = 20
BLOCK_SIZE = 30 
BATCH_SIZE = 512
LR = 1e-3
GAMMA = 0.99
INITIAL_EPSILON = 1
FINAL_EPSILON = 1e-3
NUM_DECAY_EPOCHS = 2000
NUM_EPOCHS = 3000
SAVE_INTERVAL = 1000
REPLAY_MEMORY_SIZE = 30000
LOG_PATH = "tensorboard"
SAVED_PATH = "trained_models"

def train():

    #Creating the rewards list so we can plot the results later
    rewards = []

    #Torch seed for reproducibility and configuring the environment
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir("LOG_PATH"):
        shutil.rmtree("LOG_PATH")
    os.makedirs("LOG_PATH")
    writer = SummaryWriter("LOG_PATH")
    env = Tetris(width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE)
    model = DQN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    #Training loop
    replay_memory = deque(maxlen=30000)
    epoch = 0
    while epoch < NUM_EPOCHS:
        next_steps = env.get_next_states()

        #Exploration vs exploitation
        epsilon = FINAL_EPSILON + (max(NUM_EPOCHS/2 - epoch, 0) * (INITIAL_EPSILON - FINAL_EPSILON) / NUM_DECAY_EPOCHS) # Linear decay proportinal to the number of epochs over number of decay epochs
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)

        if torch.cuda.is_available():
            next_states = next_states.cuda()
        model.eval()

        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()

        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :] 
        action = next_actions[index]

        reward, done = env.step(action, render=False)

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, done])

        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue
        if len(replay_memory) < REPLAY_MEMORY_SIZE / 10:
            continue

        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), BATCH_SIZE))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(tuple(reward if done else reward + GAMMA * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None] #Reward if done, else reward + gamma * prediction

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}/{NUM_EPOCHS}, Action: {action}, Score: {final_score}, Tetrominoes {final_tetrominoes}, Cleared lines: {final_cleared_lines}")
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % SAVE_INTERVAL == 0:
            torch.save(model, f"{SAVED_PATH}/tetris_{epoch}_{LR}_{GAMMA}")

        rewards.append((epoch, final_score, final_tetrominoes, final_cleared_lines))

    torch.save(model, f"{SAVED_PATH}/tetris_{LR}_{GAMMA}_{NUM_EPOCHS}")

    return rewards

if __name__ == "__main__":
    r = train()

    epochs, scores, tetrominoes, cleared_lines = zip(*r)

    plt.plot(epochs, scores)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.savefig(f"graphs/score_{LR}_{GAMMA}_{NUM_EPOCHS}.png")

    plt.clf()
    plt.plot(epochs, tetrominoes)
    plt.xlabel('Epoch')
    plt.ylabel('Tetrominoes')
    plt.savefig(f'graphs/tetrominoes_{LR}_{GAMMA}_{NUM_EPOCHS}.png')

    plt.clf()
    plt.plot(epochs, cleared_lines)
    plt.xlabel('Epoch')
    plt.ylabel('Cleared lines')
    plt.savefig(f'graphs/cleared_lines_{LR}_{GAMMA}_{NUM_EPOCHS}.png')  