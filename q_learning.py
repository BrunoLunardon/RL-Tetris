import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from tetris import Tetris

style.use('ggplot')

BOARD_WIDTH = 10 
BOARD_HEIGHT = 20

# Tetris environment
env = Tetris(width=BOARD_WIDTH, height=BOARD_HEIGHT, block_size=30)
env.reset()

# Environment video recording
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 300,
                          (int(1.5*10*30), 20*30))

# Some important q-learning constants
LEARNING_RATE = 0.3
DISCOUNT = 0.9
MAX_EPISODES = 100000
RENDER_INTERVAL = 1000

# Stats definitions
STATS_EVERY = 1000

total_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

# epsilon is the exploration parameter; the bigger his value, the bigger the chance for the environment to make an exploratory action
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = MAX_EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Shape of the Observation Space
OS_SHAPE = (5, (BOARD_WIDTH-1)*(BOARD_HEIGHT-1), (BOARD_WIDTH//2)*BOARD_HEIGHT, (BOARD_WIDTH)*(BOARD_HEIGHT))

q_table = np.random.uniform(low = -2, high = 0, size = OS_SHAPE) # Populating the initial q-table with random values
# TODO: this is a pretty inneficient table, since it have more than 100 million entries. Maybe there is some way to optimize it? 

# Main training loop
for episode in range(MAX_EPISODES+1):
    episode_reward = 0

    if not episode%100: print(episode)

    if episode % RENDER_INTERVAL == 0: # Intermediate episodes where the environment will show current results
        show_render = True
    
    else: show_render = False

    # First we take the possible next states
    next_steps = env.get_next_states()
    next_actions, next_states = zip(*next_steps.items())
    next_states = torch.stack(next_states)

    if torch.cuda.is_available(): # CUDA optimization, if available 
        next_states = next_states.cuda()

    # Adding the piece id to the state info
    # piece_id = torch.tensor([[env.ind]]*len(next_actions))
    # next_states = torch.cat((next_states, piece_id), 1)

    # Search for the action that returns the best q value
    q_values = [q_table[state[0]][state[1]][state[2]][state[3]] for state in next_states.int()]
    
    done = False
    while not done:
        # Verifies the probability of making an exploratory step
        if np.random.random() > epsilon:
            index = np.argmax(q_values)   
        else:
            index = np.random.randint(0, len(q_values))
        
        action = next_actions[index]    # is the same from the action that leads to that observation 
        # Applies the best action
        score, done = env.step(action, render = show_render, video = out)
        reward = score - 2

        episode_reward += reward

        if env.score > 10000: # If we reach this score, we're satisfied with this episode
            done = True
            print(f"Achieved victory in episode {episode}!!!")

        if not done: # If the training isn't finished, we update our q_values
            # Start by saving the current state's location
            current_q_loc = next_states[index].int()

            # Search for the best action in the next step
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)

            if torch.cuda.is_available(): # CUDA optimization, if available 
                next_states = next_states.cuda()

            # Adding the piece id to the new state info
            # piece_id = torch.tensor([[env.ind]]*len(next_actions))
            # next_states = torch.cat((next_states, piece_id), 1)

            # Search for the best q value obtainable in the next step
            q_values = [q_table[state[0]][state[1]][state[2]][state[3]] for state in next_states.int()]
            max_future_q = np.max(q_values)

            # Finally calculate the new value for the current state
            new_q = (1 - LEARNING_RATE) * q_table[current_q_loc[0]][current_q_loc[1]][current_q_loc[2]][current_q_loc[3]] + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # TODO: Problem, we don't have a well-defined reward :/

            q_table[current_q_loc[0]][current_q_loc[1]][current_q_loc[2]][current_q_loc[3]] = new_q # Insert the updated value into the table
    
    # update the epsilon parameter
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    total_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(total_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(total_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(total_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
        
        np.save(f"qtables/{episode}-qtable.npy", q_table)

    env.reset() # Resets environment after each episode

cv2.destroyAllWindows()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=2)
plt.show()