import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from tetris import Tetris

style.use('bmh')

BOARD_WIDTH = 10 
BOARD_HEIGHT = 20

class QL_Tetris:
    def __init__(self, render_interval = 1000, stats = 1000):
        # Tetris environment
        self.env = Tetris(width=BOARD_WIDTH, height=BOARD_HEIGHT, block_size=30)
        self.env.reset()

        # Environment video recording
        self.render_interval = render_interval
        self.out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 300,
                                (int(1.5*10*30), 20*30))

        # Stats definitions
        self.stats_interval = stats

        # Shape of the Observation Space
        OS_SHAPE = (5, (BOARD_WIDTH-1)*(BOARD_HEIGHT-1), (BOARD_WIDTH//2)*BOARD_HEIGHT, (BOARD_WIDTH)*(BOARD_HEIGHT))
        self.reset_q_table(OS_SHAPE)

        # TODO: this is a pretty inneficient table, since it have more than 100 million entries. Maybe there is some way to optimize it? 

    def reset_q_table(self, shape):
        self.q_table = np.random.uniform(low = -2, high = 0, size = shape) # Populating the initial q-table with random values

    def plot_rewards(self):
        plt.plot(self.aggr_ep_rewards['ep'], self.aggr_ep_rewards['avg'], label="average rewards")
        plt.plot(self.aggr_ep_rewards['ep'], self.aggr_ep_rewards['max'], label="max rewards")
        plt.plot(self.aggr_ep_rewards['ep'], self.aggr_ep_rewards['min'], label="min rewards")
        plt.legend(loc=2)
        plt.grid(True)
        plt.show()

    def clean_rewards(self):
        self.total_rewards = []
        self.aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

    def train(self, learn_rate, discount, max_episodes, epsilon, eps_interval = "half", target = np.inf, save_q_table = "final"):
        self.clean_rewards()
        
        # epsilon is the exploration parameter; the bigger his value, the bigger the chance for the environment to make an exploratory action
        start_epsilon_decay = 1

        if eps_interval == "half":
            end_epsilon_decay = max_episodes/2
            epsilon_decay_value = epsilon/(end_epsilon_decay - start_epsilon_decay)
        
        elif eps_interval == "full":
            end_epsilon_decay = max_episodes
            epsilon_decay_value = epsilon/(end_epsilon_decay - start_epsilon_decay)

        # Main training loop
        for episode in range(max_episodes+1): # Outer loop, referes to training epochs
            episode_reward = 0

            if not episode%100: print(episode)

            # Intermediate episodes where the environment will show current results
            if episode % self.render_interval == 0: show_render = True
            else: show_render = False

            # First we take the possible next states
            next_steps = self.env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)

            if torch.cuda.is_available(): # CUDA optimization, if available 
                next_states = next_states.cuda()

            # Search for the action that returns the best q value
            q_values = [self.q_table[state[0]][state[1]][state[2]][state[3]] for state in next_states.int()]
            
            done = False
            while not done: # Inner loop, refering to a single game

                if np.random.random() > epsilon: index = np.argmax(q_values) # Policy Step
                else: index = np.random.randint(0, len(q_values)) # Exploratory Step
                
                action = next_actions[index] 

                # Applies the best action
                score, done = self.env.step(action, render = show_render, video = self.out)
                reward = score - 2

                episode_reward += reward

                if self.env.score > target: # If we reach this score, we're satisfied with this episode
                    done = True
                    print(f"Achieved victory in episode {episode}!!!")

                if not done: # If the episode isn't finished, update the q_values
                    # Start by saving the current state's location
                    current_q_loc = next_states[index].int()

                    # Search for the best action in the next step
                    next_steps = self.env.get_next_states()
                    next_actions, next_states = zip(*next_steps.items())
                    next_states = torch.stack(next_states)

                    if torch.cuda.is_available(): # CUDA optimization, if available 
                        next_states = next_states.cuda()

                    # Search for the best q value obtainable in the next step
                    q_values = [self.q_table[state[0]][state[1]][state[2]][state[3]] for state in next_states.int()]
                    max_future_q = np.max(q_values)

                    # Finally calculate the new value for the current state
                    new_q = (1 - learn_rate) * self.q_table[current_q_loc[0]][current_q_loc[1]][current_q_loc[2]][current_q_loc[3]] + learn_rate * (reward + discount * max_future_q)
                    # TODO: Problem, we don't have a well-defined reward :/

                    self.q_table[current_q_loc[0]][current_q_loc[1]][current_q_loc[2]][current_q_loc[3]] = new_q # Insert the updated value into the table
            
            # update the epsilon parameter
            if end_epsilon_decay >= episode >= start_epsilon_decay:
                epsilon -= epsilon_decay_value

            self.total_rewards.append(episode_reward)
            if not episode % self.stats_interval:
                average_reward = sum(self.total_rewards[-self.stats_interval:])/self.stats_interval
                self.aggr_ep_rewards['ep'].append(episode)
                self.aggr_ep_rewards['avg'].append(average_reward)
                self.aggr_ep_rewards['max'].append(max(self.total_rewards[-self.stats_interval:]))
                self.aggr_ep_rewards['min'].append(min(self.total_rewards[-self.stats_interval:]))
                print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
                
            if save_q_table == "stats" and not episode % self.stats_interval:
                np.save(f"qtables/{episode}-qtable.npy", self.q_table)

            elif save_q_table == "final" and episode == max_episodes:
                np.save(f"qtables/{episode}-qtable.npy", self.q_table)

            self.env.reset() # Resets environment after each episode

        self.plot_rewards()
        cv2.destroyAllWindows()


q_model = QL_Tetris()
q_model.train(learn_rate=0.3,
              discount=0.9,
              max_episodes=10000,
              epsilon=0.5,
              eps_interval="full")
