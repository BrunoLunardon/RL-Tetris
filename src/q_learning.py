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

    def load_q_table(self, filename):
        self.q_table = np.load(f"./qtables/{filename}")

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

    def train(self, learn_rate, discount, max_episodes, target = np.inf, save_q_table = "final", plot = True):
        self.clean_rewards()

        # epsilon is the exploration parameter; the bigger his value, the bigger the chance for the environment to make an exploratory action
        epsilon = 1
        start_epsilon_decay = 1
        end_epsilon_decay = max_episodes
        epsilon_decay_value = epsilon/(end_epsilon_decay - start_epsilon_decay)
        
        # Main training loop
        for episode in range(max_episodes+1): # Outer loop, referes to training epochs
            episode_reward = 0

            # Intermediate episodes where the environment will show current results
            if episode % self.render_interval == 0: show_render = True
            else: show_render = False

            # First we take the possible states
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

                # save the current state's location
                current_q_loc = next_states[index].int()

                if done: # the model lost, and the q_value must be discouraged
                    self.q_table[current_q_loc[0]][current_q_loc[1]][current_q_loc[2]][current_q_loc[3]] = -2 

                elif self.env.score > target: # If we reach this score, we're satisfied with this episode
                    done = True
                    print(f"Achieved victory in episode {episode}!!!")

                else: # If the episode isn't finished yet, update the q_values
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

                    self.q_table[current_q_loc[0]][current_q_loc[1]][current_q_loc[2]][current_q_loc[3]] = new_q # Insert the updated value into the table
            
            # update the epsilon parameter
            if end_epsilon_decay >= episode >= start_epsilon_decay:
                epsilon -= epsilon_decay_value

            self.total_rewards.append(episode_reward)
            print(f"Episode {episode}/{max_episodes} reward: {episode_reward}")
            
            if not episode % self.stats_interval:
                average_reward = sum(self.total_rewards[-self.stats_interval:])/self.stats_interval
                self.aggr_ep_rewards['ep'].append(episode)
                self.aggr_ep_rewards['avg'].append(average_reward)
                self.aggr_ep_rewards['max'].append(max(self.total_rewards[-self.stats_interval:]))
                self.aggr_ep_rewards['min'].append(min(self.total_rewards[-self.stats_interval:]))
                print(f"Relatory: Max Reward: {self.aggr_ep_rewards['max'][-1]}, Average Reward: {average_reward}, Min Reward: {self.aggr_ep_rewards['min'][-1]}, Current Epsilon: {epsilon}")
                
            if save_q_table == "stats" and not episode % self.stats_interval:
                np.save(f"./qtables/{episode}-qtable.npy", self.q_table)

            elif save_q_table == "final" and episode == max_episodes:
                np.save(f"./qtables/{episode}-qtable.npy", self.q_table)

            self.env.reset() # Resets environment after each episode

        cv2.destroyAllWindows()
        if plot: self.plot_rewards()

    # def parameter_analysis(self):
    #     for discount in range(0.1, 1, 0.1)

if __name__ == "__main__":
    q_model = QL_Tetris()
    q_model.train(learn_rate=0.3,
                discount=0.9,
                max_episodes=20000)
