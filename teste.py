import gymnasium as gym

env = gym.make("ALE/Tetris-ram-v5", render_mode="human", repeat_action_probability=0.1, full_action_space=True)
observation, info = env.reset()
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()