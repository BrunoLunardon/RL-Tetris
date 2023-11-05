import torch 
import numpy as np
import multiprocessing as mp
from tetris import Tetris
from skopt import gp_minimize
from skopt.space import Real
import random

#choose the number of games you want to be played during the evaluation of each set of weights
N = 75

# Parameters for the Bayesian optimization

# Number of calls to the objective function
n_calls = 50
# Number of random points to sample before fitting the gaussian process
# Balance the trade-off between exploration and exploitation
n_random_starts = 35


verbose = True

random_state = 0

opt = {"width": 10, "height": 20, "block_size": 30, "fps": 300, "saved_path": "trained_models", "output": "output.mp4"}

def heuristic_game(w1,w2,w3,w4,opt,output_name=None):
    """Play a game with the given weights for the heuristic reward (HR) and return 
    the number of cleared lines

    Parameters
    ----------
    w1 : float
        Weight for the number of cleared lines in the HR range [0,1]
    w2 : float
        Weight for the number of holes in the HR range [0,1]
    w3 : float
        Weight for the bumpiness in the HR range [0,1]
    w4 : float
        Weight for the height in the HR range [0,1]
    opt : dict
        Dictionary containing the specs of the game
    output_name : str, optional
        Name of the output video file, by default None
    
    Returns
    -------
    int
        Number of cleared lines
    """


    # Create the game environment
    env = Tetris(width=opt["width"], height=opt["height"], block_size=opt["block_size"])
    env.reset()
    
    # Create the video writer if output_name is not None
    if output_name is not None:
        out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*"MJPG"), opt["fps"],
                          (int(1.5*opt["width"]*opt["block_size"]), opt["height"]*opt["block_size"]))       
        render = True
    else:
        out = None
        render = False


    while True:
        # Get the next states and actions
        next_steps = env.get_next_states()
        next_actions, _ = zip(*next_steps.items())
        
        # Stack all tensors to form a 2D tensor
        stacked_tensors = torch.stack(_)
        
        # Calculate the heuristic reward (HR) for every possible action
        # HR = w1*lines_cleared-w2*holes-w3*bumpiness-w4*height
        HR = w1*stacked_tensors[:, 0]-w2*stacked_tensors[:, 1]-w3*stacked_tensors[:, 2]-w4*stacked_tensors[:, 3]
        
        # Find the maximum of the HRs
        max_score = torch.max(HR)

        # Find the index of the tensor(s) with the maximum HR
        indices = (HR == max_score).nonzero(as_tuple=True)[0]
    
        # If there are multiple tensors with the same maximum HR, choose one randomly
        random_index = random.randint(0, len(indices)-1)

        # Perform the next action
        _, done = env.step(next_actions[indices[random_index]], render=render,video=out)

        # If the game is done, break the loop
        if done:
            if output_name is not None:
                out.release()
                # Very important to release the video writer in jupyter notebook
                # Otherwise the kernel will die
                cv2.destroyAllWindows()
            break

    # Return the number of cleared lines in that game
    return env.cleared_lines

def parallel_games(weights, opt, epochs):
    # Each process should get its copy of weights and opt, no need for global variables here.
    with mp.Pool(mp.cpu_count()) as pool:
        # Each worker will get a separate copy of the weights and the options.
        tasks = [(w1, w2, w3, w4, opt) for w1, w2, w3, w4 in weights]
        # Run the game in parallel and collect the results.
        results = pool.starmap(heuristic_game, tasks)
    return np.mean(results)

def objective(weights):
    opt = {"width": 10, "height": 20, "block_size": 30, "fps": 60, "saved_path": "trained_models", "output": "output.mp4"}
    # Run auxiliary function to play games in parallel and calculate the mean result.
    return -parallel_games([weights], opt, N)


""" Parallelizing the Bayesian optimization process itself is not feasible due to its
 inherently sequential nature. Each step builds upon the outcome of the preceding one, 
 making it a step-by-step procedure. However, an opportunity for parallelization does 
 exist within the evaluation of the objective function. This function involves playing 
 'N' individual games of Tetris, each utilizing a set of given weights, and then calculating 
 the average number of lines cleared. By executing these 'N' games concurrently, we can
 expedite the assessment of each set of weights. Hence, while the optimization sequence 
 remains linear, the efficiency gain lies in the simultaneous evaluation of the objective
 function across multiple games.
"""
if __name__ == "__main__":
    # I did't built a way so the games can be visualized while running in parrallel
    # but if you want to know if its working just open the task manager and see the
    # cpu usage while running the code it should be close to 100%.

    space = [Real(0, 1, name="w1"),
             Real(0, 1, name="w2"),
             Real(0, 1, name="w3"),
             Real(0, 1, name="w4")]


    results = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state, n_random_starts=n_random_starts, verbose=verbose)
    with open("..\\data\\results.txt", "w") as f:
        for i in range(len(results.x_iters)):
            f.write(f"{results.x_iters[i]}: {-results.func_vals[i]}\n")


        

