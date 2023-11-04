from tetris import Tetris
import numpy as np
import pandas as pd
import torch
import random
import cv2

def heuristic_game(w1,w2,w3,w4,opt,output_name=None):
    # Create the environment
    env = Tetris(width=opt["width"], height=opt["height"], block_size=opt["block_size"])
    env.reset()
    
    # Create the video writer if output_name is not None
    if output_name is not None:
        out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*"MJPG"), opt["fps"],
                          (int(1.5*opt["width"]*opt["block_size"]), opt["height"]*opt["block_size"]))       
        render = True
    else:
        render = False

    while True:
        # Get the next states and actions
        next_steps = env.get_next_states()
        next_actions, _ = zip(*next_steps.items())
        
        # Stack all tensors to form a 2D tensor
        stacked_tensors = torch.stack(_)
        
        heuristic_scores = w1*stacked_tensors[:, 0]-w2*stacked_tensors[:, 1]-w3*stacked_tensors[:, 2]-w4*stacked_tensors[:, 3]
        # heuristic_scores = w1*linhas_limpa-w2*buracos-w3*rugosidade-w4*altura
        
        # Find the maximum of the scores
        max_score = torch.max(heuristic_scores)

        # Find the index of the tensor(s) with the maximum score
        indices = (heuristic_scores == max_score).nonzero(as_tuple=True)[0]
    
        # If there are multiple tensors with the same maximum score, choose one randomly
        random_index = random.randint(0, len(indices)-1)

        # Perform the next action
        _, done = env.step(next_actions[indices[random_index]], render=render,video=out)

        # If the game is done, break the loop
        if done:
            if output_name is not None:
                out.release()
                cv2.destroyAllWindows()
            break

    return env.cleared_lines    

df=pd.read_csv("25pt_50g.csv")
arr=df.to_numpy()
best_ws=arr[-1,0:4]
best_score=arr[-1,4]
heuristic_game(best_ws[0],best_ws[1],best_ws[2],best_ws[3],opt={"width": 10, "height": 20, "block_size": 30, "fps": 144},output_name="best_game.avi")