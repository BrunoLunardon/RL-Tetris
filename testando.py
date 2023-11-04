#import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from IPython.display import clear_output
import cv2
import argparse
import sys
from tetris import Tetris, get_args
import multiprocessing as mp
import pandas as pd

counter=0

def worker(w1,w2,w3,w4,opt, output_name="output.mp4"):
    try:
        #device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        env = Tetris(width=opt["width"], height=opt["height"], block_size=opt["block_size"])
        env.reset()
        
        # out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*"MJPG"), opt["fps"],
        #                    (int(1.5*opt["width"]*opt["block_size"]), opt["height"]*opt["block_size"]))       
        while True:
            next_steps = env.get_next_states()
            next_actions, _ = zip(*next_steps.items())
            tensors=_
            #tensors = [t.to(device) for t in _]

            # Stack all tensors to form a 2D tensor
            stacked_tensors = torch.stack(tensors)
        
            heuristic_scores = w1*stacked_tensors[:, 0]-w2*stacked_tensors[:, 1]-w3*stacked_tensors[:, 2]-w4*stacked_tensors[:, 3]
            #heuristic_scores = w1*linhas_limpa-w2*buracos-w3*rugosidade-w4*altura
            # Extract second entries of all tensors into a tensor
            # second_entries = stacked_tensors[:, 1]
        
            # Find the minimum of the second entries
            max_val = torch.max(heuristic_scores)
        
            # Find the index of the tensor(s) with the minimum second entry
            indices = (heuristic_scores == max_val).nonzero(as_tuple=True)[0]
            #indices=indices.cpu()
        
            random_index = random.randint(0, len(indices)-1)

            # Using advanced indexing to get the tensor(s) with the minimum second entry
            #filtered_tensors = stacked_tensors[indices].unbind(dim=0)


            # Escolhendo uma ação aleatória
            #action = random.choice(next_actions)

            #print(next_actions[indices[0]])

            _, done = env.step(next_actions[indices[random_index]], render=False, video=None )

                
            if done:
                #out.release()
                #cv2.destroyAllWindows()
                print("Game Over!!!")
                break



        return env.cleared_lines
    except Exception as e:
        print(e)
        return 0

def heuristic_search(opt):
    Df_25pt_50g=pd.read_csv('25pt_50g.csv')
    arr_25pt_50g=Df_25pt_50g.to_numpy()
    idxs=arr_25pt_50g[:,4].argsort()
    idxs=np.flip(idxs)
    arr_25pt_50g=arr_25pt_50g[idxs]
    w1=arr_25pt_50g[4,0]
    w2=arr_25pt_50g[4,1]
    w3=arr_25pt_50g[4,2]
    w4=arr_25pt_50g[4,3]

    args=[]
    for i in range(100):
        args.append((w1,w2,w3,w4,opt))
    with mp.Pool() as pool:
        results = pool.starmap(worker, args)

    return results

if __name__=='__main__':
    opt = {"width": 10, "height": 20, "block_size": 30, "fps": 300, "saved_path": "trained_models", "output": "output.mp4"}
    results=heuristic_search(opt)
    df=pd.DataFrame(results)
    df.to_csv('linhas_limpas5.csv')
