import torch
import cv2
from tetris import Tetris

WIDTH = 10
HEIGHT = 20
BLOCK_SIZE = 30
FPS = 300
SAVED_PATH = "trained_models/3000epochs01_Exploration"
OUTPUT = "output.mp4"

def test():
    # Set seed for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # Load model and environment
    if torch.cuda.is_available():
        model = torch.load(f"{SAVED_PATH}/tetris_0.1_0.99_3000")
    else:
        model = torch.load(f"{SAVED_PATH}/tetris_0.1_0.99_3000", map_location=lambda storage, loc: storage)
    model.eval()
    env = Tetris(width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE)
    output = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 60,
                                   (int(1.5*10*30), 20*30))
    env.reset()

    if torch.cuda.is_available():
        model.cuda()
    
    # Test model
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=output)

        if done:
            print(f"Score: {env.score} | Cleared lines: {env.cleared_lines} | Tetrominoes: {env.tetrominoes}")
            break

if __name__ == "__main__":
    test()