import torch
import cv2
from tetris import Tetris

WIDTH = 10
HEIGHT = 20
BLOCK_SIZE = 30
FPS = 300
SAVED_PATH = "trained_models/3000epochs0001"
OUTPUT = "output.mp4"

def test():
    # Set seed for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # Load model and environment
    if torch.cuda.is_available():
        model = torch.load(f"{SAVED_PATH}/tetris_0.001_0.99_3000")
    else:
        model = torch.load(f"{SAVED_PATH}/tetris_0.001_0.99_3000", map_location=lambda storage, loc: storage)
    model.eval()
    env = Tetris(width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE)
    env.reset()

    if torch.cuda.is_available():
        model.cuda()

    out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (int(1.5*WIDTH*BLOCK_SIZE), HEIGHT*BLOCK_SIZE))
    
    # Test model
    i = 0
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=out)

        if done:
            print(f"Score: {env.score} | Cleared lines: {env.cleared_lines} | Tetrominoes: {env.tetrominoes}")
            env.reset()
            i += 1

        if i == 1:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()