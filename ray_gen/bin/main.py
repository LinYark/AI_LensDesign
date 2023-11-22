import os 
import sys
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
from train import train

def run():
    train()

if __name__ == "__main__":
    run()