import os 
import sys
sys.path.append(os.getcwd())
from train import train


def run():
    train()

if __name__ == "__main__":
    train()