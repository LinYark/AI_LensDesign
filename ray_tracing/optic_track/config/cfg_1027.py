import torch

config = {}
config["surfaces"] = [
    (50, 50, [True, True, False], 1.5168, 40),
    (-100, 60, [True, True, False], 2.02204, 40),
    (torch.inf, 100, [False, False, False], 1, 40),
    (100, 50, [False, False, False], 1.5168, 40),
    (-100, 60, [False, False, False], 1, 40),
    (torch.inf, torch.inf, [False, False, False], 1, 40),
]
