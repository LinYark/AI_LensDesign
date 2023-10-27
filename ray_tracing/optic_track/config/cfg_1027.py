import torch
config = {}
config["surfaces"] = [
    (50,50,[True,False],1.5168,40),
    (-100,60,[True,False],2.02204,40),
    (torch.inf,100,[True,True],1,40),
    (100,50,[False,False],1.5168,50),
    (-100,60,[False,False],1,50),
    (torch.inf,torch.inf,[False,False],1,30),
]