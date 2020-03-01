import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import xavier_normal as xavier
import  matplotlib.pyplot as plt

from data.loader import cryptoData
from models.model import  MLPRegressor

DEVICE = torch.device("cuda:0")
MODE = "train"
# MODE = "test"

if __name__ == "__main__":

    if MODE == "train":
        test = False
        breaker = 690
    else:
        test = True
        breaker = 90

    model = MLPRegressor()
    model.load_state_dict(torch.load("weights/mlpreg_final.pth"))
    
    optimizer = Adam(model.parameters(), lr=0.000005, weight_decay=0.00001)
    lossfn = nn.MSELoss(reduction='mean')

    dataloader = cryptoData("btc",test=test,DEVICE=DEVICE)

    model.to(DEVICE)


    
    for _ in range(300):
        tots = 0
        for i,(x,target) in enumerate(dataloader):
            if i == breaker:
                break

            x.unsqueeze_(0).unsqueeze_(0)

            optimizer.zero_grad()

            out = model(x)
            out = out.squeeze()
            
            loss = lossfn(out,target)
            tots+=loss.item()
            loss.backward()

            optimizer.step()
        print(_,"\t",tots/690,"\t",out.item(),"\t",target.item())
        torch.save(model.state_dict(),"weights/mlpreg_final.pth")