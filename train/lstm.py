import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import xavier_normal as xavier
import  matplotlib.pyplot as plt

from data.loader import cryptoData
from models.model import  SeqRegressor

DEVICE = torch.device("cpu")
MODE = "train"

if __name__ == "__main__":

    model = SeqRegressor()

    optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=0.00001)
    lossfn = nn.MSELoss(reduction='mean')

    if MODE == "train":
        test = False
        breaker = 690
    else:
        test = True
        breaker = 90
    
    dataloader = cryptoData("btc",test=test,DEVICE=DEVICE)
    model.to(DEVICE)

    for _ in range(3000):
        tots = 0

        for i,(x,target) in enumerate(dataloader):
            if i == breaker:
                break
            x.unsqueeze_(1)

            hidden_state = torch.ones(1,1, 20).to(DEVICE)
            cell_state = torch.ones(1,1, 20).to(DEVICE)
            model.lstm.hidden = (hidden_state,cell_state)
            

            optimizer.zero_grad()

            out = model(x)
            out = out.squeeze()
            
            loss = lossfn(out,target)
            tots+=loss.item()
            loss.backward()

            optimizer.step()
        torch.save(model.state_dict(),"lstm_final.pth")
        print(_,"\t",tots/690,"\t",out)