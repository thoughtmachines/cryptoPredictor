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
MODE = "test"
# MODE = "test"

if __name__ == "__main__":

    model = SeqRegressor()

    if MODE == "train":
        test = False
        breaker = 690
    else:
        test = True
        breaker = 90
    
    dataloader = cryptoData("btc",test=test,DEVICE=DEVICE)
    model.load_state_dict(torch.load("weights/lstm_final.pth"))
    model.to(DEVICE)

    t,h = [],[]
    for i,(x,target) in enumerate(dataloader):
        if i == breaker:
            break

        x.unsqueeze_(1)
        hidden_state = torch.ones(1,1, 20).to(DEVICE)
        cell_state = torch.ones(1,1, 20).to(DEVICE)
        model.lstm.hidden = (hidden_state,cell_state)

        out = model(x)
        out = out.squeeze()

        t.append(target.item())
        h.append(out.item())

    plt.plot(t)
    plt.plot(h)
    plt.show()
    

        
