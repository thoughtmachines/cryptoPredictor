import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import xavier_normal as xavier
import  matplotlib.pyplot as plt

from data.norm_loader import cryptoData
from models.model import  SeqRegressor

DEVICE = torch.device("cpu")
MODE = "train"
# MODE = "test"

if __name__ == "__main__":

    model = SeqRegressor(stochastic=True)

    if MODE == "train":
        test = False

    else:
        test = True

    
    dataloader = cryptoData("eth",test=test,DEVICE=DEVICE)
    model.load_state_dict(torch.load("weights/_norm_eth_lstm.pth"))
    model.to(DEVICE)

    model.eval(dataloader[0][0].unsqueeze(1))
    breaker = len(dataloader)
    t,h = [],[]
    z = 0
    for i,(x,target) in enumerate(dataloader):
        if i == breaker:
            break

        x.unsqueeze_(1)


        out = model(x)
        out = out.squeeze()

        # t.append(target.item())
        # h.append(out.item())
        # z+= abs((t[-1] - h[-1])/t[-1])

    dataloader = cryptoData("eth",test=True,DEVICE=DEVICE)
    breaker = len(dataloader)
    for i,(x,target) in enumerate(dataloader):
        if i == 690:
            break

        x.unsqueeze_(1)

        out = model(x)
        out = out.squeeze()

        t.append(target.item())
        h.append(out.item())
        z+= abs((t[-1] - h[-1])/t[-1])
    print((z/breaker))

    plt.plot(t)
    plt.plot(h)
    plt.show()
    

        
