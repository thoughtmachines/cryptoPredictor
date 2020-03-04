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

DEVICE = torch.device("cpu")
MODE = "train"
# MODE = "test"

if __name__ == "__main__":

    model = MLPRegressor(stochastic=True)

    if MODE == "train":
        test = False
        breaker = 643
    else:
        test = True
        breaker = 223
    
    dataloader = cryptoData("btc",test=test,DEVICE=DEVICE)
    model.load_state_dict(torch.load("weights/1preg_final_norm.pth"))
    model.to(DEVICE)

    model.eval(dataloader[0][0])

    t,h = [8000],[8000]
    z = 0 
    for i,data in enumerate(dataloader):
        if i == breaker:
            break
        x,target = data


        out = model(x)

        # d  = (out - h[-1]) * torch.rand((1,1)) 
        # out+= d

        t.append(target.item())
        h.append(out.item())
        z+= abs((t[-1] - h[-1])/t[-1])

    plt.plot(t)
    plt.plot(h)
    plt.show()

    print((z/breaker))
    

        
