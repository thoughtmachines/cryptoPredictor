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
MODE = "test"
# MODE = "test"

if __name__ == "__main__":

    if MODE == "train":
        test = False
        
    else:
        test = True

    model = MLPRegressor()
    model.load_state_dict(torch.load("weights/btc_mlp.pth"))
    

    dataloader = cryptoData("btc",test=test,DEVICE=DEVICE)

    model.to(DEVICE)

    breaker = len(dataloader)

    # model.eval(dataloader[0][0])

    t,h = [],[]
    z = 0 
    for i,data in enumerate(dataloader):
        if i == breaker:
            break
        x,target = data


        out = model(x)

        t.append(target.item())
        h.append(out.item())
        z+= abs((t[-1] - h[-1])/t[-1])

    plt.plot(t)
    plt.plot(h)
    plt.show()

    print((z/breaker))
    

        
