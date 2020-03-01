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

    model = MLPRegressor()

    if MODE == "train":
        test = False
        breaker = 690
    else:
        test = True
        breaker = 90
    
    dataloader = cryptoData("btc",test=test,DEVICE=DEVICE)
    model.load_state_dict(torch.load("weights/mlpreg_final.pth"))
    model.to(DEVICE)

    t,h = [],[]
    for i,data in enumerate(dataloader):
        if i == breaker:
            break
        x,target = data
        x.unsqueeze_(0).unsqueeze_(0)

        out = model(x)
        out = out.squeeze()
        print(out.item(),'\t\t',target.item())

        t.append(target.item())
        h.append(out.item())

    plt.plot(t)
    plt.plot(h)
    plt.show()
    

        
