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
MODE = "test"
# MODE = "test"

if __name__ == "__main__":

    model = SeqRegressor(stochastic=True)

    if MODE == "train":
        test = False

    else:
        test = True

    
    dataloader = cryptoData("ltc",test=test,DEVICE=DEVICE)
    model.load_state_dict(torch.load("weights/mape_norm_ltc_lstm.pth"))
    model.to(DEVICE)

    
    breaker = len(dataloader)

    for j in range(100):
        model.eval(dataloader[0][0].unsqueeze(1))
        t,h = [],[]
        z = 0 
        m = 0
        r=0
        for i,(x,target) in enumerate(dataloader):
            if i == breaker:
                break

            x.unsqueeze_(1)


            out = model(x)
            out = out.squeeze()

            t.append(target.item()*dataloader.pmax.item())
            h.append(out.item()*dataloader.pmax.item())
            z+= abs((t[-1] - h[-1])/t[-1])
            
        print(z/breaker * 100)
    
