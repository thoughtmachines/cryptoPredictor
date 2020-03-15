import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import xavier_normal as xavier
import  matplotlib.pyplot as plt

from data.unorm_loader import cryptoData
from models.model import  MLPRegressor

DEVICE = torch.device("cpu")
MODE = "test"
# MODE = "test"

if __name__ == "__main__":

    if MODE == "train":
        test = False
        
    else:
        test = True

    model = MLPRegressor(stochastic=True)
    model.load_state_dict(torch.load("weights/_unorm_ltc_mlp.pth"))
    

    dataloader = cryptoData("ltc",test=test,DEVICE=DEVICE)

    model.to(DEVICE)

    breaker = len(dataloader)

    

    for j in range(100):
        model.eval(dataloader[0][0])
        t,h = [],[]
        z = 0 
        m = 0
        r=0
        for i,data in enumerate(dataloader):
            if i == breaker:
                break
            x,target = data


            out = model(x)

            t.append(target.item()*dataloader.pmax.item())
            h.append(out.item()*dataloader.pmax.item())
            z+= abs((t[-1] - h[-1])/t[-1])
            m+= abs((t[-1] - h[-1]))**2
            r+= abs((t[-1] - h[-1])/t[-1])**2

        # plt.plot(t)
        # plt.plot(h)
        # plt.show()

        # print("MAPE",z/breaker * 100)
        # print("MAE",z/breaker)
        # print("RMSE",(r/breaker)**0.5)
        # print("MSE",m/breaker)
        
        print(z/breaker * 100)

        
