import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import xavier_normal as xavier
import  matplotlib.pyplot as plt

from data.loader import cryptoData
from models.model import  MLPRegressor, SeqRegressor

DEVICE = torch.device("cpu")


if __name__ == "__main__":


    model = MLPRegressor()
    

    dataloader = cryptoData("btc",True)
    model.load_state_dict(torch.load("xreg.pth"))
    model.to(DEVICE)
    t = []
    h = []
    for i,data in enumerate(dataloader):
        if i == 90:
            break
        x,target = data
        x.unsqueeze_(0).unsqueeze_(0)


        out = model(x)
        out = out.squeeze()

        t.append(target.item())
        h.append(out.item())

    plt.plot(t)
    plt.plot(h)
    plt.show()
    

        
