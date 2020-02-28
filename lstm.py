import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import xavier_normal as xavier
import  matplotlib.pyplot as plt

from data.loader import cryptoData
from models.model import  MLPRegressor, SeqRegressor

DEVICE = torch.device("cpu")


if __name__ == "__main__":


    model = SeqRegressor()
    

    dataloader = cryptoData("btc",test=True)
    model.load_state_dict(torch.load("new_lstm.pth"))
    model.to(DEVICE)
    t = []
    h = []
    for i,data in enumerate(dataloader):
        if i == 90:
            break
        x,target = data
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
    

        
