import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import xavier_normal as xavier

from data.loader import cryptoData
from models.model import  MLPRegressor, SeqRegressor


DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    print("Using CUDA backend")

if __name__ == "__main__":

    """

    model = MLPRegressor().to(DEVICE)
    
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    lossfn = nn.MSELoss(reduction='mean')

    dataloader = cryptoData("btc")

    model.to(DEVICE)
    for _ in range(300):
        tots = 0
        for i,data in enumerate(dataloader):
            if i == 690:
                break
            x,target = data
            x = x.to(DEVICE)
            target = target.to(DEVICE)
            
            x.unsqueeze_(0).unsqueeze_(0)

            optimizer.zero_grad()

            out = model(x)
            out = out.squeeze()
            
            loss = lossfn(out,target)
            tots+=loss.item()
            loss.backward()

            optimizer.step()
        torch.save(model.state_dict(),"x.pth")
        print(_,"\t",tots/690,"\t",out.item(),"\t",target.item())
    
    """

    model = SeqRegressor()
    model.load_state_dict(torch.load("x.pth"))
    optimizer = Adam(model.parameters(), lr=0.000001, weight_decay=0.00001)
    lossfn = nn.MSELoss(reduction='mean')

    dataloader = cryptoData("btc")

    model.to(DEVICE)

    for _ in range(3000):
        tots = 0

        

        for i,data in enumerate(dataloader):
            if i == 690:
                break
            x,target = data
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
        torch.save(model.state_dict(),"x.pth")
        print(_,"\t",tots/690,"\t",out)