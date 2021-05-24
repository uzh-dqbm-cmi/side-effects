import torch
from torch import nn
import torch.nn.functional as F

class NDD_Code(nn.Module):
    def __init__(self, D_in=1096, H1=400, H2=300, D_out=1, drop=0.5):
        super(NDD_Code, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(D_in, H1) # Fully Connected
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, D_out)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.uniform_(-1,0)