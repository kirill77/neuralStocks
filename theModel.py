from dataclasses import dataclass
import torch
import torch.nn as nn

# Global constants
N_INNER_LAYERS = 5

@dataclass
class ModelConfig:
    nInputFeatures: int = None
    nOutputFeatures: int = None

def initModelConfig(dataSet):
    config = ModelConfig()
    input, output = dataSet.__getitem__(0)
    config.nInputFeatures = len(input)
    config.nOutputFeatures = len(output)
    return config

class MyModel(nn.Module):
    def __init__(self, config, pDevice=torch.device("cpu")):
        super().__init__()
        self.relu = nn.ReLU()
        self.firstLayer = nn.Linear(in_features = config.nInputFeatures, out_features=config.nInputFeatures, device = pDevice)
        self.innerLayers = []
        for i in range(N_INNER_LAYERS):
            self.innerLayers.append(nn.Linear(in_features = config.nInputFeatures * 2, out_features=config.nInputFeatures, device = pDevice))
        self.lastLayer = nn.Linear(in_features = config.nInputFeatures * 2, out_features = config.nOutputFeatures, device = pDevice)
        self.to(pDevice)

    def forward(self, x):
        y = self.firstLayer(x)
        y1 = self.relu( y)
        y2 = self.relu(-y)
        y = torch.cat((y1, y2), dim=1)
        for innerLayer in self.innerLayers:
            y = innerLayer(y)
            y1 = self.relu( y)
            y2 = self.relu(-y)
            y = torch.cat((y1, y2), dim=1)
        y = self.lastLayer(y)
        return y
