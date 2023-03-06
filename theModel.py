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
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(in_features = config.nInputFeatures, out_features=config.nInputFeatures, device = pDevice))
        self.relu = nn.ReLU()
        for i in range(N_INNER_LAYERS):
            self.layers.append(nn.Linear(in_features = config.nInputFeatures * 2, out_features=config.nInputFeatures, device = pDevice))
        self.outputLayer = nn.Linear(in_features = config.nInputFeatures * 2, out_features = config.nOutputFeatures, device = pDevice)
        self.pDevice = pDevice
        self.to(self.pDevice)

    def loadFromDir(self, sDirName):
            print(f"loading the model from directory [{sDirName}]");
            prevModelDict = self.state_dict() # save previous model just in case
            try:
                newModelDict = torch.load(sDirName + '\\bestModel.pth')
                self.load_state_dict(newModelDict)
                print("    *** succeeded")
            except:
                print("    *** dailed")
                self.load_state_dict(prevModelDict) # restore previous model
            self.to(self.pDevice)

    def saveToDir(self, sDirName):
            modelDict = self.state_dict()
            torch.save(modelDict, sDirName + '\\bestModel.pth')

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x1 = self.relu( x)
            x2 = self.relu(-x)
            x = torch.cat((x1, x2), dim=1)
        x = self.outputLayer(x)
        return x