import torch
from barsDataSet import createBarDataSets
from theModel import initModelConfig, MyModel
import argparse
import myConstants

# parse command line args
parser = argparse.ArgumentParser(description="virtual stock trader")

parser.add_argument('--sDataDir', '--sDataDir', type=str, default='history', help="folder with input stock data that we will use for virtual trading")
parser.add_argument('--sModelDir', '--sModelDir', type=str, default='model1', help="folder with neural model weights")
args = parser.parse_args()
print(vars(args))

# 0 train fraction - means all bars will go into the testing set
trainingSet, testingSet = createBarDataSets(fTrainFraction=0, sDataDir=args.sDataDir)

modelConfig = initModelConfig(testingSet)
model = MyModel(modelConfig)
model.load_state_dict(torch.load(args.sModelDir + '\\bestModel.pth'))

nTickers = len(testingSet.allTickers)
nBarsPerTicker = int(len(testingSet.allBars) / len(testingSet.allTickers))

for iTimeBar in range(nBarsPerTicker):
    inputTensors = []
    for iTicker in range(nTickers):
        inputTensors.append(testingSet.getInputTensor(iTimeBar * nTickers, iTicker))
    inputTensor = torch.stack(inputTensors)

    # do the inference
    output = model(inputTensor)
    # convert output to probabilities
    outputPrbb = torch.softmax(output, dim=0)

    outputList = outputPrbb.tolist()
    # find the ticker for which we're most sure what to do

    i = 0
