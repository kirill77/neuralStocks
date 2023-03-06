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
model.loadFromDir(args.sModelDir)
# disable gradients (only inference here)
model.eval()
torch.set_grad_enabled(False)

nTickers = len(testingSet.allTickers)
nBarsPerTicker = int(len(testingSet.allBars) / len(testingSet.allTickers))

nTradableTimeBars = nBarsPerTicker - (myConstants.N_PAST_BARS + myConstants.N_FUTURE_BARS - 1)
for iTimeBar in range(nTradableTimeBars):
    inputTensors = []
    for iTicker in range(nTickers):
        inputTensors.append(testingSet.getInputTensor(iTimeBar * nTickers, iTicker))
    inputTensor = torch.stack(inputTensors)

    # do the inference
    output = model(inputTensor)

    # convert output to probabilities
    outputPrbb = torch.softmax(output, dim=1)

    y = outputPrbb.tolist()

    # find the ticker for which we're most sure what to do
    fMaxPrbb = -1
    iBestTicker = None
    decision = None
    for iTicker in range(nTickers):
        fBuyPrbb = y[iTicker][myConstants.BUY] + y[iTicker][myConstants.STRONG_BUY]
        if (fBuyPrbb > fMaxPrbb):
            fMaxPrbb = fBuyPrbb
            iBestTicker = iTicker
            decision = myConstants.BUY
        fSellPrbb = y[iTicker][myConstants.SELL] + y[iTicker][myConstants.STRONG_SELL]
        if (fSellPrbb > fMaxPrbb):
            fMaxPrbb = fSellPrbb
            iBestTicker = iTicker
            decision = myConstants.SELL

    # if probability is higher than the threshold
    if (fMaxPrbb > myConstants.TRADE_THRESHOLD):
         virtualTrader.makeATrade(testingSet, iTimeBar, iTicker, decision)
  