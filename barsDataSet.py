import torch
from torch.utils.data import Dataset
import functools
import csv
from datetime import datetime
from bar import Bar
import copy

# global constants
N_MAX_TICKERS = 10
N_MAX_BARS_PER_TICKER = -1
N_TRAINING_BARS_PERCENT = 80
STRONG_SELL = 0
SELL = 1
BUY = 2
STRONG_BUY = 3
NOT_SURE = 4
N_PAST_BARS = 3
N_FUTURE_BARS = 1

def compareBars(a, b):
    if a.startTime < b.startTime:
        return -1
    if a.startTime > b.startTime:
        return 1
    return -1 if a.iTicker < b.iTicker else 1

def validateBars(allTickers, allBars):
    nTickers = len(allTickers)
    for barIndex, bar in enumerate(allBars):
        # check that we're not missing any tickers and they go in ascending order
        assert(bar.iTicker == barIndex % nTickers)
        if (bar.iTicker == 0):
            # check that ticker 0 has larger time than the previous ticker
            assert(barIndex == 0 or bar.startTime > allBars[barIndex - 1].startTime)
        else:
            # all all other tickers have the same time as the previous ticker
            assert(bar.startTime == allBars[barIndex - 1].startTime)

class BarsDataSet(Dataset):

    def __init__(self, allTickers, allBars):
        self.allTickers = allTickers
        self.allBars = allBars

        validateBars(self.allTickers, self.allBars)

        # create all examples
        # to qualify as an example - all PAST_BARS and all FUTURE_BARS of the given stock must be known
        self.allExamples = []
        for firstBarIndex in range(0, len(allBars)):
            allValid = True
            for i in range(0, N_PAST_BARS + N_FUTURE_BARS):
                barIndex = firstBarIndex + i * len(allTickers)
                if barIndex >= len(allBars) or allBars[barIndex].fVolume is None:
                    allValid = False
                    break
            if (allValid):
                self.allExamples.append(firstBarIndex)

    def __len__(self):
        return len(self.allExamples)

    def getPredictedTickerTensor(self, iPredictedTicker):
        nTickers = len(self.allTickers)
        tPredictedTicker = torch.zeros(nTickers)
        tPredictedTicker[iPredictedTicker] = 1
        return tPredictedTicker

    def updateScalingValues(self, iFirstBar):
        assert(self.allBars[iFirstBar].iTicker == 0) # check that iFirstBar makes sense
        nTickers = len(self.allTickers)
        self.pVolScalers = [0] * nTickers
        self.pPriceScalers = [0] * nTickers
        pCounts = [0] * nTickers
        for iBar in range(iFirstBar, iFirstBar + N_PAST_BARS * nTickers):
            bar = self.allBars[iBar]
            if (bar.fVolume is None): # means bar is invalid (we have some gaps in the dataset)
                continue
            self.pVolScalers[bar.iTicker] += bar.fVolume
            self.pPriceScalers[bar.iTicker] += (bar.fMin + bar.fMax + bar.fOpen + bar.fClose) / 4
            pCounts[bar.iTicker] += 1
        for i in range(0, nTickers):
            assert(pCounts[i] > 0 and pCounts[i] <= N_PAST_BARS)
            self.pVolScalers[i] / pCounts[i]
            self.pPriceScalers[i] / pCounts[i]

    def getBarTensor(self, iBar):
            bar = self.allBars[iBar]
            fVolScaler = self.pVolScalers[bar.iTicker]
            fPriceScaler = self.pPriceScalers[bar.iTicker]
            if (bar.fVolume is None):
                tmpList = [0, 0, 0, 0, 0]
            else:
                tmpList = [1, bar.fMin / fPriceScaler, bar.fMax / fPriceScaler, bar.fClose / fPriceScaler, bar.fVolume / fVolScaler]
            return torch.tensor(tmpList)

    def getDecisionTensor(self, iPresentBar):
        nTickers = len(self.allTickers)
        presentBar = self.allBars[iPresentBar]
        # strong-sell, sell, not-sure, buy, strong-buy
        y = torch.zeros(5)
        for i in range(0, N_FUTURE_BARS):
            iFutureBar = iPresentBar + (i + 1) * nTickers
            futureBar = self.allBars[iFutureBar]
            if (futureBar.fMax / presentBar.fClose > 1.05):
                y[STRONG_BUY] = 1
            elif (futureBar.fMax / presentBar.fClose > 1.025):
                y[BUY] = 1
            if (futureBar.fMin / presentBar.fClose < 0.95):
                y[STRONG_SELL] = 1
            elif (futureBar.fMin / presentBar.fClose < 0.975):
                y[SELL] = 1

        nNonZero = torch.count_nonzero(y).item()
        if (nNonZero > 1):
            y = torch.zeros(5)
            y[NOT_SURE] = 1
        elif (nNonZero == 0):
            y[NOT_SURE] = 1

        return y

    def getTimeTensor(self, iFirstBar):
        x = torch.zeros(5 + 7 + N_PAST_BARS - 1)
        time0 = self.allBars[iFirstBar].startTime
        weekDay = time0.weekday()
        assert(weekDay >= 0 and weekDay < 5)
        x[weekDay] = 1
        hour = time0.hour - 9
        assert(hour >= 0 and hour < 7)
        x[5 + hour] = 1
        nTickers = len(self.allTickers)
        # this takes a note if the time difference between the bars is non-standard
        for i in range(1, N_PAST_BARS):
            time1 = self.allBars[iFirstBar + i * nTickers].startTime
            diffSeconds = int((time1 - time0).total_seconds())
            time0 = time1
            if (diffSeconds != 3600):
                x[5 + 7 + i - 1] = 1
        return x

    def __getitem__(self, idx):
        nTickers = len(self.allTickers)
        iExampleBar = self.allExamples[idx]
        iFirstBar = int(iExampleBar / nTickers) * nTickers
        iPredictedTicker = self.allBars[iExampleBar].iTicker
 
        x = self.getPredictedTickerTensor(iPredictedTicker)

        # encode time information
        tTime = self.getTimeTensor(iFirstBar)
        x = torch.cat((x, tTime), 0)

        # encode all past bars
        self.updateScalingValues(iFirstBar)
        for iBar in range(0, nTickers * N_PAST_BARS):
            tBar = self.getBarTensor(iFirstBar + iBar)
            x = torch.cat((x, tBar), 0)

        # encode the trading decision
        iPresentBar = iFirstBar + iPredictedTicker + nTickers * (N_PAST_BARS - 1)
        y = self.getDecisionTensor(iPresentBar)

        return x, y

def readTickersAndBars():
    # load list of tickers
    with open("tickers.txt", 'r') as f:
        data = f.read()
    allTickers = data.splitlines()
    allTickers = [w.strip() for w in allTickers] # get rid of any leading or trailing white space
    allTickers = [w for w in allTickers if w] # get rid of any empty strings
    if (N_MAX_TICKERS != -1):
        allTickers = allTickers[0:N_MAX_TICKERS]
    print(f"number of tickers in the dataset: {len(allTickers)}")

    allBars = []

    # for each ticker - load its bars from the file
    for iTicker, sTicker in enumerate(allTickers):
        fileName = f"./history/{sTicker}_1h.csv"
        print(f"loading bars for {sTicker}")
        nBarsThisTicker = 0;
        with open (fileName) as csvFile:
            csvRows = csv.reader(csvFile)
            for csvRow in csvRows:
                bar = Bar(iTicker, csvRow)
                if (bar.iTicker < 0):
                    continue
                allBars.append(bar)
                nBarsThisTicker += 1
                if N_MAX_BARS_PER_TICKER > 0 and nBarsThisTicker >= N_MAX_BARS_PER_TICKER:
                    break
    print(f"loaded {len(allBars)} bars")

    return allTickers, allBars

def verifyBarsDataSet(dataSet):
    # check that the number of items is correct
    firstItem = dataSet.__getitem__(0) # first item must exist
    lastItem = dataSet.__getitem__(len(dataSet) - 1) # last item must exist
    # one after last item must NOT exist
    errorOccurred = False
    try:
        invalidItem = dataSet.__getitem__(len(dataSet))
    except: errorOccurred = True
    assert(errorOccurred)

def createBarDataSets():
    allTickers, allBars = readTickersAndBars()

    # sort all bars by time, and then by the ticker index
    print("sorting bars...")
    allBars.sort(key = functools.cmp_to_key(compareBars))

    # insert invalid bars at the places where some data was missing
    print("fixing bars...")
    nTickers = len(allTickers)
    for barIndex, bar in enumerate(allBars):
        assumedTickerIndex = barIndex % nTickers
        if (bar.iTicker != assumedTickerIndex):
            if (assumedTickerIndex > 0):
                allBars.insert(barIndex, copy.copy(allBars[barIndex - 1]))
            else:
                allBars.insert(barIndex, copy.copy(allBars[barIndex]))
            allBars[barIndex].notifyNoData(assumedTickerIndex)

    print("validating bars...")
    validateBars(allTickers, allBars)

    # split all bars to training and testing datasets
    print("splitting bars to training and testing datasets...")
    nBarsPerTicker = len(allBars) / nTickers
    assert(nBarsPerTicker == int(nBarsPerTicker))
    nBarsPerTicker = int(nBarsPerTicker)
    nTrainingBarsPerTicker = int(nBarsPerTicker * N_TRAINING_BARS_PERCENT / 100)
    nTrainingBars = nTrainingBarsPerTicker * nTickers;
    trainingBars = allBars[0:nTrainingBars]
    testingBars = allBars[nTrainingBars:]

    print("creating training and testing datasets...")
    trainingSet = BarsDataSet(allTickers, trainingBars)
    verifyBarsDataSet(trainingSet)
    testingSet = BarsDataSet(allTickers, testingBars)
    verifyBarsDataSet(testingSet)
    return trainingSet, testingSet