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
        # each example is essentially an index of a bar that we think we should be able to predict the future for
        # to qualify - the bar must have valid PAST_BARS and valid FUTURE_BARS
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

        # prepare the arrays where we'll cache tensors during training
        nBarsPerTicker = int(len(allBars) / len(allTickers))
        assert(nBarsPerTicker * len(allTickers) == len(allBars))
        self.pPriceScalers = [None] * nBarsPerTicker
        self.pVolumeScalers = [None] * nBarsPerTicker
        self.pOneBarAllTickersTensors = [None] * nBarsPerTicker
        self.pDecisionTensors = [None] * len(self.allExamples)
        self.pTimeTensors = [None] * nBarsPerTicker

    def __len__(self):
        return len(self.allExamples)

    def getPredictedTickerTensor(self, iPredictedTicker):
        nTickers = len(self.allTickers)
        tPredictedTicker = torch.zeros(nTickers)
        tPredictedTicker[iPredictedTicker] = 1
        return tPredictedTicker

    # to get values in approximately [0, 1] interval, we scale prices and volume by coefficients equal to
    # average of prices and volumes in that interval. we compute those coefficients once and cache them in
    # self.pVolumeScalers and self.pPriceScalers
    def getScalerTensors(self, iFirstBar):
        iTimeBar = int(iFirstBar / len(self.allTickers))
        if (self.pPriceScalers[iTimeBar] is None):
            assert(self.allBars[iFirstBar].iTicker == 0) # check that iFirstBar makes sense
            nTickers = len(self.allTickers)
            pVolumeScalers = [0] * nTickers
            pPriceScalers = [0] * nTickers
            pCounts = [0] * nTickers
            for iBar in range(iFirstBar, iFirstBar + N_PAST_BARS * nTickers):
                bar = self.allBars[iBar]
                if (bar.fVolume is None): # means bar is invalid (we have some gaps in the dataset)
                    continue
                pVolumeScalers[bar.iTicker] += bar.fVolume
                pPriceScalers[bar.iTicker] += (bar.fMin + bar.fMax + bar.fOpen + bar.fClose) / 4
                pCounts[bar.iTicker] += 1
            for i in range(0, nTickers):
                assert(pCounts[i] > 0 and pCounts[i] <= N_PAST_BARS)
                pVolumeScalers[i] /= pCounts[i]
                pPriceScalers[i] /= pCounts[i]
            self.pPriceScalers[iTimeBar] = torch.tensor(pPriceScalers)
            self.pVolumeScalers[iTimeBar] = torch.tensor(pVolumeScalers)
        return self.pPriceScalers[iTimeBar], self.pVolumeScalers[iTimeBar]
    
    def getOneBarAllTickersTensor(self, iFirstBar):
            iTimeBar = int(iFirstBar / len(self.allTickers))
            if (self.pOneBarAllTickersTensors[iTimeBar] is None):
                nTickers = len(self.allTickers)
                x = [0] * 5 * nTickers
                for iBar in range(0, nTickers):
                    bar = self.allBars[iFirstBar + iBar]
                    if (bar.fVolume is not None):
                        x[iBar] = bar.fMin
                        x[iBar + nTickers] = bar.fMax
                        x[iBar + nTickers * 2] = bar.fClose
                        x[iBar + nTickers * 3] = bar.fVolume
                        x[iBar + nTickers * 4] = 1 # indicates this is a valid bar
                self.pOneBarAllTickersTensors[iTimeBar] = torch.tensor(x)
            return self.pOneBarAllTickersTensors[iTimeBar]

    def getPastBarsTensor(self, iFirstBar):
            nTickers = len(self.allTickers)
            pPriceScalers, pVolumeScalers = self.getScalerTensors(iFirstBar)
            xx = torch.tensor([])

            for i in range(N_PAST_BARS):
                x = self.getOneBarAllTickersTensor(iFirstBar + i * nTickers)
                # normalize prices and volume by scalers
                x[:nTickers] /= pPriceScalers;
                x[nTickers:2*nTickers] /= pPriceScalers
                x[2*nTickers:3*nTickers] /= pPriceScalers
                x[3*nTickers:4*nTickers] /= pVolumeScalers
                xx = torch.cat((xx, x), 0)

            return xx

    def getDecisionTensor(self, iExample, iPresentBar):
        if (self.pDecisionTensors[iExample] is None):
            nTickers = len(self.allTickers)
            presentBar = self.allBars[iPresentBar]

            # one-hot encoding. classes are: strong-sell, sell, not-sure, buy, strong-buy
            y = [0] * 5
            for i in range(0, N_FUTURE_BARS):
                iFutureBar = iPresentBar + (i + 1) * nTickers
                futureBar = self.allBars[iFutureBar]
                if (futureBar.fMax / presentBar.fClose > 1.05):
                    y[STRONG_BUY] = 1.
                elif (futureBar.fMax / presentBar.fClose > 1.025):
                    y[BUY] = 1.
                if (futureBar.fMin / presentBar.fClose < 0.95):
                    y[STRONG_SELL] = 1.
                elif (futureBar.fMin / presentBar.fClose < 0.975):
                    y[SELL] = 1.

            nNonZero = 5 - y.count(0)
            if (nNonZero > 1):
                y = [0] * 5
                y[NOT_SURE] = 1.
            elif (nNonZero == 0):
                y[NOT_SURE] = 1.

            self.pDecisionTensors[iExample] = torch.tensor(y)

        return self.pDecisionTensors[iExample]

    def getTimeTensor(self, iFirstBar):
        nTickers = len(self.allTickers)
        iTime = int(iFirstBar / nTickers)
        if self.pTimeTensors[iTime] is None:
            iFirstBar = iTime * nTickers
            x = [0] * (5 + 7 + N_PAST_BARS - 1)
            time0 = self.allBars[iFirstBar].startTime
            weekDay = time0.weekday()
            assert(weekDay >= 0 and weekDay < 5)
            x[weekDay] = 1
            hour = time0.hour - 9
            assert(hour >= 0 and hour < 7)
            x[5 + hour] = 1
            # this takes a note if the time difference between the bars is non-standard
            for i in range(1, N_PAST_BARS):
                time1 = self.allBars[iFirstBar + i * nTickers].startTime
                diffSeconds = int((time1 - time0).total_seconds())
                time0 = time1
                if (diffSeconds != 3600):
                    x[5 + 7 + i - 1] = 1 # non-standard difference
            self.pTimeTensors[iTime] = torch.tensor(x)
        return self.pTimeTensors[iTime]

    def __getitem__(self, iExample):
        nTickers = len(self.allTickers)
        iExampleBar = self.allExamples[iExample]
        iFirstBar = int(iExampleBar / nTickers) * nTickers
        iPredictedTicker = self.allBars[iExampleBar].iTicker
 
        x = self.getPredictedTickerTensor(iPredictedTicker)

        # encode time information
        tTime = self.getTimeTensor(iFirstBar)
        x = torch.cat((x, tTime), 0)

        # encode all past bars
        tBar = self.getPastBarsTensor(iFirstBar)
        x = torch.cat((x, tBar), 0)

        # encode the trading decision
        iPresentBar = iFirstBar + iPredictedTicker + nTickers * (N_PAST_BARS - 1)
        y = self.getDecisionTensor(iExample, iPresentBar)

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