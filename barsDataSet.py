import torch
from torch.utils.data import Dataset
import functools

# global constants
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

class BarsDataSet(Dataset):

    def __init__(self, allTickers, allBars):
        self.allTickers = allTickers
        self.allBars = allBars

        # sort all bars by time, and then by the ticker index
        allBars.sort(key = functools.cmp_to_key(compareBars))

        nTickers = len(self.allTickers)
        for barIndex in range(nTickers, len(allBars), nTickers):
            # check that dates to in ascending order
            assert(allBars[barIndex].startTime > allBars[barIndex - nTickers].startTime)

        for barIndex, bar in enumerate(allBars):
            assumedTickerIndex = barIndex % nTickers
            # check that we're not missing any tickers and they go in ascending order
            assert(assumedTickerIndex == bar.iTicker)
            # check that 
            assert(bar.iTicker == 0 or bar.startTime == allBars[barIndex - 1].startTime)

        nStartTimes = len(self.allBars) / nTickers

        self.nTotalExamples = int(nStartTimes - N_PAST_BARS - N_FUTURE_BARS + 1)
        self.nTrainingExamples = int(self.nTotalExamples * 9 / 10)
        # on each timestep each ticker is a separate example
        self.nTotalExamples *= len(allTickers)
        self.nTrainingExamples *= len(allTickers)

    def __len__(self):
        return self.nTrainingExamples

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
            self.pVolScalers[bar.iTicker] += bar.fVolume
            self.pPriceScalers[bar.iTicker] += (bar.fMin + bar.fMax + bar.fOpen + bar.fClose) / 4
            pCounts[bar.iTicker] += 1
        for i in range(0, nTickers):
            assert(pCounts[i] == N_PAST_BARS) # we are supposed to compute average over past bars
            self.pVolScalers[i] / pCounts[i]
            self.pPriceScalers[i] / pCounts[i]

    def getBarTensor(self, iBar):
            bar = self.allBars[iBar]
            fVolScaler = self.pVolScalers[bar.iTicker]
            fPriceScaler = self.pPriceScalers[bar.iTicker]
            tmpList = [bar.fMin / fPriceScaler, bar.fMax / fPriceScaler, bar.fClose / fPriceScaler, bar.fVolume / fVolScaler]
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
            y = torch.zeroes(5)
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
        for i in range(0, N_PAST_BARS):
            time1 = self.allBars[iFirstBar + (i + 1) * nTickers].startTime
            diffSeconds = int((time1 - time0).total_seconds())
            time0 = time1
            if (diffSeconds != 3600):
                x[5 + 7 + i] = 1 # take note if the time difference between the bars is non-standard
        return x

    def __getitem__(self, idx):
        nTickers = len(self.allTickers)
        iFirstBar = nTickers * int(idx / nTickers)
        iPredictedTicker = idx % nTickers
 
        x = self.getPredictedTickerTensor(iPredictedTicker)

        # encode time information
        tTime = self.getTimeTensor(iFirstBar)
        x = torch.cat((x, tTime), 0)

        # encode all past bars (must be nTickers * N_PAST_BARS tensors)
        self.updateScalingValues(iFirstBar)
        for iBar in range(0, nTickers * N_PAST_BARS):
            tBar = self.getBarTensor(iFirstBar + iBar)
            x = torch.cat((x, tBar), 0)

        iPresentBar = iFirstBar + iPredictedTicker + nTickers * (N_PAST_BARS - 1)
        y = self.getDecisionTensor(iPresentBar)

        return x, y
