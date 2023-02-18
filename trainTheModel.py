import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import csv
from datetime import datetime

import functools

# global constants
STRONG_SELL = 0
SELL = 1
BUY = 2
STRONG_BUY = 3
NOT_SURE = 4
N_PAST_BARS = 3
N_FUTURE_BARS = 1
N_MAX_TICKERS = 10
N_MAX_BARS_PER_TICKER = 100

class Bar:
    def __init__(self, iTicker, csvRow):
        self.iTicker = -1;
        # try to read data from the row
        if (len(csvRow) < 6):
            return
        try:
            substring = csvRow[0][0:19]
            self.startTime = datetime.strptime(substring, "%Y-%m-%d %H:%M:%S")
            self.fOpen = float(csvRow[1])
            self.fMax = float(csvRow[2])
            self.fMin = float(csvRow[3])
            self.fClose = float(csvRow[4])
            self.fVolume = float(csvRow[5])
        except: return
        self.iTicker = iTicker

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

    def updateScalingValues(self, iStartTime):
        nTickers = len(self.allTickers)
        self.pVolScalers = [0] * nTickers
        self.pPriceScalers = [0] * nTickers
        pCounts = [0] * nTickers
        for iBar in range(iStartTime, iStartTime + N_PAST_BARS * nTickers):
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

    def __getitem__(self, idx):
        nTickers = len(self.allTickers)
        iStartTime = int(idx / nTickers)
        self.updateScalingValues(iStartTime)
        iPredictedTicker = idx % nTickers
 
        x = self.getPredictedTickerTensor(iPredictedTicker)

        # encode all past bars (must be nTickers * N_PAST_BARS tensors)
        for iBar in range(0, nTickers * N_PAST_BARS):
            tBar = self.getBarTensor(iStartTime * nTickers + iBar)
            x = torch.cat((x, tBar), 0)

        iPresentBar = iPredictedTicker + nTickers * (iStartTime + N_PAST_BARS - 1)
        y = self.getDecisionTensor(iPresentBar)

        return x, y

allTickers, allBars = readTickersAndBars()
dataSet = BarsDataSet(allTickers, allBars)

# check that the number of items is correct
firstItem = dataSet.__getitem__(0)
lastItem = dataSet.__getitem__(dataSet.nTotalExamples - 1)
errorOccurred = False
try:
    invalidItem = dataSet.__getitem__(dataSet.nTotalExamples)
except: errorOccurred = True
assert(errorOccurred) # error must have occurred because we were trying to get a non-existent item
