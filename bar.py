from datetime import datetime

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
    def notifyNoData(self, iTicker):
        self.fOpen = None
        self.fMax = None
        self.fMin = None
        self.fClose = None
        self.fVolume = None
        self.iTicker = iTicker
