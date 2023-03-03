
# Simple Data Loader for debugging
class MyDataLoader:
    def __init__(self, dataset = None, batch_size = None, shuffle = True):
        self.pDataSet = dataset
        self.nElementsPerBatch = batch_size

    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        n = min(len(self.pDataSet) - self.index, self.nElementsPerBatch)
        if (n == 0):
            raise StopIteration
        
        _x, _y = self.pDataSet[self.index]
        x = [_x]
        y = [_y]
        self.index += 1
        n -= 1
        for i in range(n):
            _x, _y = self.pDataSet[self.index]
            x.append(_x)
            y.append(_y)
            self.index += 1

        return torch.stack(x), torch.stack(y)
