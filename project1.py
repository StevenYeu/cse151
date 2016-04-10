import numpy
import math
#Shiyao Liu, A10626758

class abalone:
    def __init__(self):
        self.sampleCounter = {}
        with open('abalone.data') as f:
            self.sampleSize = sum(1 for _ in f)
        for i in range(self.sampleSize):
            self.sampleCounter[i] = 0
        self.Nn = round(self.sampleSize*0.1)
        numpy.random.seed(1)
        self.genRandom = 0
        self.Nr = self.sampleSize

    def sampler(self, iters):
        for _ in range(iters):
            for i in range(self.sampleSize):
                self.genRandom = numpy.random.rand()

                if self.Nr == 0:
                    break
                pickP = self.Nn/self.Nr
                if self.genRandom < pickP:
                    self.Nn -= 1
                    self.sampleCounter[i] += 1
                self.Nr -= 1

            self.Nr = self.sampleSize
            self.Nn = round(self.sampleSize*0.1)

        mean = numpy.mean(list(self.sampleCounter.values()))/float(iters)
        std = numpy.std(list(self.sampleCounter.values()))/float(iters)

        return mean, std

if __name__ == '__main__':
    for i in range(1,6):
        sample = abalone()
        print(sample.sampler(10 ** i))




