__author__ = 'Shawn'

from Sampler import *
import sys


if __name__ == '__main__':
    sampler = sample(sys.argv[1])
    sampler.setCounter()
    sampler.sampler(1)
    sampler.load()
    sampler.split()
    sampler.set_training(sampler.zscale(sampler.training_set))
    sampler.set_test(sampler.zscale(sampler.test_set))



