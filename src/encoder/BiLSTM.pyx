# -*- coding: utf-8 -*-
import dynet as dy
import numpy as np

from support.logger import logging
from models.ModelBase import ModelBase
logger = logging.getLogger("ENCODER")


class Encoder(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)
        self.x2int = {}
        self.int2x = []
        self.fInd = 0
        self.component = ["inDim", "hidDim", "layers",
                          "x2int", "int2x"]
        return

    def buildLexicon(self, dataset, lexiconSize=50000, minFrequency=3):
        index = 0
        logger.info("Encoder lexicon minFreq: {}".format(minFrequency))
        self.x2int, self.int2x = ModelBase.buildLexicon(
            self, dataset, index=index, lexiconSize=lexiconSize,
            minFrequency=minFrequency)
        logger.info("Encoder lexicon size: {}".format(len(self.int2x)))
        return

    def convertDataset(self, dataset):
        sample = dataset[0]
        self.fInd = len(sample)
        result = [sample + (sample[0],) for sample in dataset]
        return ModelBase.convertDataset(
            self, result, index=self.fInd,
            w2int=self.x2int, int2w=self.int2x)

    def buildModel(self, pc, inputDim, hiddenDim, layers):
        logger.info("inDim={}, hidDim={}".format(inputDim, hiddenDim))
        self.inDim = inputDim
        self.hidDim = hiddenDim
        self.layers = layers
        self.xLookup = pc.add_lookup_parameters((len(self.int2x), inputDim))
        self.encoderForward =\
            dy.LSTMBuilder(layers, inputDim, hiddenDim, pc)
        self.encoderBackward =\
            dy.LSTMBuilder(layers, inputDim, hiddenDim, pc)
        return

    def encode(self, rawInput):
        tokens = self.tokenise(rawInput)
        sent = [self.x2int[word] for word in tokens]
        return self.encodeSent(sent)

    def tokenise(self, rawInput):
        return [word if word in self.x2int else '<UNK>' for word in rawInput]

    def encodeSent(self, sent):
        forwardOutput = []
        backwardOutput = []
        hx = []

        # initialise states
        forwardState = self.encoderForward.initial_state()
        backwardState = self.encoderBackward.initial_state()

        # iterate in both directions
        for i in range(len(sent)):
            forwardState = forwardState.add_input(
                dy.lookup(self.xLookup, sent[i]))
            forwardOutput.append(forwardState.output())

            backwardState = backwardState.add_input(
                dy.lookup(self.xLookup, sent[len(sent) - i - 1]))
            backwardOutput.insert(0, backwardState.output())

        # concatenate forward and backward outputs
        for i in range(len(sent)):
            hx.append(
                dy.concatenate([forwardOutput[i], backwardOutput[i]]))

        return hx
