# -*- coding: utf-8 -*-
# Added copy mechanism from See et al., 2017 (CL2017118)
import dynet as dy
import numpy as np

from decoder.plugins.PluginBase import PluginBase


class DecoderPlugin(PluginBase):
    def __init__(self, decoder):
        self.alignInd = 0
        return

    def convertDataset(self, decoder, dataset):
        sample = dataset[0]
        self.alignInd = len(sample)
        result = []
        for sample in dataset:
            f, e = sample[:2]
            sampleAlign = []
            for item in e:
                sampleAlign.append([i for i in range(len(f)) if f[i] == item])
            result.append(sample + (sampleAlign,))
        return result

    def buildModel(self, decoder, pc, inDim, hidDim, layers):
        decoder.cp2_h = pc.add_parameters((3 * hidDim))
        decoder.cp2_b = pc.add_parameters((1))
        return

    def distLoss(self, decoder, i, sample, h_hat, alpha, dist, word, hx):
        # Plus [[]] for the final <EOS> symbol
        align = sample[self.alignInd] + [[]]
        p_gen = dy.logistic(
            dy.dot_product(decoder.cp2_h, h_hat) + decoder.cp2_b)
        copyAlpha = 0
        for value in align[i]:
            copyAlpha += alpha[value]
        loss = []
        if decoder.int2y[word] == "<UNK>":
            loss.append(-dy.log(1 - p_gen))
        return loss + [-dy.log(p_gen * dist[word] + (1 - p_gen) * copyAlpha)]

    def computeSentLoss(self, decoder, hx, sample):
        return

    def distDecode(self, decoder, rawF, e, h_hat, alpha, dist):
        p_gen = dy.logistic(dy.dot_product(decoder.cp2_h, h_hat) +
                            decoder.cp2_b)
        yIndex = np.argmax(dist.npvalue())
        if p_gen.value() >= 0.5 and decoder.int2y[yIndex] != '<UNK>':
            return None
        else:
            copyIndex = np.argmax(alpha.npvalue())
            return rawF[copyIndex]
