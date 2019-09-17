# -*- coding: utf-8 -*-
# Use copy mechanism for dictionary (Phrase Table with single words)
# LexN1M6 reference: 20190717 17:40
# One-to-one translation used here.
import os
import dynet as dy
import numpy as np
import natlang as nl
from support.logger import logging
logger = logging.getLogger('PLUGIN')

from decoder.plugins.LexPG import DecoderPlugin as PluginBase


class DecoderPlugin(PluginBase):
    def buildModel(self, decoder, pc, inDim, hidDim, layers):
        PluginBase.buildModel(self, decoder, pc, inDim, hidDim, layers)
        decoder.lexN1M6_V = pc.add_parameters((1, 3 * hidDim))
        decoder.lexN1M6_W = pc.add_parameters((3 * hidDim, 3 * hidDim))
        decoder.lexN1M6_U = pc.add_parameters((3 * hidDim, 2 * hidDim))
        return

    def copyBeta(self, decoder):
        W, U, V = decoder.lexN1M6_W, decoder.lexN1M6_U, decoder.lexN1M6_V
        if len(decoder.hx) == 1:
            return dy.zeros((1,)) + 1
        Why = W * decoder.h_hat
        scores =\
            [V * dy.tanh(Why + U * hx_i) for hx_i in decoder.hx]
        beta = dy.softmax(dy.concatenate(scores))
        return beta

    def distLoss(self, decoder, i, sample, h_hat, alpha, dist, word, hx):
        # Plus [[]] for the final <EOS> symbol
        align = sample[self.alignInd] + [[]]
        p_gen = dy.logistic(
            dy.dot_product(decoder.cp2_h, h_hat) + decoder.cp2_b)
        beta = self.copyBeta(decoder)
        lexBeta = 0
        for value in align[i]:
            lexBeta += beta[value]
        loss = []
        if decoder.int2y[word] == "<UNK>":
            loss.append(-dy.log(1 - p_gen))
        return loss + [-dy.log(p_gen * dist[word] + (1 - p_gen) * lexBeta)]

    def distDecode(self, decoder, rawF, e, h_hat, alpha, dist):
        if not hasattr(self, 'LexN1GammaProb'):
            self.LexN1GammaProb = 0
            self.LexN1GammaCount = 0
            self.LexN1ThetaProb = 0
            self.LexN1ThetaCount = 0

        p_gen = dy.logistic(dy.dot_product(decoder.cp2_h, h_hat) +
                            decoder.cp2_b)
        yIndex = np.argmax(dist.npvalue())
        if (p_gen.value() >= self.threshold and
                decoder.int2y[yIndex] != '<UNK>'):
            self.LexN1GammaProb += dist.npvalue()[yIndex]
            self.LexN1GammaCount += 1
            return None
        else:
            self.LexN1ThetaProb += dist.npvalue()[yIndex]
            self.LexN1ThetaCount += 1
            beta = self.copyBeta(decoder)
            copyIndex = np.argmax(beta.npvalue())
            fWord = rawF[copyIndex]
            if fWord in self.lexicon:
                fLex = self.lexicon[fWord]
            else:
                return rawF[copyIndex]
            dist = dist.npvalue()
            if rawF[copyIndex] in self.lexicon:
                cand = [eWord for eWord in fLex
                        if eWord in decoder.y2int]
                cand = [(eWord, dist[decoder.y2int[eWord]]) for eWord in cand]
                candUNK = [eWord for eWord in fLex
                           if eWord not in decoder.y2int]
                if len(candUNK) != 0:
                    candUNK.sort(key=lambda x: -fLex[x])
                    candUNK = candUNK[0]
                    cand.append((candUNK, dist[decoder.y2int['<UNK>']]))
                cand.sort(key=lambda x: -x[1])
                cand = cand[0]
                return cand[0]
            else:
                return rawF[copyIndex]
