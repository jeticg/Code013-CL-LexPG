# -*- coding: utf-8 -*-
# Use copy mechanism for dictionary (Phrase Table with single words)
# LexN1M3 reference: 20190609 17:00
# One-to-one translation used here.
import os
import dynet as dy
import numpy as np
import natlang as nl
from support.logger import logging
logger = logging.getLogger('PLUGIN')

from decoder.plugins.PluginBase import PluginBase


def loadFreqN1M3(fileName):
    data = nl.load(fileName)
    lex = {}

    for item in data:
        if item[0] not in lex:
            lex[item[0]] = {}
        lex[item[0]][item[1]] = int(item[2])
    return lex


class DecoderPlugin(PluginBase):
    def __init__(self, decoder, lexFiles=[], threshold=0.5):
        self.lexicon = {}
        for lexFile in lexFiles:
            logger.info('Loading ' + lexFile)
            loaded = loadFreqN1M3(lexFile)
            for word in loaded:
                if word not in self.lexicon:
                    self.lexicon[word] = loaded[word]
                else:
                    for eWord in loaded[word]:
                        if eWord not in self.lexicon[word]:
                            self.lexicon[word][eWord] = 0
                        self.lexicon[word][eWord] += loaded[word][eWord]
        logger.info('Lexicon loaded, ' + str(len(self.lexicon)) + ' entries')

        self.threshold = threshold
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
                sampleAlign.append(
                    [i for i in range(len(f))
                     if (f[i] in self.lexicon and
                         item in self.lexicon[f[i]]) or
                         f[i] == item
                    ]
            )
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
        lexAlpha = 0
        for value in align[i]:
            lexAlpha += alpha[value]
        loss = []
        if decoder.int2y[word] == "<UNK>":
            loss.append(-dy.log(1 - p_gen))
        return loss + [-dy.log(p_gen * dist[word] + (1 - p_gen) * lexAlpha)]

    def computeSentLoss(self, decoder, hx, sample):
        return

    def refresh(self, decoder, rawF, hx):
        # To support models trained before threshold interface was introduced
        if not hasattr(self, 'threshold'):
            self.threshold = 0.5
        return

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
            copyIndex = np.argmax(alpha.npvalue())
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
