# -*- coding: utf-8 -*-
# Use copy mechanism for dictionary (Phrase Table with single words)
# LexN1M10 reference: 20190814 09
import os
import dynet as dy
import numpy as np
import natlang as nl
from support.logger import logging
logger = logging.getLogger('PLUGIN')

from decoder.plugins.CopySimple import DecoderPlugin as PluginBase
from decoder.plugins.LexPG import loadFreqN1M3


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
        return

    def distDecode(self, decoder, rawF, e, h_hat, alpha, dist):
        if not hasattr(self, 'LexN1GammaProb'):
            self.LexN1GammaProb = 0
            self.LexN1GammaCount = 0
            self.LexN1ThetaProb = 0
            self.LexN1ThetaCount = 0
        yIndex = np.argmax(dist.npvalue())
        if decoder.int2y[yIndex] != '<UNK>':
            self.LexN1GammaProb += dist.npvalue()[yIndex]
            self.LexN1GammaCount += 1
            return None
        else:
            self.LexN1ThetaProb += dist.npvalue()[yIndex]
            self.LexN1ThetaCount += 1
            copyProb = alpha.npvalue()
            copyIndex = np.argmax(copyProb)
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
