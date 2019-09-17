# -*- coding: utf-8 -*-
# Use copy mechanism for dictionary (Phrase Table with single words)
# LexN1M8 reference: 20190726 13:36
import os
import dynet as dy
import numpy as np
import natlang as nl
from support.logger import logging
logger = logging.getLogger('PLUGIN')

from decoder.plugins.LexPG import DecoderPlugin as PluginBase


class DecoderPlugin(PluginBase):
    def __init__(self, decoder, lexFiles=[], threshold=0.5):
        PluginBase.__init__(self, decoder, lexFiles, threshold)
        self.FEATS = 3
        return

    def buildModel(self, decoder, pc, inDim, hidDim, layers):
        FEAT_DIM = 3 * hidDim
        decoder.lexN1M8_v_gen = pc.add_parameters((1, 3 * hidDim))
        decoder.lexN1M8_W_gen = pc.add_parameters((3 * hidDim, 3 * hidDim))
        decoder.lexN1M8_b_gen = pc.add_parameters((3 * hidDim))
        decoder.lexN1M8_v_pc = pc.add_parameters((1, FEAT_DIM))
        decoder.lexN1M8_W_pc = pc.add_parameters((FEAT_DIM, 3 * hidDim))
        decoder.lexN1M8_U_pc = pc.add_parameters((FEAT_DIM, 2 * hidDim))
        decoder.lexN1M8_O_pc = pc.add_parameters((FEAT_DIM, self.FEATS))
        return

    def convertDataset(self, decoder, dataset):
        dataset = PluginBase.convertDataset(self, decoder, dataset)
        sample = dataset[0]
        self.featInd = len(sample)
        result = []
        for sample in dataset:
            f = sample[0]
            features = [self.lexFeat(word) for word in f]
            result.append(sample + (features,))
        return result

    def lexFeat(self, word):
        feature = np.zeros(self.FEATS)
        if word in self.lexicon:
            feature[0] = 1
            if len(self.lexicon[word]) != 1:
                feature[1] = 1
            if word in self.lexicon[word]:
                feature[2] = 1
        return feature

    def lexPC(self, decoder, feats):
        Why = decoder.lexN1M8_W_pc * decoder.h_hat
        PC = [decoder.lexN1M8_v_pc * dy.tanh(
            Why +
            decoder.lexN1M8_U_pc * decoder.hx[i] +
            decoder.lexN1M8_O_pc * dy.inputVector(feats[i]))
            for i in range(len(decoder.hx))]
        p_gen = decoder.lexN1M8_v_gen * dy.tanh(
            decoder.lexN1M8_W_gen * decoder.h_hat + decoder.lexN1M8_b_gen)
        PC = [p_gen] + PC
        PC = dy.softmax(dy.concatenate(PC))
        return PC

    def distLoss(self, decoder, i, sample, h_hat, alpha, dist, word, hx):
        # Plus [[]] for the final <EOS> symbol
        align = sample[self.alignInd] + [[]]
        features = sample[self.featInd]
        PC = self.lexPC(decoder, features)
        p_gen, PC = PC[0], PC[1:]
        lexPC = 0
        for value in align[i]:
            lexPC += alpha[value] * PC[value]
        loss = []
        if decoder.int2y[word] == "<UNK>":
            loss.append(-dy.log(1 - p_gen))
        return loss + [-dy.log(p_gen * dist[word] + lexPC)]

    def distDecode(self, decoder, rawF, e, h_hat, alpha, dist):
        if not hasattr(self, 'LexN1GammaProb'):
            self.LexN1GammaProb = 0
            self.LexN1GammaCount = 0
            self.LexN1ThetaProb = 0
            self.LexN1ThetaCount = 0
        features = [self.lexFeat(word) for word in rawF]
        PC = self.lexPC(decoder, features).value()
        p_gen, PC = PC[0], PC[1:]
        yIndex = np.argmax(dist.npvalue())
        if (p_gen >= self.threshold and
                decoder.int2y[yIndex] != '<UNK>'):
            self.LexN1GammaProb += dist.npvalue()[yIndex]
            self.LexN1GammaCount += 1
            return None
        else:
            self.LexN1ThetaProb += dist.npvalue()[yIndex]
            self.LexN1ThetaCount += 1
            copyProb = PC * alpha.npvalue()
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
