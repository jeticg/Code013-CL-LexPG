# -*- coding: utf-8 -*-
# Use copy mechanism for dictionary (Phrase Table with single words)
# LexN1M11 reference: 20190814 09:00
import os
import dynet as dy
import numpy as np
import natlang as nl
from support.logger import logging
logger = logging.getLogger('PLUGIN')

from decoder.plugins.LexPG_F import DecoderPlugin as PluginBase


class DecoderPlugin(PluginBase):
    def buildModel(self, decoder, pc, inDim, hidDim, layers):
        self.FEAT_DIM = 3 * hidDim
        self.FEATS = 3
        decoder.lexN1M8_v_gen = pc.add_parameters((1, 3 * hidDim))
        decoder.lexN1M8_W_gen = pc.add_parameters((3 * hidDim, 3 * hidDim))
        decoder.lexN1M8_b_gen = pc.add_parameters((3 * hidDim))
        decoder.lexN1M8_v_pc = pc.add_parameters((1, self.FEAT_DIM))
        decoder.lexN1M8_W_pc = pc.add_parameters((self.FEAT_DIM, 3 * hidDim))
        decoder.lexN1M8_U_pc = pc.add_parameters((self.FEAT_DIM, 2 * hidDim))
        decoder.lexN1M8_O_pc = pc.add_parameters((self.FEAT_DIM, self.FEATS))

        # LexBeta
        decoder.lexN1M6_V = pc.add_parameters((1, 3 * hidDim))
        decoder.lexN1M6_W = pc.add_parameters((3 * hidDim, 3 * hidDim))
        decoder.lexN1M6_U = pc.add_parameters((3 * hidDim, 2 * hidDim))
        return

    def lexBeta(self, decoder):
        W, U, V = decoder.lexN1M6_W, decoder.lexN1M6_U, decoder.lexN1M6_V
        if len(decoder.hx) == 1:
            return dy.zeros((1,)) + 1
        Why = W * decoder.h_hat
        scores =\
            [V * dy.tanh(Why + U * hx_i) for hx_i in decoder.hx]
        beta = dy.softmax(dy.concatenate(scores))
        return beta

    def distLoss(self, decoder, i, sample, h_hat, alpha, dist, word, hx):
        align = sample[self.alignInd] + [[]]
        features = sample[self.featInd]
        PC = self.lexPC(decoder, features)
        p_gen, PC = PC[0], PC[1:]
        beta = self.lexBeta(decoder)
        lexPC = 0
        for value in align[i]:
            lexPC += beta[value] * PC[value]
        lexPD = (1 - p_gen) * sum([beta[value] for value in align[i]])
        loss = []
        if decoder.int2y[word] == "<UNK>":
            loss.append(-dy.log(1 - p_gen))
        return loss +\
            [-dy.log(p_gen * dist[word] + (lexPC + lexPD) / 2)]

    def distDecode(self, decoder, rawF, e, h_hat, alpha, dist):
        return PluginBase.distDecode(self, decoder, rawF, e, h_hat,
                                     self.lexBeta(decoder),
                                     dist)
