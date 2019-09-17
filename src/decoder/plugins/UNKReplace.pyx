# -*- coding: utf-8 -*-
# Added copy mechanism from Gu et al., 2016
import dynet as dy
import numpy as np
from decoder.plugins.PluginBase import PluginBase


class DecoderPlugin(PluginBase):
    def distDecode(self, decoder, rawF, e, h_hat, alpha, dist):
        yIndex = np.argmax(dist.npvalue())

        if decoder.int2y[yIndex] != "<UNK>":
            return None
        else:
            copyIndex = np.argmax(alpha.npvalue())
            return rawF[copyIndex]
