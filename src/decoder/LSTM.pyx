# -*- coding: utf-8 -*-
# LSTM Decoder with simple attention and plugin support
import dynet as dy
import numpy as np

from support.logger import logging
from models.ModelBase import ModelBase
logger = logging.getLogger('DECODER')


class Decoder(ModelBase):
    def __init__(self, maxTestLen=100):
        ModelBase.__init__(self)
        self.maxTestLen = maxTestLen
        self.y2int = {}
        self.int2y = []
        self.yIntFreq = []
        self.plugins = []
        self.component = ["inDim", "hidDim", "layers",
                          "y2int", "int2y", "yIntFreq",
                          "plugins"]
        return

    def buildLexicon(self, dataset, lexiconSize=50000, minFrequency=5):
        index = 1
        logger.info("Decoder lexiconY minFreq: {}".format(minFrequency))
        self.y2int, self.int2y = ModelBase.buildLexicon(
            self, dataset, index=index, lexiconSize=lexiconSize,
            minFrequency=minFrequency)
        for key in ["<SOS>", "<EOS>"]:
            self.y2int[key] = len(self.int2y)
            self.int2y.append(key)
        for plugin in self.plugins:
            plugin.buildLexicon(self, dataset, lexiconSize=lexiconSize)
        logger.info("Decoder lexicon size: {}".format(len(self.int2y)))
        return

    def convertDataset(self, dataset):
        index = 1

        result = dataset
        for plugin in self.plugins:
            result = plugin.convertDataset(self, result)

        for i in range(len(result)):
            sample = result[i]
            newE = [item for item in sample[index]] + ["<EOS>"]
            result[i] = sample[:index] + (newE, ) + sample[index + 1:]

        result = ModelBase.convertDataset(
            self, result, index=index, w2int=self.y2int, int2w=self.int2y)
        return result

    def buildModel(self, pc, inDim, hidDim, layers):
        for plugin in self.plugins:
            plugin.buildModel(self, pc, inDim, hidDim, layers)
        logger.info("inDim={}, hidDim={}".format(inDim, hidDim))
        self.inDim = inDim
        self.hidDim = hidDim
        self.layers = layers
        outputSize = len(self.int2y)
        # Lookups
        self.initLookup = pc.add_lookup_parameters((1, 3 * hidDim))
        self.yLookup = pc.add_lookup_parameters((outputSize, inDim))

        # Softmax output
        self.readout =\
            pc.add_parameters((outputSize, 3 * hidDim))
        self.bias = pc.add_parameters(outputSize)

        # LSTMs
        self.decoder = dy.LSTMBuilder(
            layers, 3 * hidDim + inDim, hidDim, pc)

        # attention
        self.w_c = pc.add_parameters((3 * hidDim, 3 * hidDim))
        self.w_a = pc.add_parameters((hidDim, hidDim))
        self.u_a = pc.add_parameters((hidDim, 2 * hidDim))
        self.v_a = pc.add_parameters((1, hidDim))
        return

    def computeSentLoss(self, hx, sample, dropout=0.5):
        self.hx = hx
        f, e = sample[:2]

        # Initialise sentLoss
        sentLoss = []
        for plugin in self.plugins:
            loss = plugin.computeSentLoss(self, hx, sample)
            if loss is None:
                continue
            if isinstance(loss, list):
                sentLoss += loss
            else:
                sentLoss.append(loss)

        # initial vectors to feed decoder - 3*h, its result should be <s>
        dist = None
        h_hat = self.initLookup[0]
        initLookup = self.yLookup[self.y2int["<SOS>"]]
        feedback = dy.concatenate([initLookup, h_hat])
        decoderState = self.decoder.initial_state()

        # run the decoder through the output sequences and aggregate loss
        for i, word in enumerate(e):
            for plugin in self.plugins:
                pluginFeedback = plugin.feedback(
                    self, i-1, h_hat, dist, feedback, training=True)
                if pluginFeedback is not None:
                    feedback = pluginFeedback
            decoderState = decoderState.add_input(feedback)
            # returns h x batchSize matrix
            hy = decoderState.output()
            self.hy = hy

            # compute attention context vector for each sequence in the batch
            # (returns 2h x batchSize matrix)
            h_hat, alpha = self.attention(hx, hy)
            self.h_hat = h_hat

            # compute output
            # h = readout * h_hat + bias
            dist = dy.softmax(self.bias + self.readout * h_hat)

            # compute loss
            if dropout is not None:
                dy.dropout(dist, dropout)
            for plugin in self.plugins:
                loss = plugin.distLoss(self,
                                       i, sample, h_hat, alpha, dist, word, hx)
                if loss is None:
                    continue
                if isinstance(loss, list):
                    sentLoss += loss
                else:
                    sentLoss.append(loss)
            sentLoss.append(-dy.log(dist[word]))

            # Use gold result for feedback during training(computeLoss==True)
            feedback = dy.lookup(self.yLookup, word)
            feedback = dy.concatenate([feedback, h_hat])

        return [loss for loss in sentLoss if loss is not None]

    def attention(self, hx, hy):
        # hx dimension: seqLen x 2h x batchSize
        # hy dimension: h x batchSize
        W, U, V = self.w_a, self.u_a, self.v_a

        if len(hx) == 1:
            # no need to attend if sequence length is 1
            h_hat = dy.tanh(self.w_c * dy.concatenate([hy, hx[0]]))
            return h_hat, dy.zeros((1,)) + 1

        # Calcuate attention scores
        # scores = [V * tanh(W * hy + U * hx_i) for hx_i in hx]
        Why = W * hy
        scores =\
            [V * dy.tanh(Why + U * hx_i) for hx_i in hx]
        alpha = dy.softmax(dy.concatenate(scores))
        # compute context vector with weighted sum for each seq in batch
        c = dy.concatenate_cols(hx) * alpha
        # compute output vector using current decoder state and context vector
        h_hat = dy.tanh(self.w_c * dy.concatenate([c, hy]))
        return h_hat, alpha

    def decodeGreedy(self, rawF, hx):
        self.hx = hx
        for plugin in self.plugins:
            plugin.refresh(self, rawF, hx)

        output = []
        i = 0
        self.perplexity = 0

        # initialize the decoder rnn
        dist = None
        h_hat = self.initLookup[0]
        initLookup = self.yLookup[self.y2int["<SOS>"]]
        feedback = dy.concatenate([initLookup, h_hat])
        decoderState = self.decoder.initial_state()

        # run the decoder through the sequence and predict output symbols
        while len(output) < self.maxTestLen and (len(output) == 0 or
                                                 output[-1] != "<EOS>"):
            for plugin in self.plugins:
                pluginFeedback = plugin.feedback(
                    self, len(output)-1, h_hat, dist, feedback)
                if pluginFeedback is not None:
                    feedback = pluginFeedback
            # get current h of the decoder
            decoderState = decoderState.add_input(feedback)
            hy = decoderState.output()
            self.hy = hy

            # perform attention step
            h_hat, alpha = self.attention(hx, hy)
            self.h_hat = h_hat

            # compute output probabilities
            # h = readout * h_hat + bias
            dist = dy.softmax(self.bias + self.readout * h_hat)

            # find best candidate output - greedy
            yIndex = np.argmax(dist.npvalue())
            rawOut = self.int2y[yIndex]
            for plugin in self.plugins:
                pluginOut =\
                    plugin.distDecode(self, rawF, output, h_hat, alpha, dist)
                if pluginOut is not None:
                    rawOut = pluginOut

            output.append(rawOut)
            if rawOut in self.y2int:
                yIndex = self.y2int[rawOut]
            if output[-1] != "<EOS>":
                self.perplexity += np.log(dist.npvalue()[yIndex])

            # prepare for the next iteration - "feedback"
            feedback = dy.concatenate([self.yLookup[yIndex], h_hat])
            i += 1

        # remove the end seq symbol
        return output[:-1]

    def decode(self, rawF, hx, beamSize=None):
        self.hx = hx
        if beamSize is None:
            if hasattr(self, "beamSize") and self.beamSize != 0:
                beamSize = self.beamSize
            else:
                return self.decodeGreedy(rawF, hx)

        for plugin in self.plugins:
            plugin.refresh(self, rawF, hx)

        alphas_mtx = []

        # complete sequences and their probabilities
        finalPredictions = []
        # initialize the decoder rnn
        dist = None
        h_hat = self.initLookup[0]
        initLookup = self.yLookup[self.y2int["<SOS>"]]
        feedback = dy.concatenate([initLookup, h_hat])
        for plugin in self.plugins:
            pluginFeedback = plugin.feedback(
                self, -1, h_hat, dist, feedback, training=True)
            if pluginFeedback is not None:
                feedback = pluginFeedback
        decoderState = self.decoder.initial_state()

        # holds beam step index mapped to
        # (output, prob, prevState, feedback)
        # tuples
        beam = {-1: [([], 1.0, decoderState, feedback)]}
        i = 0

        # expand another step if didn't reach max length and there's still
        # beams to expand
        while i < self.maxTestLen and len(beam[i - 1]) > 0:

            # create all expansions from the previous beam:
            newBeam = []
            for output, prob, decoderState, feedback in beam[i - 1]:
                decoderState = decoderState.add_input(feedback)
                hy = decoderState.output()
                self.hy = hy

                # perform attention step
                h_hat, alpha = self.attention(hx, hy)
                self.h_hat = h_hat

                # compute output probabilities
                # h = readout * h_hat + bias
                dist = dy.softmax(self.bias + self.readout * h_hat)
                yIndices = self.argmax(dist.npvalue(), beamSize)

                for yIndex in yIndices:
                    rawOut = self.int2y[yIndex]
                    p = dist.npvalue()[yIndex]
                    for plugin in self.plugins:
                        pluginOut =\
                            plugin.distDecode(
                                self, rawF, output, h_hat, alpha, dist)
                        if pluginOut is not None:
                            rawOut = pluginOut

                    output = output + [rawOut]
                    if rawOut in self.y2int:
                        yIndex = self.y2int[rawOut]
                    else:
                        yIndex = self.y2int['<UNK>']
                    p = dist.npvalue()[yIndex]
                    prob = prob * p

                    feedback = dy.concatenate([self.yLookup[yIndex], h_hat])
                    for plugin in self.plugins:
                        pluginFeedback = plugin.feedback(
                            self, len(output)-1, h_hat, dist, feedback)
                        if pluginFeedback is not None:
                            feedback = pluginFeedback

                    if not len(output) == 1 and\
                            (i == self.maxTestLen - 1 or
                             output[-1] == "<EOS>"):
                        finalPredictions.append((output[:-1], prob))
                    else:
                        newBeam.append(
                            (output, prob, decoderState, feedback))

            # add the most probable expansions from all hypotheses to the beam
            nextProbs = np.array([p for (s, p, r, a) in newBeam])
            indices = self.argmax(nextProbs, beamSize)
            beam[i] = [newBeam[l] for l in indices]
            i += 1

        # get nbest results from final states found in search
        probs = np.array([p for (s, p) in finalPredictions])
        index = np.argmax(probs)
        output = finalPredictions[index][0]
        self.perplexity = np.log(finalPredictions[index][1])
        return output
