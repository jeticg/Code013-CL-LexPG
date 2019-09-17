# -*- coding: utf-8 -*-
import os
import errno
import gzip
import dynet as dy
import pickle as pickle

from support.logger import logging
from natlang.format.astTree import AstNode
logger = logging.getLogger("MODEL")


class ModelBase:
    def __init__(self):
        self.inDim = 0
        self.hidDim = 0
        self.layers = 0
        self.component = []
        return

    def loadModel(self, fileName=None, force=False):
        fileName = os.path.expanduser(fileName)
        if fileName.endswith("pklz"):
            pklFile = gzip.open(fileName, 'rb')
        else:
            pklFile = open(fileName, 'rb')

        entity = vars(self)
        # load components
        for componentName in self.component:
            if componentName not in entity:
                raise RuntimeError("object " + componentName +
                                   " doesn't exist in this class")
            entity[componentName] = self.__loadObjectFromFile(pklFile)

        pklFile.close()
        return

    def buildLexicon(self, dataset, index, lexiconSize=50000, minFrequency=0):
        w2int = {}
        int2w = []
        for sample in dataset:
            for word in sample[index]:
                if word not in w2int:
                    w2int[word] = 0
                w2int[word] += 1

        return _lexCreate(w2int, lexiconSize, minFrequency, ["<UNK>"])

    def buildNodeLexicon(self, dataset, index, lexiconSize, minFrequency):
        w2int = {}
        t2int = {}
        l2int = {}
        for i, sample in enumerate(dataset):
            x = sample[index].astTree
            if not isinstance(x, AstNode):
                raise TypeError("Error target input type: target must be" +
                                " AstNode instance\n" +
                                "Item #" + str(i) + " : " + str(x))
            _, _, valCol, _, _ = x.columnFormat()
            for entry in valCol:
                if len(entry) == 1:
                    label = entry[0]
                    if label not in l2int:
                        l2int[label] = 0
                    l2int[label] += 1
                else:
                    pos, word = entry
                    if word not in w2int:
                        w2int[word] = 0
                    w2int[word] += 1
                    if pos not in t2int:
                        t2int[pos] = 0
                    t2int[pos] += 1
            continue

        w2int, int2w = _lexCreate(w2int, lexiconSize, minFrequency, ["<UNK>"])
        t2int, int2t = _lexCreate(t2int, lexiconSize, 0, ["<UNK>"])
        l2int, int2l = _lexCreate(l2int, lexiconSize, 0, ["NULL",
                                                          "<UNK>"])
        return w2int, int2w, t2int, int2t, l2int, int2l

    def convertDataset(self, dataset, index, w2int, int2w):
        result = []
        for sample in dataset:
            item = sample[index]
            newItem = []
            for word in item:
                if word in w2int:
                    newItem.append(w2int[word])
                else:
                    newItem.append(w2int["<UNK>"])

            result.append(sample[:index] + (newItem, ) + sample[index + 1:])
        return result

    def saveModel(self, fileName=""):
        if fileName == "":
            logger.warning("Destination not specified, components will not" +
                           " be saved")
            return
        entity = vars(self)
        if fileName.endswith("pklz"):
            output = gzip.open(fileName, 'wb')
        elif fileName.endswith("pkl"):
            output = open(fileName, 'wb')
        else:
            fileName = fileName + ".pkl"
            output = open(fileName, 'wb')

        # dump components
        for componentName in self.component:
            if componentName not in entity:
                raise RuntimeError("object in _savedModelFile doesn't exist")
            self.__saveObjectToFile(entity[componentName], output)

        output.close()
        return

    def __loadObjectFromFile(self, pklFile):
        a = pickle.load(pklFile)
        return a

    def __saveObjectToFile(self, a, output):
        pickle.dump(a, output)
        return

    def argmax(self, x, beamSize=1):
        result = sorted(enumerate(x), key=lambda t: -t[1])[:beamSize]
        return [entry[0] for entry in result]


def _lexCreate(w2int, lexiconSize, minFrequency, extra):
    int2w = []
    lex = [w for w in list(w2int.items()) if w[1] >= minFrequency]
    lex = sorted(lex, key=lambda w: w[1])[-lexiconSize:]
    w2int = dict(lex)
    for i, w in enumerate(extra):
        w2int[w] = i
    for key in w2int:
        w2int[key] = len(int2w)
        int2w.append(key)
    return w2int, int2w


def saveModel(path, pc, encoder, decoder):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            logger.warning("{} already exist, will be overwritten".format(
                path))
    encoder.saveModel(path + "/encoder.pkl")
    decoder.saveModel(path + "/decoder.pkl")
    pc.save(path + "/params.dynet")
    logger.debug("Model saved to " + str(path))
    return


def loadModel(path, pc, encoder, decoder):
    logger.info("Loading model from " + str(path))
    encoder.loadModel(path + "/encoder.pkl")
    decoder.loadModel(path + "/decoder.pkl")
    encoder.buildModel(
        pc, encoder.inDim, encoder.hidDim,
        encoder.layers)
    decoder.buildModel(
        pc, decoder.inDim, decoder.hidDim,
        decoder.layers)
    pc.populate(path + "/params.dynet")
    logger.debug("Loaded")
    return
