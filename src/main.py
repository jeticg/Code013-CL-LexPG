# -*- coding: utf-8 -*-
# Python version: 2/3
#
# Jetic's NMT Experiment Main Programme
# Simon Fraser University
# Jetic Gu
#
#
import sys
import importlib
import inspect
import dynet as dy

from support.logger import logging, initialiseLogger, getCommitHash
import support.configHandler as configHandler

from encoder.BiLSTM import Encoder
from models.ModelBase import loadModel
from trainer import trainModel
import evaluator
from evaluator import trans2str
__version__ = "v0.7a"


def translate(input):
    f = input.strip().split()
    return evaluator.translate(encoder, decoder, f)


def demoMode():
    def onScreen(output):
        if output is None:
            print("[WARN] No previous output available")
        else:
            print(output)
        return

    __logger.debug("Running in demo mode. Enter sentences for translation")
    __logger.debug("Programme terminates upon receiving <EOF>")
    __logger.debug("View structured representation of previous output using " +
                   "'onscreen' command")
    if config["plot"] is True:
        if "plot" not in vars(decoder):
            __logger.warning("Current decoder does not support plotting!")
            config["plot"] = False
        else:
            decoder.plot = True
            from support.plot import plotAlignmentWithScore, showPlot
            import numpy as np
    output = None
    outputStr = None
    while True:
        try:
            line = input("> ")
        except (EOFError):
            break
        if line == "onscreen":
            onScreen(output)
            continue

        __logger.info("Source:\t" + line)
        output = translate(line)
        outputStr = evaluator.trans2str(output)
        __logger.info("Target:\t" + outputStr)
    return


if __name__ == "__main__":
    # Loading options here, the main programme reads the config file and parses
    # command line options
    initialiseLogger('main.log')
    __logger = logging.getLogger('MAIN')
    __logger.debug("""Code013 CL-LexPG %s""" % __version__)
    __logger.debug("""Based on Projekt013, Projekt003, Projekt005""")
    __logger.debug("--Commit#: {}".format(getCommitHash()))
    config = configHandler.processConfig(__version__)
    trainDataset = configHandler.configLoadDataset(
        config, 'trainData', config['trainSize'])
    valDataset = configHandler.configLoadDataset(
        config, 'validationData', config['trainSize'])
    testDataset = configHandler.configLoadDataset(
        config, 'testData', config['testSize'])
    pc = dy.ParameterCollection()

    __logger.info("Loading encoder: " + config['encoder'] +
                  "; decoder: " + config['decoder'])
    if len(config['encoderOption']) != 0:
        __logger.info("Encode options: " + str(config['encoderOption']))
    if len(config['decoderOption']) != 0:
        __logger.info("decode options: " + str(config['decoderOption']))

    # Loading models, that includes the encoder and decoder.
    Encoder = importlib.import_module("encoder." + config['encoder']).Encoder
    encoder = Encoder()
    encoder.option = config["encoderOption"]
    Decoder = importlib.import_module("decoder." + config['decoder']).Decoder
    decoder = Decoder()
    decoder.option = config["decoderOption"]
    decoder.alpha = config["alpha"]
    if len(config['plugins']) != 0:
        if "plugins" not in decoder.component:
            __logger.info("Selected decoder does not support plugins")
        for plugin in config['plugins']:
            __logger.info("decoder plugin: " + plugin)
            DecoderPlugin = importlib.import_module(
                "decoder.plugins." + plugin).DecoderPlugin
            pluginInitParams = inspect.getargspec(DecoderPlugin.__init__)[0]
            if "lexFiles" in pluginInitParams:
                __logger.debug("Injecting pluginLex option into plugin: " +
                               plugin)
                decoder.plugins.append(
                    DecoderPlugin(decoder, lexFiles=config["pluginLex"]))
            else:
                decoder.plugins.append(DecoderPlugin(decoder))
    if not config["attention"]:
        decoder.atten = False
    if config["beamSize"] != 0:
        decoder.beamSize = config["beamSize"]
        __logger.info("Beam Search enabled, size: " + str(config["beamSize"]))

    # Load pretrained model
    # if not specified, initialise the model randomly
    if config["loadModel"] != "":
        loadModel(config["loadModel"], pc, encoder, decoder)
        if config['forcePlugin'] is True:
            __logger.warning('forcePlugin option is enabled, replacing plugin')
            decoder.plugins = []
            for plugin in config['plugins']:
                __logger.info("decoder plugin: " + plugin)
                DecoderPlugin = importlib.import_module(
                    "decoder.plugins." + plugin).DecoderPlugin
                pluginInitParams =\
                    inspect.getargspec(DecoderPlugin.__init__)[0]
                if "lexFiles" in pluginInitParams:
                    __logger.debug("Injecting pluginLex option into plugin: " +
                                   plugin)
                    decoder.plugins.append(
                        DecoderPlugin(decoder, lexFiles=config["pluginLex"]))
                else:
                    decoder.plugins.append(DecoderPlugin(decoder))
    else:
        if trainDataset is None:
            __logger.warning("No selected trainDataset and savedModel")
            __logger.debug("Terminating")
            exit()
        __logger.info("Building Model with trainDataset")
        __logger.info("Maximum lexiconSize: " + str(config["lexiconSize"]))
        encoder.buildLexicon(trainDataset,
                             lexiconSize=config["lexiconSize"],
                             minFrequency=config["srcLexBar"])
        decoder.buildLexicon(trainDataset,
                             lexiconSize=config["lexiconSize"],
                             minFrequency=config["tgtLexBar"])
        encoder.buildModel(
            pc, config["inputDim"], config["hiddenDim"],
            config["layer"])
        decoder.buildModel(
            pc, config["inputDim"], config["hiddenDim"],
            config["layer"])

    # Perform training
    if trainDataset is not None:
        __logger.debug("Training with trainDataset")
        if config["warmup"]:
            __logger.debug("""Warmup activated, first epoch of training will
                              not utilise plugins""")
        trainModel(pc, encoder, decoder, trainDataset,
                   epochs=config["epochs"],
                   batchSize=config["batchSize"],
                   validationDataset=valDataset,
                   trainerName=config["trainer"],
                   warmup=config["warmup"],
                   savePath=config["saveModel"],
                   dropout=config["dropout"],
                   maxValLossPatience=config["maxValLossPatience"],
                   maxTrainLen=config['maxLen'])
        __logger.debug("Training complete")

    # Threshold experiment
    # Interactive mode only
    def threshold(threshold):
        decoder.plugins[0].threshold = threshold
        __logger.debug("Testing with testDataset and threshold=" +
                       str(threshold))
        evaluator.evaluate(encoder, decoder, testDataset, config['output'])
        __logger.debug("Test complete")

    # Perform testing
    if testDataset is not None:
        if config['disablePlugin']:
            plugins = decoder.plugins
            decoder.plugins = []
        __logger.debug("Testing with testDataset")
        evaluator.evaluate(encoder, decoder, testDataset, config['output'])
        __logger.debug("Test complete")
        if config['disablePlugin']:
            decoder.plugins = plugins

    if config["demo"] is True:
        demoMode()
