# -*- coding: utf-8 -*-
import argparse
import os
import sys
import importlib
import configparser
import ast
from configparser import SafeConfigParser

from support.logger import logging
from natlang.loader import ParallelDataLoader
__logger = None


class EnvInterpolation(configparser.BasicInterpolation):
    """Interpolation which expands environment variables in values."""

    def before_get(self, parser, section, option, value, defaults):
        return os.path.expandvars(value)


def processExtraOptions(option):
    if isinstance(option, str):
        if '{' in option and '}' in option:
            option = ast.literal_eval(option)
        else:
            option = option.split('=')
            if len(option) == 1:
                option = {option[0]: True}
            elif len(option) == 2:
                option = dict([option])
            else:
                raise ValueError(
                    "natlang.dataLoader.load: invalid option")
    if option is None:
        option = {}
    if not isinstance(option, dict):
        raise ValueError(
            "support.configHandler.processExtraOptions: invalid option")
    return option


def processConfig(version):
    """
    Process arguments and config file

    @param version: str, version info
    @return: config, a list of configurations
    """
    global __logger
    __logger = logging.getLogger('MAIN')
    config = {
        'demo': False,
        'plot': False,
        'beamSize': 0,

        'dataDir': '',
        'sourceLanguage': '',
        'targetLanguage': '',
        'srcLoader': 'txtOrTree',
        'tgtLoader': 'txtOrTree',
        'loaderOption': '{}',

        'trainData': '',
        'validationData': '',
        'testData': '',

        'trainSize': sys.maxsize,
        'testSize': sys.maxsize,
        'lexiconSize': 50000,
        'srcLexBar': 3,
        'tgtLexBar': 5,
        'batchSize': 64,
        'epochs': 20,
        'dropout': 0,

        'plugins': '',
        'pluginLex': '',
        'disablePlugin': False,
        'forcePlugin': False,
        'warmup': False,

        'encoder': "BiLSTM",
        'decoder': "LSTM",
        'encoderOption': "{}",
        'decoderOption': "{}",
        'output': 'output.txt',
        'trainer': 'AdamTrainer',
        'maxLen': sys.maxsize,
        'maxValLossPatience': 5,
        'attention': True,
        'alpha': 1.0,

        'inputDim': 256,
        'hiddenDim': 256,
        'layer': 2,

        'loadModel': "",
        'saveModel': "save",
        'forceLoad': False
    }

    configGeneralSection = {
        'DataDirectory': 'dataDir',
        'TargetLanguageSuffix': 'targetLanguage',
        'SourceLanguageSuffix': 'sourceLanguage',
        'SourceFormatLoader': 'srcLoader',
        'TargetFormatLoader': 'tgtLoader',
        'loaderOption': 'loaderOption',
        'SourceLexiconBar': 'srcLexBar',
        'TargetLexiconBar': 'tgtLexBar',
        'MaxLexicon': 'lexiconSize',
    }

    configDataSection = {
        'TrainFilePrefix': 'trainData',
        'ValidateFilePrefix': 'validationData',
        'TestFilePrefix': 'testData',
        'dataLoader': 'loadTreeDataset',
        'LexFiles': 'pluginLex',
    }

    configFileModelSection = {
        "HiddenDimension": "hiddenDim",
        "InputDimension": "inputDim",
        "Layer": "layer",
        "batchSize": "batchSize",
        "maxLength": "maxLen",
    }

    ap = argparse.ArgumentParser(
        description="""Jetic's NMT system %s""" % version)
    ap.add_argument('--dynet-mem')
    ap.add_argument('--dynet-gpu')
    ap.add_argument('--dynet-autobatch')
    ap.add_argument('--dynet-gpus')
    ap.add_argument('--dynet-devices')
    ap.add_argument(
        "--demo", dest="demo", action='store_true',
        help="demonstration mode")
    ap.add_argument(
        "--disablePlugin", dest="disablePlugin", action='store_true',
        help="disable plugins during testing"
    )
    ap.add_argument(
        "--forcePlugin", dest="forcePlugin", action='store_true',
        help="replace original plugin with the newly specified after loading"
    )
    ap.add_argument(
        "--maxLen", dest="maxLen", type=int,
        help="Maximum training length")
    ap.add_argument(
        "--plot", dest="plot", action='store_true',
        help="plot attention graph in demonstration mode")
    ap.add_argument(
        "--lexiconSize", dest="lexiconSize", type=int,
        help="maximum lexicon size")
    ap.add_argument(
        "--beamSize", dest="beamSize", type=int,
        help="Enable beam-search by specifying the beam size")
    ap.add_argument(
        '--srcLexBar', dest="srcLexBar", type=int,
        help="Minimum occurrence requirement to be included in the lexicon")
    ap.add_argument(
        '--tgtLexBar', dest="tgtLexBar", type=int,
        help="Minimum occurrence requirement to be included in the lexicon")
    ap.add_argument(
        "--alpha", dest="alpha", type=float,
        help="alpha hyperparameter in DRNN(LM)")
    ap.add_argument(
        "--plugins", dest="plugins",
        help="plugins to use")
    ap.add_argument(
        "-d", "--datadir", dest="dataDir",
        help="data directory")
    ap.add_argument(
        "--train", dest="trainData",
        help="prefix of training data file")
    ap.add_argument(
        "--test", dest="testData",
        help="prefix of testing data file")
    ap.add_argument(
        "--source", dest="sourceLanguage",
        help="suffix of source language")
    ap.add_argument(
        "--target", dest="targetLanguage",
        help="suffix of target language")
    ap.add_argument(
        "--srcLoader", dest="srcLoader",
        help="Source language file format loader name"),
    ap.add_argument(
        "--tgtLoader", dest="tgtLoader",
        help="Target language file format loader name"),
    ap.add_argument(
        "--loaderOption", dest="loaderOption",
        help="Extra parameters passed to the loaders")
    ap.add_argument(
        "--lexFiles", dest="pluginLex",
        help="Lexicons (LexNX series) for the plugins")
    ap.add_argument(
        "-n", "--trainSize", dest="trainSize", type=int,
        help="Number of sentences to use for training")
    ap.add_argument(
        "-v", "--testSize", dest="testSize", type=int,
        help="Number of sentences to use for testing")
    ap.add_argument(
        "--batchsize", dest="batchSize", type=int,
        help="Batch Size")
    ap.add_argument(
        "--dropout", dest="dropout", type=float,
        help="Dropout(output), float")
    ap.add_argument(
        "--disable-attention", dest="attention", action='store_false',
        help="Disable attention")
    ap.add_argument(
        "--warmup", dest="warmup", action='store_true',
        help="""Initialise the baseline model before training with plugins.
                This can sometimes prevent exploding/vanishing gradient, but
                might require plugin support""")
    ap.add_argument(
        "-i", "--epochs", dest="epochs", type=int,
        help="Number of epochs to train")
    ap.add_argument(
        "--encoder", dest="encoder",
        help="encoder to use, default is BiLSTM")
    ap.add_argument(
        "--encoderOption", dest="encoderOption",
        help="Extra parameters passed to the encoder")
    ap.add_argument(
        "--decoder", dest="decoder",
        help="decoder to use, default is LSTM")
    ap.add_argument(
        "--decoderOption", dest="decoderOption",
        help="Extra parameters passed to the decoder")
    ap.add_argument(
        "--trainer", dest="trainer",
        help="trainer to use, default is AdamTrainer")
    ap.add_argument(
        "--maxValLossPatience", dest="maxValLossPatience", type=int,
        help="maxValLossPatience to use, default is 5")
    ap.add_argument(
        "--hiddenDim", dest="hiddenDim", type=int,
        help="size of hidden dimension")
    ap.add_argument(
        "--inputDim", dest="inputDim", type=int,
        help="size of input dimension")
    ap.add_argument(
        "--layer", dest="layer", type=int,
        help="number of layers for neural network")
    ap.add_argument(
        "-o", "--outputToFile", dest="output",
        help="Path to output file")
    ap.add_argument(
        "-c", "--config", dest="config",
        help="Path to config file")
    ap.add_argument(
        "-s", "--saveModel", dest="saveModel",
        help="Where to save the model")
    ap.add_argument(
        "-l", "--loadModel", dest="loadModel",
        help="Specify the model file to load")

    args = ap.parse_args()

    # Process config file
    if args.config:
        # Check local config path
        if not os.path.isfile(args.config):
            __logger.error("The config file doesn't exist: %s\n" % args.config)
            sys.exit(1)

        # Initialise the config parser
        __logger.info("Reading configurations from file: %s" % (args.config))
        cp = SafeConfigParser(interpolation=EnvInterpolation())
        cp.read(args.config)

        # Process the contents of config file
        for key in configGeneralSection:
            try:
                if cp.get('General', key) != '':
                    config[configGeneralSection[key]] = cp.get('General', key)
            except configparser.NoOptionError:
                pass

        for key in configDataSection:
            try:
                if cp.get('Data', key) != '':
                    config[configDataSection[key]] = cp.get('Data', key)
            except configparser.NoOptionError:
                pass

        for key in configFileModelSection:
            try:
                if cp.get('Model', key) != '':
                    config[configFileModelSection[key]] =\
                        cp.getint('Model', key)
            except configparser.NoOptionError:
                pass

    # Reset default values to config file
    ap.set_defaults(**config)
    args = ap.parse_args()
    config.update(vars(args))
    config["encoderOption"] = processExtraOptions(config["encoderOption"])
    config["decoderOption"] = processExtraOptions(config["decoderOption"])
    if config['plugins']:
        config['plugins'] = config['plugins'].strip().split(',')
    else:
        config['plugins'] = []
    if config['pluginLex']:
        config['pluginLex'] = config['pluginLex'].strip().split(',')
        config['pluginLex'] = [config['dataDir'] + '/' + entry
                               for entry in config['pluginLex']]
    else:
        config['pluginLex'] = []
    return config


def configLoadDataset(config, key='testData', linesToLoad=sys.maxsize):
    if config[key] != "":
        sourceFile = os.path.expanduser(
            "%s.%s" % (os.path.join(config['dataDir'], config[key]),
                       config['sourceLanguage']))
        targetFile = os.path.expanduser(
            "%s.%s" % (os.path.join(config['dataDir'], config[key]),
                       config['targetLanguage']))

        __logger.debug("Loading " + key +
                       " src/tgt file using " +
                       config['srcLoader'] + "/" + config['tgtLoader'])
        loader = ParallelDataLoader(config['srcLoader'], config['tgtLoader'])
        dataset = loader.load(sourceFile, targetFile, linesToLoad,
                              option=config["loaderOption"])
        __logger.debug("Loaded, valid entries: " + str(len(dataset)))
        return dataset
    return None
