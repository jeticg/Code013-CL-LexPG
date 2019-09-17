# -*- coding: utf-8 -*-
import os
import re
import random
import progressbar
import dynet as dy
import numpy as np

from support.logger import logging
from natlang.exporter import RealtimeExporter
from natlang.format.tree import Node
logger = logging.getLogger('EVALUATE')


def calcScore(reference, output):
    outputFile = os.path.dirname(__file__) + '/.output.tmp'
    refFile = os.path.dirname(__file__) + '/.gold.tmp'

    with open(outputFile, 'w') as file:
        for i, line in enumerate(output):
            file.write('{}\n'.format(line.strip()))

    with open(refFile, 'w') as file:
        for i, line in enumerate(reference):
            file.write('{}\n'.format(line.strip()))

    result = {}
    result["BLEU"] = calcBLEUFromFile(refFile, outputFile)
    result["Accuracy"] = calcACC(reference, output)
    return result


def calcBLEUFromFile(refFile, outputFile):
    os.chdir(os.path.dirname(__file__))
    pathBLEU = outputFile + '.evalBLEU'
    os.system('perl support/multi-bleu-detok.perl {} < {} > {}'.format(
        refFile, outputFile, pathBLEU))
    with open(pathBLEU, 'r') as f:
        lines = f.readlines()

    if len(lines) > 0:
        var = re.search(r'BLEU\s+=\s+(.+?),', lines[0])
        bleu = var.group(1)
    else:
        bleu = 0

    return float(bleu)


def calcACC(reference, output):
    def proc(code):
        code = code.strip()
        if '//' in code:
            code = code.split("//")[0].strip()
        code = code.replace('\n', ' \\n ')
        code = code.replace('False', '0')
        code = code.replace('True', '1')
        return code.strip()
    count = 0.0
    matched = 0.0
    for f, e in zip(output, reference):
        count += 1.0
        if proc(f) == proc(e):
            matched += 1.0
    if count == 0.0:
        return None
    return matched / count * 100.0


def translate(encoder, decoder, f, beamSize=None):
    dy.renew_cg()
    hx = encoder.encode(f)
    return decoder.decode(f, hx, beamSize=beamSize)


def trans2str(e):
    result = e
    if isinstance(result, Node):
        result = result.export().replace('\n', ' \\n ')
    elif isinstance(result, list):
        if len(result) != 0 and result[-1] == "<EOS>":
            result = result[:-1]
        result = ' '.join(result)
    elif not isinstance(result, str):
        result = ' '.join(result)
    if '//' in result:
        result = result.split("//")[0].strip()
    result = transPostProc(result)
    return result


def transPostProc(result):
    # handling BPE
    return result.replace("@@ ", '')


def evaluate(encoder, decoder, dataset,
             outputToFile=None, printResult=False):
    logger.info("Start testing")
    evalRefere = []
    evalOutput = []
    output = []

    widgets = [progressbar.Bar('>'), ' ', progressbar.ETA(),
               progressbar.FormatLabel(
               '; Total: %(value)d sents (in: %(elapsed)s)')]

    testProgress =\
        progressbar.ProgressBar(widgets=widgets,
                                maxval=len(dataset)).start()

    if outputToFile is not None:
        outFile = RealtimeExporter(outputToFile)

    totalLength = 0
    perplexity = 0
    for i, (f, eRef) in enumerate(dataset):
        strE = trans2str(translate(encoder, decoder, f))
        totalLength += len(strE)
        perplexity += decoder.perplexity
        output.append(strE)

        strERef = trans2str(eRef)
        strF = ' '.join(f)

        if printResult:
            logger.info('f: {}'.format(strF))
            logger.info('eRef: {}'.format(strERef))
            logger.info('eOut: {}'.format(strE))

        evalOutput.append(strE)
        evalRefere.append(strERef)
        testProgress.update(i + 1)
        if outputToFile is not None:
            if isinstance(output[-1], Node):
                outFile.write(output[-1].export())
            else:
                outFile.write(output[-1])

    if outputToFile is not None:
        outFile = None

    result = calcScore(evalRefere, evalOutput)
    result['Perplexity'] = np.exp(-perplexity / totalLength)
    for plugin in decoder.plugins:
        for attr in plugin.__dict__:
            if attr[:4] == 'LexN' and (
                    isinstance(plugin.__dict__[attr], float) or
                    isinstance(plugin.__dict__[attr], int)):
                result[attr] = plugin.__dict__[attr]
    for key in result:
        logger.info('{} score: {}'.format(key, result[key]))
    logger.info('Finished testing')
    return
