# -*- coding: utf-8 -*-
import time
import random
import progressbar
import numpy as np
import dynet as dy

from models.ModelBase import saveModel
from support.logger import logging
from natlang.format.astTree import AstNode


def trainModel(pc, encoder, decoder, dataset, epochs,
               batchSize, trainerName='AdamTrainer',
               warmup=False,
               convergence=-1,
               maxTrainLen=50, validationDataset=None, maxValLossPatience=5,
               maxValLossIncrease=-1,
               savePath=None, dropout=None):
    logger = logging.getLogger('TRAIN')
    random.seed(17)
    logger.info("Training settings:")
    logger.info("--Ignoring target sentences longer than {} words".format(
        maxTrainLen))
    logger.info("--Trainer: {}".format(trainerName))
    logger.info("--Convergence parameter: {}".format(convergence))
    logger.info("--Maximum Epoch: {}".format(epochs))
    logger.info("--Randomising batchsize: {}".format(batchSize))
    if dropout is not None and (dropout <= 0 or dropout >= 1):
        dropout = None
    logger.info("--Output dropout: {}".format(dropout))
    if validationDataset is not None:
        validationDataset = decoder.convertDataset(validationDataset)
        validationDataset = encoder.convertDataset(validationDataset)
        logger.info("--Validation size: {}".format(len(validationDataset)))
        if maxValLossPatience > 0:
            logger.info("--Max(#epoch) validation loss patience: {}".format(
                maxValLossPatience))
        elif maxValLossIncrease > 0:
            logger.info("--Max(#epoch) validation loss increase: {}".format(
                maxValLossIncrease))
    else:
        logger.info("--No validation dataset")
    logger.debug("Start training")

    # sort training sentences by length in descending order
    logger.debug("Removing not needed entries")
    dataset = [(f, e) for f, e in dataset
               if len(e) <= maxTrainLen and len(f) > 0 and len(e) > 0]
    logger.debug("Converting dataset")
    originalData = dataset
    dataset = decoder.convertDataset(dataset)
    dataset = encoder.convertDataset(dataset)
    logger.debug("Sorting dataset entries")
    dataset.sort(
        key=lambda t: - t[1].depth
        if isinstance(t[1], AstNode) else -len(t[1]))
    maxLen = max([len(sample[1]) for sample in dataset])  # sample[1] is e
    logger.info("Dataset size: {}".format(len(dataset)))
    logger.info("Maximum Target Length: {}".format(maxLen))
    trainOrder =\
        [x * batchSize for x in range(len(dataset) // batchSize + 1)]

    trainer = getattr(dy, trainerName)(pc)

    totalLoss = 0
    totalSent = 0
    previousEpochLoss = float('inf')

    # Progress bar init
    widgets = [progressbar.Bar('>'), ' ', progressbar.ETA(),
               progressbar.FormatLabel(
               '; Total: %(value)d sents (in: %(elapsed)s)')]
    trainProgressBar =\
        progressbar.ProgressBar(widgets=widgets,
                                maxval=epochs * len(dataset)).start()

    epoch = 0
    lastValidationLoss = float("inf")
    minValidationLoss = float("inf")
    valLossPatience = 0
    valLossIncrease = 0
    logger.debug("Starting first epoch")
    fInd = encoder.fInd
    for epoch in range(epochs):
        epochLoss = 0

        # shuffle the batch start indices in each epoch
        random.shuffle(trainOrder)
        batchesPerEpoch = len(trainOrder)
        start = time.time()

        if epoch == 0 and warmup:
            tmp = decoder.plugins
            decoder.plugins = []

        # go through batches
        for i, index in enumerate(trainOrder, start=1):
            # get batch examples
            batch = dataset[index:index + batchSize]
            # skip empty batches
            if len(batch) == 0:
                continue
            batchLoss = []
            dy.renew_cg()
            for sample in batch:
                f, e = sample[fInd], sample[1]
                hx = encoder.encodeSent(f)
                batchLoss += decoder.computeSentLoss(hx, sample,
                                                     dropout=dropout)
                totalSent += 1
                trainProgressBar.update(totalSent)
            batchLoss = dy.esum(batchLoss)
            epochLoss += batchLoss.scalar_value()
            batchLoss.backward()
            trainer.update()

        if epoch == 0 and warmup:
            decoder.plugins = tmp

        epochLoss /= len(dataset)
        epochLossChange = (previousEpochLoss - epochLoss) / epochLoss
        logger.info("Epoch #{}, avgSamploss={}".format(epoch, epochLoss) +
                    ", change={0:.2f}%".format(epochLossChange * 100))

        if validationDataset is not None:
            validationLoss = 0
            for sample in validationDataset:
                f, e = sample[fInd], sample[1]
                dy.renew_cg()
                hx = encoder.encodeSent(f)
                loss = decoder.computeSentLoss(hx, sample, dropout=dropout)
                loss = dy.esum(loss)
                validationLoss += loss.scalar_value()
            validationLoss /= len(validationDataset)
            logger.info("Validation avgSampLoss {}".format(validationLoss))

            # Process min validation loss
            if minValidationLoss < validationLoss:
                valLossPatience += 1
            else:
                valLossPatience = 0
                minValidationLoss = validationLoss
                if savePath is None or savePath == "":
                    saveModel("save_minValLoss", pc, encoder, decoder)

            # Process last validation loss
            if lastValidationLoss < validationLoss:
                valLossIncrease += 1
            else:
                valLossIncrease = 0
                lastValidationLoss = validationLoss

            # Stop condition
            if maxValLossPatience > 0:
                if valLossPatience >= maxValLossPatience:
                    break
            elif maxValLossIncrease > 0:
                if valLossIncrease >= maxValLossIncrease:
                    break

        if epochLossChange < convergence:
            break
        previousEpochLoss = epochLoss
        start = time.time()
        if savePath is not None and savePath != "":
            saveModel("{}_E{}".format(savePath, epoch),
                      pc, encoder, decoder)

    # update progress bar after completing training
    trainProgressBar.finish()
    logger.debug("Exiting final epoch")
    if savePath is not None and savePath != "":
        saveModel("{}_Final".format(savePath), pc, encoder, decoder)
    logger.info('Final epoch: #{}, loss: {}'.format(epoch, epochLoss))
    logger.info('Finished training')

    return pc
