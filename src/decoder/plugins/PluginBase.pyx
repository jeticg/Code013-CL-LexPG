class PluginBase:
    def __init__(self, decoder):
        return

    def buildLexicon(self, decoder, dataset, lexiconSize=50000):
        return

    def convertDataset(self, decoder, dataset):
        return dataset

    def buildModel(self, decoder, pc, inDim, hidDim, layers):
        return

    def distLoss(self, decoder, i, sample, h_hat, alpha, dist, word, hx):
        return

    def computeSentLoss(self, decoder, hx, sample):
        return

    def distDecode(self, decoder, rawF, e, h_hat, alpha, dist):
        return

    def feedback(self, decoder, i, h_hat, dist, feedback, training=False):
        return

    def refresh(self, decoder, rawF, hx):
        return
