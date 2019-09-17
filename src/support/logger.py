# -*- coding: utf-8 -*-
import logging
import subprocess


def initialiseLogger(logFile='main.log'):
    logging.basicConfig(
        level=logging.DEBUG, filename=logFile,
        format='%(asctime)s %(process)d:%(name)s [%(levelname)s]: %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter(
        '%(asctime)s %(process)d:%(name)s [%(levelname)s]: %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def getCommitHash():
    return subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'])[:-1]
