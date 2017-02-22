#!/usr/bin/env python
"""
This is the data preprocess for hand pose estimation.
"""

import _init_paths
import numpy as np
import scipy.io
from data.importers import NYUImporter, ICVLImporter
from data.dataset import NYUDataset, ICVLDataset
import h5py

def convertNYUSequenceToh5(Sequence, cacheDir, datasetName):
    """
    :param Sequence: orderDict
    :return: None
    """
    config = Sequence.config
    name = '{}/{}_{}.h5'.format(cacheDir, datasetName, Sequence.name)

    Dataset = NYUDataset([Sequence])
    dpt, gt3D = Dataset.imgStackDepthOnly(Sequence.name)

    depth = []
    gtorig = []
    gtcrop = []
    T = []
    gt3Dorig = []
    gt3Dcrop = []
    com = []
    fileName = []

    for i in xrange(len(Sequence.data)):
        data = Sequence.data[i]
        depth.append(data.dpt)
        gtorig.append(data.gtorig)
        gtcrop.append(data.gtcrop)
        T.append(data.T)
        gt3Dorig.append(data.gt3Dorig)
        gt3Dcrop.append(data.gt3Dcrop)
        com.append(data.com)
        fileName.append(int(data.fileName[-11:-4]))


    dataset = h5py.File(name, 'w')
    dataset['com'] = np.asarray(com)
    dataset['inds'] = np.asarray(fileName)
    dataset['config'] = config['cube']
    dataset['depth'] = np.asarray(dpt)
    dataset['joint'] = np.asarray(gt3D)
    dataset['gt3Dorig'] = np.asarray(gt3Dorig)
    dataset.close()

def convertICVLSequenceToh5():
    pass

if __name__ == '__main__':

    cacheDir = './dataset/cache'

    # create NYU dataset
    print("create NYU dataset")

    datasetName = 'NYU'
    di = NYUImporter('./dataset/' + datasetName, cacheDir=cacheDir)

    Seq1 = di.loadSequence('train')
    trainSeqs = [Seq1]

    Seq2_1 = di.loadSequence('test_1')
    Seq2_2 = di.loadSequence('test_2')
    testSeqs = [Seq2_1, Seq2_2]

    convertNYUSequenceToh5(Seq1, cacheDir, datasetName)
    print("Seq1 ok!")
    convertNYUSequenceToh5(Seq2_1, cacheDir, datasetName)
    print("Seq2_1 ok!")
    convertNYUSequenceToh5(Seq2_2, cacheDir, datasetName)
    print("Seq2_2 ok!")

    # create ICVL dataset
    print("create ICVL dataset")

    datasetName = 'ICVL'
    di = ICVLImporter('./dataset' + datasetName, cacheDir=cacheDir)
    Seq1 = di.loadSequence('train', ['0'])
    trainSeqs = [Seq1]

    Seq2 = di.loadSequence('test_seq_1')
    testSeqs = [Seq2]
