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
import os

class convertDatasetToh5(object):
    """base class for dataset transformation"""
    def __init__(self, cacheDir, datasetName):
        self.cacheDir = cacheDir
        self.datasetName = datasetName

class convertNYUDatasetToh5(convertDatasetToh5):
    """convert NYU dataset to h5"""
    def __init__(self, cacheDir, dataName):
        super(convertNYUDatasetToh5, self).__init__(cacheDir, dataName)

    def createSequence(self):
        """create NYU Sequence, train and test sequence"""
        print("create NYU dataset")

        di = NYUImporter('../dataset/' + self.datasetName, cacheDir=self.cacheDir)

        Seq1 = di.loadSequence('train',flip=True,rotation=True) # train sequence
        Seq2_1 = di.loadSequence('test_1') # test sequence 1
        Seq2_2 = di.loadSequence('test_2') # test sequence 2

        self.convert(Seq1)
        print("{} Train Seq1 ok!".format(self.datasetName))
        self.convert(Seq2_1)
        print("{} Test Seq1 ok!".format(self.datasetName))
        self.convert(Seq2_2)
        print("{} Test Seq2 ok!".format(self.datasetName))

    def convert(self, sequence):
        """convert NYU sequence"""
        config = sequence.config
        name = '{}/{}_{}.h5'.format(self.cacheDir, self.datasetName, sequence.name)
        if os.path.isfile(name):
            print '{} exist, please check if h5 is right!'.format(name)
            return

        Dataset = NYUDataset([sequence])
        dpt, gt3D = Dataset.imgStackDepthOnly(sequence.name)

        depth = []
        gtorig = []
        gtcrop = []
        T = []
        gt3Dorig = []
        gt3Dcrop = []
        com = []
        fileName = []

        for i in xrange(len(sequence.data)):
            data = sequence.data[i]
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


class convertICVLDatasetToh5(convertDatasetToh5):
    """convert ICVL dataset to h5"""
    def __init__(self, cacheDir, dataName):
        super(convertICVLDatasetToh5, self).__init__(cacheDir, dataName)

    def createSequence(self):
        """create ICVL Sequence, train and test sequence"""
        print("create ICVL dataset")

        di = ICVLImporter('../dataset/' + self.datasetName, cacheDir=self.cacheDir)
        Seq1 = di.loadSequence('train') # use dataset totally
        Seq2_1 = di.loadSequence('test_seq_1')
        Seq2_2 = di.loadSequence('test_seq_2')

        self.convert(Seq1)
        print("{} Train Seq1 ok!".format(self.datasetName))
        self.convert(Seq2_1)
        print("{} Test Seq1 ok!".format(self.datasetName))
        self.convert(Seq2_2)
        print("{} Test Seq2 ok!".format(self.datasetName))

    def convert(self, sequence):
        """convert ICVL sequence"""
        config = sequence.config
        name = '{}/{}_{}.h5'.format(self.cacheDir, self.datasetName, sequence.name)

        if os.path.isfile(name):
            print '{} exist, please check if h5 is right!'.format(name)
            return

        Dataset = ICVLDataset([sequence])
        dpt, gt3D = Dataset.imgStackDepthOnly(sequence.name)

        depth = []
        gtorig = []
        gtcrop = []
        T = []
        gt3Dorig = []
        gt3Dcrop = []
        com = []
        fileName = []

        for i in xrange(len(sequence.data)):
            data = sequence.data[i]
            depth.append(data.dpt)
            gtorig.append(data.gtorig)
            gtcrop.append(data.gtcrop)
            T.append(data.T)
            gt3Dorig.append(data.gt3Dorig)
            gt3Dcrop.append(data.gt3Dcrop)
            com.append(data.com)
            fileName.append(int(data.fileName[(data.fileName.find('image_') + 6) : (data.fileName.find('.png'))]))

        dataset = h5py.File(name, 'w')

        dataset['com'] = np.asarray(com)
        dataset['inds'] = np.asarray(fileName)
        dataset['config'] = config['cube']
        dataset['depth'] = np.asarray(dpt)
        dataset['joint'] = np.asarray(gt3D)
        dataset['gt3Dorig'] = np.asarray(gt3Dorig)
        dataset.close()

if __name__ == '__main__':

    cacheDir = '../dataset/cache'

    NYUCreator = convertNYUDatasetToh5(cacheDir, 'NYU')
    NYUCreator.createSequence()

    ICVLCreator = convertICVLDatasetToh5(cacheDir, 'ICVL')
    ICVLCreator.createSequence()
