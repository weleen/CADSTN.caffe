"""
This is the data preprocess for hand pose estimation.
"""

import numpy as np
import scipy.io
from data.importers import NYUImporter
from data.importers import ICVLImporter
from data.dataset import NYUDataset
import h5py

def convertNYUSequenceToh5(Sequence):
    """
    :param Sequence: orderDict
    :return: None
    """
    config = Sequence.config
    name = 'cache/{}_{}.h5'.format('NYU', Sequence.name)

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


if __name__ == '__main__':

    rng = np.random.RandomState(1234)

    print("create dataset")

    di = NYUImporter('./dataset/NYU')
    Seq1 = di.loadSequence('train', rng=rng)
    trainSeqs = [Seq1]

    Seq2_1 = di.loadSequence('test_1')
    Seq2_2 = di.loadSequence('test_2')
    testSeqs = [Seq2_1, Seq2_2]

    convertNYUSequenceToh5(Seq1)
    print("Seq1 ok!")
    convertNYUSequenceToh5(Seq2_1)
    print("Seq2_1 ok!")
    convertNYUSequenceToh5(Seq2_2)
    print("Seq2_2 ok!")