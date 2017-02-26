#!/usr/bin/env python
import _init_paths
import numpy as np
from data.importers import NYUImporter
from data.dataset import NYUDataset
from util.realtimehandposepipeline import RealtimeHandposePipeline

root = '/home/wuyiming/git/Hand'
if __name__ == '__main__':
    di = NYUImporter(root + '/dataset/NYU', cacheDir=root + '/dataset/cache/')
    Seq2 = di.loadSequence('test_1')
    testSeqs = [Seq2]

    testDataSet = NYUDataset(testSeqs)
    test_data, test_gt3D = testDataSet.imgStackDepthOnly('test_1')
    
    config = {'fx': 588., 'fy': 587., 'cube': (300, 300, 300)}
    netPath = root + '/models/NYU/hand_lstm_small_frame_size/deploy.prototxt'
    netWeight  = root + '/weights/NYU/hand_lstm_small_frame_size/hand_lstm_small_frame_size_iter_200000.caffemodel'
    rtp = RealtimeHandposePipeline(di, config, netPath, netWeight)

    # use filenames
    filenames = []
    for i in testSeqs[0].data:
        filenames.append(i.fileName)
    rtp.processFiles(filenames)
