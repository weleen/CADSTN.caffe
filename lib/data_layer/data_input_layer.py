#!/usr/bin/env python

__author__ = 'WuYiming'

import caffe
import numpy as np
import random
import h5py
import yaml

cachePath = '/home/wuyiming/git/Hand/dataset/cache/'

class DataRead(object):
    """Read the data from h5py"""
    def __init__(self, name='NYU', phase='train', path=cachePath, clip_length=16, dim=3):
        """
        :param dataPath:
        :param name:
        """
        self.name = name
        self.phase = phase
        self.cachePath = path
        self.clip_length = clip_length # how many frames each sequence
        self.data = {}
        self.dim = dim

    def loadData(self):
        """
        :return:
        """
        import os
        if self.name == 'NYU' and self.phase == 'test':
            # concate NYU test sequence together
            assert os.path.isfile(self.cachePath + self.name + '_' + self.phase + '_1.h5')\
            and os.path.isfile(self.cachePath + self.name + '_' + self.phase + '_2.h5'),\
                '{} or {} is not exists!'.format((self.cachePath + self.name + '_' + self.phase + '_1.h5'), \
                                                 (self.cachePath + self.name + '_' + self.phase + '_2.h5'))
            dataFile_1 = h5py.File(self.cachePath + self.name + '_' + self.phase + '_1.h5', 'r')
            dataFile_2 = h5py.File(self.cachePath + self.name + '_' + self.phase + '_2.h5', 'r')
            size_1 = dataFile_1['com'].shape[0]
            size_2 = dataFile_2['com'].shape[0]

            print('size of dataset is test1: {} and test2: {}'.format(size_1, size_2))

            self.data['com'] = np.array(dataFile_1['com']).tolist()
            self.data['com'].extend(np.array(dataFile_2['com']).tolist())
            self.data['com'] = np.array(self.data['com'])

            self.data['inds'] = np.array(dataFile_1['inds']).tolist()
            self.data['inds'].extend(np.array(dataFile_2['inds']).tolist())
            self.data['inds'] = np.array(self.data['inds'])

            self.data['config'] = np.array(dataFile_1['config']).reshape(1, self.dim).repeat(size_1, axis=0).tolist()
            self.data['config'].extend(np.array(dataFile_2['config']).reshape(1, self.dim).repeat(size_2, axis=0).tolist())
            self.data['config'] = np.array(self.data['config'])

            self.data['depth'] = np.array(dataFile_1['depth']).tolist()
            self.data['depth'].extend(np.array(dataFile_2['depth']).tolist())
            self.data['depth'] = np.array(self.data['depth'])

            self.data['joint'] = np.array(dataFile_1['joint']).tolist()
            self.data['joint'].extend(np.array(dataFile_2['joint']).tolist())
            self.data['joint'] = np.array(self.data['joint'])
        elif self.name == 'NYU' and self.phase == 'train':
            assert os.path.isfile(self.cachePath + self.name + '_' + self.phase + '.h5'),\
                '{} file is not exists!'.format(self.phase)
            dataFile = h5py.File(self.cachePath + self.name + '_' + self.phase + '.h5', 'r')
            size = dataFile['com'].shape[0]
            print('size of dataset is {}'.format(size))
            self.data['com'] = np.array(dataFile['com'])
            self.data['inds'] = np.array(dataFile['inds'])
            self.data['config'] = np.array(dataFile['config']).reshape(1, self.dim).repeat(size, axis=0)
            self.data['depth'] = np.array(dataFile['depth'])
            self.data['joint'] = np.array(dataFile['joint'])
        elif self.name == 'ICVL' and self.phase == 'test_1':
            # test seq 1, we test ICVL dataset seperately
            File1 = self.cachePath + self.name + '_' + self.phase[:-1] + 'seq_1.h5'
            assert os.path.isfile(File1), '{} is not exists!'.format(File1)

            dataFile_1 = h5py.File(File1, 'r')
            size_1 = dataFile_1['com'].shape[0]

            print('size of dataset is test1: {}'.format(size_1))

            self.data['com'] = np.array(dataFile_1['com'])
            self.data['inds'] = np.array(dataFile_1['inds'])
            self.data['config'] = np.array(dataFile_1['config']).reshape(1, self.dim).repeat(size_1, axis=0)
            self.data['depth'] = np.array(dataFile_1['depth'])
            self.data['joint'] = np.array(dataFile_1['joint'])
        elif self.name == 'ICVL' and self.phase == 'test_2':
            File2 = self.cachePath + self.name + '_' + self.phase[:-1] + 'seq_2.h5'
            assert os.path.isfile(File2), '{} is not exists!'.format(File2)

            dataFile_2 = h5py.File(File2, 'r')
            size_2 = dataFile_2['com'].shape[0]

            print('size of dataset is test1: {}'.format(size_2))

            self.data['com'] = np.array(dataFile_2['com'])
            self.data['inds'] = np.array(dataFile_2['inds'])
            self.data['config'] = np.array(dataFile_2['config']).reshape(1, self.dim).repeat(size_2, axis=0)
            self.data['depth'] = np.array(dataFile_2['depth'])
            self.data['joint'] = np.array(dataFile_2['joint'])
        elif self.name == 'ICVL' and self.phase == 'train':
            assert os.path.isfile(self.cachePath + self.name + '_' + self.phase + '.h5'), \
                '{} file is not exists!'.format(self.cachePath + self.name + '_' + self.phase + '.h5')
            dataFile = h5py.File(self.cachePath + self.name + '_' + self.phase + '.h5', 'r')
            size = dataFile['com'].shape[0]
            print('size of dataset is {}'.format(size))
            self.data['com'] = np.array(dataFile['com'])
            self.data['inds'] = np.array(dataFile['inds'])
            self.data['config'] = np.array(dataFile['config']).reshape(1, self.dim).repeat(size, axis=0)
            self.data['depth'] = np.array(dataFile['depth'])
            self.data['joint'] = np.array(dataFile['joint'])
        else:
            raise Exception('unknow dataset {} or phase {}.'.format(self.name, self.phase))

        print('dataset: {} phase: {}'.format(self.name, self.phase))


    def dataToSeq(self):
        """
        Transform the dataset into sequence
        :return: sequence of frames
        """
        dataSize = self.data['inds'].shape[0]
        self.seqSize = int(np.ceil(dataSize / float(self.clip_length)))
        seq = []
        for i in xrange(self.seqSize):
            current_seq = []
            for j in xrange(self.clip_length):
                ind = i * self.clip_length + j
                if ind >= dataSize:
                    tmp = current_seq[-1]
                else:
                    tmp = {'com': self.data['com'][ind],
                           'inds': self.data['inds'][ind],
                           'depth': self.data['depth'][ind],
                           'joint': self.data['joint'][ind],
                           'config': self.data['config'][ind],
                           'clip_markers': 1 if j != 0 else 0}
                current_seq.append(tmp.copy())
            seq.append(current_seq)

        return seq


class sequenceGenerator(object):
    def __init__(self, buffer_size, clip_length, num_seq, seq_dict):
        """
        :param buffer_size:
        :param clip_length:
        :param num_seq:
        :param seq_dict:
        """
        self.buffer_size = buffer_size
        self.clip_length = clip_length
        self.N = self.buffer_size*self.clip_length
        self.num_seq = num_seq
        self.idx = 0
        self.seq_dict = np.array(seq_dict)

    def __call__(self):
        """
        :return:
        """

        if self.idx + self.buffer_size >= self.num_seq:
            idx_list = range(self.idx, self.num_seq)
            idx_list.extend(range(0, self.buffer_size - (self.num_seq - self.idx)))
        else:
            idx_list = range(self.idx, self.idx + self.buffer_size)

        self.idx += self.buffer_size
        if self.idx >= self.num_seq:
            self.idx = self.idx - self.num_seq

        #print('index list = ', idx_list)
        return self.seq_dict[idx_list]


class videoRead(caffe.Layer):

    def initialize(self):
        """set the defualt param, overwrite it"""
        self.name = 'NYU'
        self.train_or_test = 'test'
        self.N = self.buffer_size * self.frames
        self.idx = 0
        self.path = cachePath
        self.joints = 14
        self.imagesize = 128
        self.dim = 3

    def setup(self, bottom, top):
        """setup for layer"""
        layer_params = yaml.load(self.param_str)
        # read the layer param contain the sequence number and sequence size
        self.buffer_size, self.frames = map(int, layer_params['sequence_num_size'].split())
        self.initialize()

        dataReader = DataRead(self.name, self.train_or_test, self.path, self.frames, self.dim)
        dataReader.loadData()
        self.seq_dict = dataReader.dataToSeq()

        self.sequence_generator = sequenceGenerator(self.buffer_size, self.frames,\
                                                   len(self.seq_dict), self.seq_dict)

        self.top_names = ['depth', 'joint', 'clip_markers', 'com', 'config', 'inds']
        print 'Outputs: ', self.top_names
        if len(top) != len(self.top_names):
            raise Exception('Incorrect number of outputs (expect %d, got %d)'\
                            %(len(self.top_names), len(top)))

        for top_index, name in enumerate(self.top_names):
            if name == 'depth':
                shape = (self.N, 1, self.imagesize, self.imagesize)
            elif name == 'joint':
                shape = (self.N, self.dim * self.joints)
            elif name == 'clip_markers':
                shape = (self.N, )
            elif name == 'com':
                shape = (self.N, self.dim)
            elif name == 'config':
                shape = (self.N, self.dim)
            elif name == 'inds':
                shape = (self.N, )
            top[top_index].reshape(*shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        data = self.sequence_generator()

        depth = np.zeros((self.N, 1, self.imagesize, self.imagesize))
        joint = np.zeros((self.N, self.dim * self.joints))
        cm = np.zeros((self.N, ))
        com = np.zeros((self.N, self.dim))
        config = np.zeros((self.N, self.dim))
        inds = np.zeros((self.N))
        #print 'data = ',data
        #print 'data.shape = ',data.shape

        # rearrange the dataset for LSTM
        for i in xrange(self.frames):
            for j in xrange(self.buffer_size):
                idx = i*self.buffer_size + j
                depth[idx] = data[j][i]['depth']
                joint[idx] = data[j][i]['joint'].reshape(self.joints*3)
                cm[idx] = data[j][i]['clip_markers']
                com[idx] = data[j][i]['com']
                config[idx] = data[j][i]['config']
                inds[idx] = data[j][i]['inds']

        for top_index, name in zip(range(len(top)), self.top_names):
            if name == 'depth':
                for i in range(self.N):
                    top[top_index].data[i, ...] = depth[i]
            elif name == 'joint':
                for i in range(self.N):
                    top[top_index].data[i, ...] = joint[i]
            elif name == 'clip_markers':
                top[top_index].data[...] = cm
            elif name == 'com':
                for i in range(self.N):
                    top[top_index].data[i, ...] = com[i]
            elif name == 'config':
                for i in range(self.N):
                    top[top_index].data[i, ...] = config[i]
            elif name == 'inds':
                top[top_index].data[...] = inds

    def backward(self):
        pass


class NYUTrainSeq(videoRead):
    def initalize(self):
        self.name = 'NYU'
        self.train_or_test = 'train'
        self.N = self.buffer_size*self.frames
        self.idx = 0
        self.path = cachePath
        self.joints = 14
        self.dim = 3
        self.imagesize = 128

class NYUTestSeq(videoRead):
    def initalize(self):
        self.name = 'NYU'
        self.train_or_test = 'test'
        self.N = self.buffer_size*self.frames
        self.idx = 0
        self.path = cachePath
        self.joints = 14
        self.dim = 3
        self.imagesize = 128

class ICVLTrainSeq(videoRead):
    def initialize(self):
        self.name = 'ICVL'
        self.train_or_test = 'train'
        self.N = self.buffer_size*self.frames
        self.idx = 0
        self.path = cachePath
        self.joints = 16
        self.dim = 3
        self.imagesize = 128

class ICVLTestSeq1(videoRead):
    def initialize(self):
        self.name = 'ICVL'
        self.train_or_test = 'test_1'
        self.N = self.buffer_size*self.frames
        self.idx = 0
        self.path = cachePath
        self.joints = 16
        self.dim = 3
        self.imagesize = 128

class ICVLTestSeq2(videoRead):
    def initialize(self):
        self.name = 'ICVL'
        self.train_or_test = 'test_2'
        self.N = self.buffer_size*self.frames
        self.idx = 0
        self.path = cachePath
        self.joints = 16
        self.dim = 3
        self.imagesize = 128