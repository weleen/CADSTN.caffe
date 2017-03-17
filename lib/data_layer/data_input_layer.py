#!/usr/bin/env python

__author__ = 'WuYiming'

import _init_paths
import caffe
import numpy as np
import yaml
from data.importers import NYUImporter, ICVLImporter
from data.dataset import NYUDataset, ICVLDataset

root = '/home/wuyiming/git/Hand'
cachePath = root + '/dataset/cache'
DEBUG = False

class DataRead(object):
    """Read the data"""
    def __init__(self, name='NYU', phase='train', path=cachePath, clip_length=16, dim=3, dsize=(96, 96)):
        """
        :param dataPath:
        :param name:
        """
        self.name = name
        self.phase = phase
        self.cachePath = path + str(dsize[0]) + '/'
        self.clip_length = clip_length # how many frames each sequence
        self.data = {}
        self.dim = dim
        self.dsize = dsize

    def convert(self, sequence, size_before=None):
        """convert sequence data"""
        config = sequence.config
        if self.name == 'NYU':
            Dataset = NYUDataset([sequence])
        elif self.name == 'ICVL':
            Dataset = ICVLDataset([sequence])
        dpt, gt3D = Dataset.imgStackDepthOnly(sequence.name)

        dataset = {}
        gtorig = []
        gtcrop = []
        T = []
        gt3Dorig = []
        gt3Dcrop = []
        com = []
        fileName = []

        size = len(sequence.data)
        print('size of {} {} dataset is {}'.format(self.name, sequence.name, size))
        for i in xrange(len(sequence.data)):
            data = sequence.data[i]
            gtorig.append(data.gtorig)
            gtcrop.append(data.gtcrop)
            T.append(data.T)
            gt3Dorig.append(data.gt3Dorig)
            gt3Dcrop.append(data.gt3Dcrop)
            com.append(data.com)
            if self.name == 'NYU':
                fileName.append(int(data.fileName[-11:-4]))
            elif self.name == 'ICVL' and size_before is not None:
                fileName.append(int(data.fileName[(data.fileName.find('image_') + 6): \
                    (data.fileName.find('.png'))]) + size_before)

        dataset['depth'] = np.asarray(dpt)
        dataset['gtorig'] = np.asarray(gtorig)
        dataset['gtcrop'] = np.asarray(gtcrop)
        dataset['T'] = np.asarray(T)
        dataset['gt3Dorig'] = np.asarray(gt3Dorig)
        dataset['gt3Dcrop'] = np.asarray(gt3Dcrop)
        dataset['com'] = np.asarray(com)
        dataset['inds'] = np.asarray(fileName)
        dataset['config'] = np.asarray(config['cube']).reshape(1, self.dim).repeat(size, axis=0)
        dataset['joint'] = np.asarray(gt3D)

        return dataset

    def loadData(self):
        """
        load the dataset
        :return: dataset
        """
        print('create {} {} dataset'.format(self.name, self.phase))
        if self.name == 'NYU':
            di = NYUImporter(root + '/dataset/' + self.name, cacheDir=self.cachePath)
            if self.phase == 'train':
                sequence = di.loadSequence('train', flip=True, rotation=True, dsize=self.dsize)  # train sequence
                self.data = self.convert(sequence)
            elif self.phase == 'test':
                sequence1 = di.loadSequence('test_1', dsize=self.dsize)  # test sequence 1
                sequence2 = di.loadSequence('test_2', dsize=self.dsize)  # test sequence 2
                data_1 = self.convert(sequence1)
                data_2 = self.convert(sequence2)

                self.data['depth'] = np.concatenate([data_1['depth'], data_2['depth']])
                self.data['gtorig'] = np.concatenate([data_1['gtorig'], data_2['gtorig']])
                self.data['gtcrop'] = np.concatenate([data_1['gtcrop'], data_2['gtcrop']])
                self.data['T'] = np.concatenate([data_1['T'], data_2['T']])
                self.data['gt3Dorig'] = np.concatenate([data_1['gt3Dorig'], data_2['gt3Dorig']])
                self.data['gt3Dcrop'] = np.concatenate([data_1['gt3Dcrop'], data_2['gt3Dcrop']])
                self.data['com'] = np.concatenate([data_1['com'], data_2['com']])
                self.data['inds'] = np.concatenate([data_1['inds'], data_2['inds']])
                self.data['config'] = np.concatenate([data_1['config'], data_2['config']])
                self.data['joint'] = np.concatenate([data_1['joint'], data_2['joint']])

        elif self.name == 'ICVL':
            di = ICVLImporter(root + '/dataset/' + self.name, cacheDir=self.cachePath)
            if self.phase == 'train':
                sequence = di.loadSequence('train', dsize=self.dsize)  # use dataset totally
                self.data = self.convert(sequence)
            elif self.phase == 'test':
                sequence1 = di.loadSequence('test_seq_1', dsize=self.dsize)  # test sequence 1
                sequence2 = di.loadSequence('test_seq_2', dsize=self.dsize)  # test sequence 2
                data_1 = self.convert(sequence1)
                size_1 = data_1['com'].shape[0]
                data_2 = self.convert(sequence2, size_before=size_1) # concate two test sequence together

                self.data['depth'] = np.concatenate([data_1['depth'], data_2['depth']])
                self.data['gtorig'] = np.concatenate([data_1['gtorig'], data_2['gtorig']])
                self.data['gtcrop'] = np.concatenate([data_1['gtcrop'], data_2['gtcrop']])
                self.data['T'] = np.concatenate([data_1['T'], data_2['T']])
                self.data['gt3Dorig'] = np.concatenate([data_1['gt3Dorig'], data_2['gt3Dorig']])
                self.data['gt3Dcrop'] = np.concatenate([data_1['gt3Dcrop'], data_2['gt3Dcrop']])
                self.data['com'] = np.concatenate([data_1['com'], data_2['com']])
                self.data['inds'] = np.concatenate([data_1['inds'], data_2['inds']])
                self.data['config'] = np.concatenate([data_1['config'], data_2['config']])
                self.data['joint'] = np.concatenate([data_1['joint'], data_2['joint']])

        else:
            raise Exception('unknow dataset {} or phase {}.'.format(self.name, self.phase))

        return self.data


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
                    tmp = {'depth': self.data['depth'][ind],
                           'gtorig': self.data['gtorig'][ind],
                           'gtcrop': self.data['gtcrop'][ind],
                           'T': self.data['T'][ind],
                           'gt3Dorig': self.data['gt3Dorig'][ind],
                           'gt3Dcrop': self.data['gt3Dcrop'][ind],
                           'com': self.data['com'][ind],
                           'inds': self.data['inds'][ind],
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
        self.buffer_size = int(layer_params['buffer_size'])
        self.frames = int(layer_params['frame_size'])
        self.suffle = (layer_params['shuffle'] == "true")
        self.imagesize = int(layer_params['size'])
        self.initialize()

        dataReader = DataRead(self.name, self.train_or_test, self.path, self.frames, \
                              self.dim, dsize=(self.imagesize, self.imagesize))
        dataReader.loadData()
        self.seq_dict = dataReader.dataToSeq()

        self.sequence_generator = sequenceGenerator(self.buffer_size, self.frames,\
                                                   len(self.seq_dict), self.seq_dict)

        self.top_names = ['depth', 'gtorig', 'gtcrop', 'T', 'gt3Dorig',
                          'gt3Dcrop', 'joint', 'clip_markers', 'com', 'config', 'inds']
        print 'Outputs: ', self.top_names
        if len(top) != len(self.top_names):
            raise Exception('Incorrect number of outputs (expect %d, got %d)'\
                            %(len(self.top_names), len(top)))

        if DEBUG:
            print "configuration: "
            print "dataset = {}, phase = {}".format(self.name, self.train_or_test)
            print "buffer_size = {}, frame_size = {}".format(self.buffer_size, self.frames)
            print "image_size = {}".format(self.imagesize)

        for top_index, name in enumerate(self.top_names):
            if name == 'depth':
                shape = (self.N, 1, self.imagesize, self.imagesize)
            elif name == 'gtorig':
                shape = (self.N, self.joints, self.dim)
            elif name == 'gtcrop':
                shape = (self.N, self.joints, self.dim)
            elif name == 'T':
                shape = (self.N, 3 * 3)
            elif name == 'gt3Dorig':
                shape = (self.N, self.joints, self.dim)
            elif name == 'gt3Dcrop':
                shape = (self.N, self.joints, self.dim)
            elif name == 'joint':
                shape = (self.N, self.joints, self.dim)
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
        gtorig = np.zeros((self.N, self.joints, self.dim))
        gtcrop = np.zeros((self.N, self.joints, self.dim))
        T = np.zeros((self.N, 3 * 3))
        gt3Dorig = np.zeros((self.N, self.joints, self.dim))
        gt3Dcrop = np.zeros((self.N, self.joints, self.dim))
        joint = np.zeros((self.N, self.joints, self.dim))
        cm = np.zeros((self.N, ))
        com = np.zeros((self.N, self.dim))
        config = np.zeros((self.N, self.dim))
        inds = np.zeros((self.N))

        # rearrange the dataset for LSTM
        for i in xrange(self.frames):
            for j in xrange(self.buffer_size):
                idx = i*self.buffer_size + j
                depth[idx] = data[j][i]['depth']
                gtorig[idx] = data[j][i]['gtorig']
                gtcrop[idx] = data[j][i]['gtcrop']
                T[idx] = data[j][i]['T'].reshape(3*3)
                gt3Dorig[idx] = data[j][i]['gt3Dorig']
                gt3Dcrop[idx] = data[j][i]['gt3Dcrop']
                joint[idx] = data[j][i]['joint']
                cm[idx] = data[j][i]['clip_markers']
                com[idx] = data[j][i]['com']
                config[idx] = data[j][i]['config']
                inds[idx] = data[j][i]['inds']

        if DEBUG:
            print "top shape:"
            print "top[depth] shape = {}, copy from {}".format(top[0].shape, depth.shape)
            print "top[gtorig] shape = {}, copy from {}".format(top[1].shape, gtorig.shape)
            print "top[gtcrop] shape = {}, copy from {}".format(top[2].shape, gtcrop.shape)
            print "top[T] shape = {}, copy from {}".format(top[3].shape, T.shape)
            print "top[gt3Dorig] shape = {}, copy from {}".format(top[4].shape, gt3Dorig.shape)
            print "top[gt3Dcrop] shape = {}, copy from {}".format(top[5].shape, gt3Dcrop.shape)
            print "top[joint] shape = {}, copy from {}".format(top[6].shape, joint.shape)
            print "top[clip_markers] shape = {}, copy from {}".format(top[7].shape, cm.shape)
            print "top[com] shape = {}, copy from {}".format(top[8].shape, com.shape)
            print "top[config] shape = {}, copy from {}".format(top[9].shape, config.shape)
            print "top[inds] shape = {}, copy from {}".format(top[10].shape, inds.shape)

        for top_index, name in zip(range(len(top)), self.top_names):
            if name == 'depth':
                for i in range(self.N):
                    top[top_index].data[i, ...] = depth[i]
            elif name == 'gtorig':
                for i in range(self.N):
                    top[top_index].data[i, ...] = gtorig[i]
            elif name == 'gtcrop':
                for i in range(self.N):
                    top[top_index].data[i, ...] = gtcrop[i]
            elif name == 'T':
                for i in range(self.N):
                    top[top_index].data[i, ...] = T[i]
            elif name == 'gt3Dorig':
                for i in range(self.N):
                    top[top_index].data[i, ...] = gt3Dorig[i]
            elif name == 'gt3Dcrop':
                for i in range(self.N):
                    top[top_index].data[i, ...] = gt3Dcrop[i]
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
    def initialize(self):
        self.name = 'NYU'
        self.train_or_test = 'train'
        self.N = self.buffer_size*self.frames
        self.idx = 0
        self.path = cachePath
        self.joints = 14
        self.dim = 3

class NYUTestSeq(videoRead):
    def initialize(self):
        self.name = 'NYU'
        self.train_or_test = 'test'
        self.N = self.buffer_size*self.frames
        self.idx = 0
        self.path = cachePath
        self.joints = 14
        self.dim = 3

class ICVLTrainSeq(videoRead):
    def initialize(self):
        self.name = 'ICVL'
        self.train_or_test = 'train'
        self.N = self.buffer_size*self.frames
        self.idx = 0
        self.path = cachePath
        self.joints = 16
        self.dim = 3

class ICVLTestSeq(videoRead):
    def initialize(self):
        self.name = 'ICVL'
        self.train_or_test = 'test'
        self.N = self.buffer_size*self.frames
        self.idx = 0
        self.path = cachePath
        self.joints = 16
        self.dim = 3


if __name__ == '__main__':
    #data = DataRead(name='NYU', phase='train',dsize=(128, 128))
    #data_load = data.loadData()
    #data = DataRead(name='NYU', phase='test', dsize=(128, 128))
    #data_load = data.loadData()
    #data = DataRead(name='ICVL', phase='train',dsize=(128, 128))
    #data_load = data.loadData()
    #data = DataRead(name='ICVL', phase='test', dsize=(128, 128))
    #data_load = data.loadData()
    
    #data = DataRead(name='NYU', phase='train',dsize=(96, 96))
    #data_load = data.loadData()
    #data = DataRead(name='NYU', phase='test', dsize=(96, 96))
    #data_load = data.loadData()
    #data = DataRead(name='ICVL', phase='train',dsize=(96, 96))
    #data_load = data.loadData()
    #data = DataRead(name='ICVL', phase='test', dsize=(96, 96))
    #data_load = data.loadData()
    pass