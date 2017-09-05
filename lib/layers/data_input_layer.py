#!/usr/bin/env python

import os
import sys
import _init_paths
import caffe
import numpy as np
import yaml
import time
from data.importers import NYUImporter, ICVLImporter
from data.dataset import NYUDataset, ICVLDataset

root = '/home/wuyiming/git/Hand-dev'
cachePath = root + '/dataset/cache'
DEBUG = False


class DataRead(object):
    """Read the data"""

    def __init__(self, name='NYU', phase='train', path=cachePath, dim=3, dsize=(128, 128), aug=False):
        """
        :param dataPath:
        :param name:
        """
        self.name = name
        self.phase = phase
        self.aug = aug
        if aug:
            # if train NYU, use augmented folder
            self.cachePath = path + '_augment'
        else:  # if not, use original folder
            self.cachePath = path
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
        com = []
        fileName = []
        dpt3D = []
        size = len(sequence.data)
        print('size of {} {} dataset is {}'.format(self.name, sequence.name, size))
        for i in xrange(len(sequence.data)):
            data = sequence.data[i]
            com.append(data.com)
            dpt3D.append(data.dpt3D)
            if self.name == 'NYU':
                fileName.append(int(data.fileName[-11:-4]))
            elif self.name == 'ICVL' and self.phase == 'train' and size_before is None:
                part_folder = data.fileName[(data.fileName.find('Depth/')+7): \
                    (data.fileName.find('/image'))]
                part_file = data.fileName[(data.fileName.find('image_') + 6): \
                    (data.fileName.find('.png'))]
                p = part_folder.split('/')
                if len(p) == 2:
                    # augmneted folder, e.g. '90/' '-90/'
                    p[0] = p[0].replace('-', '1')
                    part_folder = p[0] + p[1]
                index_name = int(part_folder) * 100000 + int(part_file)
                fileName.append(index_name)
            elif self.name == 'ICVL' and self.phase == 'test' and size_before is not None:
                fileName.append(int(data.fileName[(data.fileName.find('image_') + 6): \
                    (data.fileName.find('.png'))]) + size_before)
            else:
                fileName.append(int(data.fileName[(data.fileName.find('image_') + 6): \
                    (data.fileName.find('.png'))]))

        dataset['depth'] = np.asarray(dpt)
        dataset['dpt3D'] = np.asarray(dpt3D)
        dataset['com'] = np.asarray(com)
        dataset['inds'] = np.asarray(fileName)
        dataset['config'] = np.asarray(config['cube']).reshape(1, self.dim).repeat(size, axis=0)
        dataset['joint'] = np.asarray(gt3D)

        return dataset

    def loadData(self):
        """
        load the dataset
        :return: data(dict)
        """
        data = {}  # dataset would be return
        print('create {} {} dataset'.format(self.name, self.phase))
        if self.name == 'NYU':
            di = NYUImporter(root + '/dataset/' + self.name, cacheDir=self.cachePath)
            if self.phase == 'train':
                if self.aug:  # do augmentation for training
                    sequence = di.loadSequence('train', rotation=True, docom=True, dsize=self.dsize)
                else:
                    sequence = di.loadSequence('train', docom=True, dsize=self.dsize)
                data = self.convert(sequence)
            elif self.phase == 'test':
                sequence1 = di.loadSequence('test_1', docom=True, dsize=self.dsize)  # test sequence 1
                sequence2 = di.loadSequence('test_2', docom=True, dsize=self.dsize)  # test sequence 2
                data_1 = self.convert(sequence1)
                data_2 = self.convert(sequence2)

                data['depth'] = np.concatenate([data_1['depth'], data_2['depth']])
                data['dpt3D'] = np.concatenate([data_1['dpt3D'], data_2['dpt3D']])
                data['com'] = np.concatenate([data_1['com'], data_2['com']])
                data['inds'] = np.concatenate([data_1['inds'], data_2['inds']])
                data['config'] = np.concatenate([data_1['config'], data_2['config']])
                data['joint'] = np.concatenate([data_1['joint'], data_2['joint']])

        elif self.name == 'ICVL':
            di = ICVLImporter(root + '/dataset/' + self.name, cacheDir=self.cachePath)
            if self.phase == 'train':
                if self.aug:
                    # augment by adding rotation images
                    sequence = di.loadSequence('train', ['0', '45', '67-5', '90', '-90', '-180'], docom=True, dsize=self.dsize)
                else:
                    sequence = di.loadSequence('train',['0'], docom=True, dsize=self.dsize)
                data = self.convert(sequence)
            elif self.phase == 'test':
                sequence1 = di.loadSequence('test_seq_1', docom=True, dsize=self.dsize)  # test sequence 1
                sequence2 = di.loadSequence('test_seq_2', docom=True, dsize=self.dsize)  # test sequence 2

                data_1 = self.convert(sequence1)
                size_1 = data_1['com'].shape[0]
                data_2 = self.convert(sequence2, int(size_1))  # concate two test sequence together

                data['depth'] = np.concatenate([data_1['depth'], data_2['depth']])
                data['dpt3D'] = np.concatenate([data_1['dpt3D'], data_2['dpt3D']])
                data['com'] = np.concatenate([data_1['com'], data_2['com']])
                data['inds'] = np.concatenate([data_1['inds'], data_2['inds']])
                data['config'] = np.concatenate([data_1['config'], data_2['config']])
                data['joint'] = np.concatenate([data_1['joint'], data_2['joint']])
        else:
            raise Exception('unknow dataset {} or phase {}.'.format(self.name, self.phase))

        return data


class sequenceGenerator(object):
    def __init__(self, buffer_size, frame_size, dataSize, data, shuffle, phase):
        """
        :param buffer_size:
        :param frame_size:
        :param num_seq:
        :param seq_dict:
        """
        self.buffer_size = buffer_size
        self.frame_size = frame_size
        self.dataSize = dataSize
        self.idx = 0
        self.data = data
        self.shuffle = shuffle
        self.phase = phase

    def generate_list(self, idx):
        list_i = []
        list_i.append(idx)
        for i in range(1, self.frame_size):
            current_file_path = self.data['inds'][idx + i]
            previous_file_path = self.data['inds'][idx + i - 1]
            current_folder_name = int(current_file_path) / 100000
            previous_folder_name = int(previous_file_path) / 100000
            current_file_index = int(current_file_path) % 100000
            previous_file_index = int(previous_file_path) % 100000
            if current_folder_name != previous_folder_name:
                while (len(list_i) != self.frame_size):
                    list_i.append(idx + i - 1)
                break
            if int(current_file_index) - int(previous_file_index) > 20:
                while (len(list_i) != self.frame_size):
                    list_i.append(idx + i - 1)
                break
            else:
                list_i.append(idx + i)

        return list_i

    def generate_test_list(self, idx):
        list_i = []
        list_i.append(idx)
        for i in range(1, self.frame_size):
            current_inds = self.data['inds'][idx + i]
            previous_inds = self.data['inds'][idx + i - 1]
            if current_inds >= 702 and previous_inds < 702:
                while (len(list_i) != self.frame_size):
                    list_i.append(idx + i - 1)
            elif current_inds >= 1595:
                while (len(list_i) != self.frame_size):
                    list_i.append(idx + i - 1)
            elif int(current_inds) - int(previous_inds) > 20:
                while (len(list_i) != self.frame_size):
                    list_i.append(idx + i - 1)
            else:
                list_i.append(idx + i)
        return list_i

    def __call__(self):
        """
        :return:
        """
        batch = []
        if self.shuffle:
            # random choice index for training
            # get the start idx for lstm
            idx_list = np.random.choice(range(0, self.dataSize - self.frame_size + 1), self.buffer_size)

            for idx in idx_list:
                if self.phase == 'train':
                    list_i = self.generate_list(idx)
                else:
                    list_i = self.generate_test_list(idx)
                minibatch = []

                for i in list_i:
                    minibatch.append({'depth': self.data['depth'][i],
                                      'dpt3D': self.data['dpt3D'][i],
                                      'com': self.data['com'][i],
                                      'inds': self.data['inds'][i],
                                      'joint': self.data['joint'][i],
                                      'config': self.data['config'][i],
                                      'clip_markers': 1 if i != list_i[0] else 0})
                batch.append(minibatch)
        else:
            # start from 0, without overlap between two batches
            if self.idx + self.buffer_size * self.frame_size > self.dataSize:
                idx_list = range(self.idx, self.dataSize, self.frame_size)
                while len(idx_list) != self.buffer_size:
                    idx_list.append(idx_list[-1])
            else:
                idx_list = range(self.idx, self.idx + self.buffer_size * self.frame_size, self.frame_size)
            self.idx += self.buffer_size * self.frame_size
            if self.idx >= self.dataSize:
                self.idx = 0

            for idx in idx_list:
                minibatch = []
                start = idx
                end = idx + self.frame_size
                for i in xrange(start, end):
                    if i < self.dataSize:
                        temp = {'depth': self.data['depth'][i],
                                'dpt3D': self.data['dpt3D'][i],
                                'com': self.data['com'][i],
                                'inds': self.data['inds'][i],
                                'joint': self.data['joint'][i],
                                'config': self.data['config'][i],
                                'clip_markers': 1 if i != start else 0}
                    minibatch.append(temp)
                batch.append(minibatch)

        return batch


class videoRead(caffe.Layer):
    def initialize(self):
        """set the defualt param, overwrite it"""
        self.name = 'NYU'
        self.phase = 'test'
        self.N = self.buffer_size * self.frame_size
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
        self.frame_size = int(layer_params['frame_size'])
        # whether we do augmentation, only in train dataset
        self.aug = (layer_params['augment'] == "true")
        self.shuffle = (layer_params['shuffle'] == "true")
        self.imagesize = int(layer_params['size'])
        self.initialize()
        dataReader = DataRead(self.name, self.phase, self.path, self.dim,
            dsize=(self.imagesize, self.imagesize), aug=self.aug)
        self.data = dataReader.loadData()
        self.sequence_generator = sequenceGenerator(self.buffer_size,
            self.frame_size, len(self.data['depth']),
            self.data, self.shuffle, self.phase)

        self.top_names = ['depth', 'dpt3D', 'joint', 'clip_markers', 'com', 'config', 'inds']
        print 'Outputs: ', self.top_names
        if len(top) != len(self.top_names):
            raise Exception('Incorrect number of outputs (expect %d, got %d)' \
                            % (len(self.top_names), len(top)))

        if DEBUG:
            print "configuration: "
            print "dataset = {}, phase = {}".format(self.name, self.train_or_test)
            print "buffer_size = {}, frame_size = {}".format(self.buffer_size, self.frame_size)
            print "image_size = {}".format(self.imagesize)

        for top_index, name in enumerate(self.top_names):
            if name == 'depth':
                shape = (self.N, 1, self.imagesize, self.imagesize)
            elif name == 'dpt3D':
                shape = (self.N, 8, self.imagesize, self.imagesize)
            elif name == 'joint':
                shape = (self.N, self.joints, self.dim)
            elif name == 'clip_markers':
                shape = (self.N,)
            elif name == 'com':
                shape = (self.N, self.dim)
            elif name == 'config':
                shape = (self.N, self.dim)
            elif name == 'inds':
                shape = (self.N,)
            top[top_index].reshape(*shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        data = self.sequence_generator()

        depth = np.zeros((self.N, 1, self.imagesize, self.imagesize))
        dpt3D = np.zeros((self.N, 8, self.imagesize, self.imagesize))
        joint = np.zeros((self.N, self.joints, self.dim))
        cm = np.zeros((self.N,))
        com = np.zeros((self.N, self.dim))
        config = np.zeros((self.N, self.dim))
        inds = np.zeros((self.N))

        # rearrange the dataset for LSTM
        for i in xrange(self.frame_size):
            for j in xrange(self.buffer_size):
                idx = i * self.buffer_size + j
                depth[idx] = data[j][i]['depth']
                dpt3D[idx] = data[j][i]['dpt3D']
                joint[idx] = data[j][i]['joint']
                cm[idx] = data[j][i]['clip_markers']
                com[idx] = data[j][i]['com']
                config[idx] = data[j][i]['config']
                inds[idx] = data[j][i]['inds']

        if DEBUG:
            print "top shape:"
            print "top[depth] shape = {}, copy from {}".format(top[0].shape, depth.shape)
            print "top[joint] shape = {}, copy from {}".format(top[1].shape, joint.shape)
            print "top[clip_markers] shape = {}, copy from {}".format(top[2].shape, cm.shape)
            print "top[com] shape = {}, copy from {}".format(top[3].shape, com.shape)
            print "top[config] shape = {}, copy from {}".format(top[4].shape, config.shape)
            print "top[inds] shape = {}, copy from {}".format(top[5].shape, inds.shape)

        for top_index, name in zip(range(len(top)), self.top_names):
            if name == 'depth':
                for i in range(self.N):
                    top[top_index].data[i, ...] = depth[i]
            elif name == 'dpt3D':
                for i in range(self.N):
                    top[top_index].data[i, ...] = dpt3D[i]
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
        self.phase = 'train'
        self.N = self.buffer_size * self.frame_size
        self.path = cachePath
        self.joints = 14
        self.dim = 3


class NYUTestSeq(videoRead):
    def initialize(self):
        self.name = 'NYU'
        self.phase = 'test'
        self.N = self.buffer_size * self.frame_size
        self.path = cachePath
        self.joints = 14
        self.dim = 3


class ICVLTrainSeq(videoRead):
    def initialize(self):
        self.name = 'ICVL'
        self.phase = 'train'
        self.N = self.buffer_size * self.frame_size
        self.path = cachePath
        self.joints = 16
        self.dim = 3


class ICVLTestSeq(videoRead):
    def initialize(self):
        self.name = 'ICVL'
        self.phase = 'test'
        self.N = self.buffer_size * self.frame_size
        self.idx = 0
        self.path = cachePath
        self.joints = 16
        self.dim = 3


if __name__ == '__main__':
    # data = DataRead(name='NYU', phase='train',dsize=(128, 128))
    # data_load = data.loadData()
    # sequence_generator = sequenceGenerator(128, 1, len(data_load), data_load, True)
    # for i in range(10):
    #     batch = sequence_generator()

    # print "hello"
    # data = DataRead(name='NYU', phase='test', dsize=(128, 128))
    # data_load = data.loadData()

    data = DataRead(name='ICVL', phase='test', dsize=(128, 128), aug=False)
    data_load = data.loadData()
    sequence_generator = sequenceGenerator(1, 16, len(data_load['depth']), data_load, True, 'test')
    for i in range(100):
        batch = sequence_generator()

        # data = DataRead(name='ICVL', phase='test', dsize=(128, 128))
        # data_load = data.loadData()
        # data = DataRead(name='NYU', phase='train',dsize=(128, 128), baseline=False)
        # data_load = data.loadData()
        # data = DataRead(name='NYU', phase='test', dsize=(128, 128), baseline=False)
        # data_load = data.loadData()
        # data = DataRead(name='ICVL', phase='train',dsize=(128, 128), baseline=False)
        # data_load = data.loadData()
        # data = DataRead(name='ICVL', phase='test', dsize=(128, 128), baseline=False)
        # data_load = data.loadData()
