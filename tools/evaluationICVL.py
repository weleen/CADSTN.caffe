#!/usr/bin/env python
"""
Evaluate the prediction
"""

from _init_paths import *
import caffe
import matplotlib
import numpy as np
matplotlib.use('Agg')  # plot to file
import matplotlib.pyplot as plt
import os
import time
import scipy.io as scio
from data_layer.data_input_layer import *
from data.transformations import transformPoint2D
from util.handpose_evaluation import ICVLHandposeEvaluation
from data.importers import ICVLImporter, DepthImporter

DEBUG = False

fx, fy, ux, uy = 241.42, 241.42, 160, 120
def jointsImgTo3D(sample):
    """
    Normalize sample to metric 3D
    :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
    :return: normalized joints in mm
    """
    ret = np.zeros((sample.shape[0], 3), np.float32)
    for i in range(sample.shape[0]):
        ret[i] = jointImgTo3D(sample[i])
    return ret


def jointImgTo3D(sample):
    """
    Normalize sample to metric 3D
    :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
    :return: normalized joints in mm
    """
    ret = np.zeros((3,), np.float32)
    # convert to metric using f
    ret[0] = (sample[0] - ux) * sample[2] / fx
    ret[1] = (sample[1] - uy) * sample[2] / fy
    ret[2] = sample[2]
    return ret

def loadGt(gt_file):
    """
    load ground truth joint
    loadGt(str) -> np.array
    :param gt_file: path to ground truth file
    :return: joint in xyz coordinate
    """
    #gt_file = '../dataset/ICVL/test_seq_1.txt'
    gt = []
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            gt.append(map(float, line.split(' ')[1:-1]))
    gt = np.array(gt)
    gt3D = []
    for i in xrange(gt.shape[0]):
        gt3D.append(jointsImgTo3D(gt[i].reshape(16, 3)))
    gt3D = np.array(gt3D)
    return gt3D

def loadPredFile(filepath, estimation_mode='uvd'):
    """
    load the prediction file
    loadPredFile(str, str) -> np.array

    :param filepath: prediction file path
    :param estimation_mode: the coordinat of joint, e.g. uvd or xyz
    :return: prediction joints
    """
    assert os.path.isfile(filepath), "{} is not a file!".format(filepath)
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            lines[index] = map(float, line.split())
        joints = np.array(lines)
        n, d = joints.shape
        dim = 3
        return joints.reshape(n, d/dim, dim)

def predictJoints(model_name, weights_num, store=True, dataset='ICVL', gpu_or_cpu='gpu'):
    """
    predict joints in xyz coordinate
    predictJoints(str, str, bool, str, str) -> (np.array, str)

    :param model_name: prototxt name e.g. lstm
    :param weights_num: caffemodel number e.g. 160000
    :param store: store the predicted joints in file
    :return: predicted_joints: predicted joints
             file_name: file name store the joints
    """
    model_def = '../models/' + dataset + '/hand_' + model_name + '/hand_' + model_name + '.prototxt'
    model_weights = '../weights/' + dataset + '/hand_' + model_name + '/hand_' + model_name + '_iter_' + weights_num + '.caffemodel'

    assert os.path.isfile(model_def), '{} is not a file!'.foramt(model_def)
    assert os.path.isfile(model_weights), '{} is not a file!'.format(model_weights)

    print 'load prototxt from {}'.format(model_def)
    print 'load weights from {}'.format(model_weights)

    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # extract seq_size (video num), frame_size (frames in video) and joint_size (dimension need to regress) from the blob
    if model_name == 'baseline':
        frame_size, joint_size = net.blobs['joint_pred'].data.shape
        seq_size = 1
    else:
        frame_size, seq_size, joint_size = net.blobs['pred_joint'].data.shape
    dim = 3 # estimate 3 dimension x, y and z

    # recognize different dataset
    if dataset == 'ICVL':
        test_num = 702
    else:
        assert 0, 'unknow dataset {}'.format(dataset)

    if gpu_or_cpu == 'gpu':
        caffe.set_device(0)
        caffe.set_mode_gpu()


    if DEBUG:
        print 'frame_size = ', frame_size
        print 'seq_size = ', seq_size
        print 'joint_size = ', joint_size / dim
        print 'dim = ', dim
        print 'using {} to run {} test dataset'.format(gpu_or_cpu, dataset)

    if store:
        # store the predicted xyz into files
        file_name = '../result/OURS/' + dataset + '/hand_' + model_name + '_' + weights_num + '.txt'

        if os.path.isfile(file_name):
            print '{} exists, read file directly.'.format(file_name)
            return loadPredFile(file_name), file_name

    # calculate the predicted joints in xyz coordinate
    predicted_joints = np.zeros((test_num, joint_size / dim, dim))

    t_start = time.time()
    for i in xrange(np.int(np.ceil(float(test_num) / (frame_size * seq_size)))):
        net.forward()
        print 'test iter = ', i
        for j, ind in enumerate(net.blobs['inds'].data):
            row = j / seq_size
            col = j % seq_size
            if model_name == 'baseline':
                predicted_joints[int(ind)] = (net.blobs['joint_pred'].data[j].reshape(joint_size / dim, dim) * \
                                              125 + net.blobs['com'].data[j].reshape(1, 3)).copy()
            else:
                predicted_joints[int(ind)] = (net.blobs['pred_joint'].data[row][col].reshape(joint_size / dim, dim) \
                                              * net.blobs['config'].data[j][0] / 2 \
                                              + net.blobs['com'].data[j].reshape(1, 3)).copy()
    t_end = time.time()
    print 'time elapse {}'.format((t_end - t_start) / test_num)

    if store:
        print 'write the result in {}'.format(file_name)
        with open(file_name, 'w') as f:
            for i in xrange(predicted_joints.shape[0]):
                for item in predicted_joints[i].reshape(joint_size):
                    f.write("%s " % item)
                f.write("\n")
        predicted_joints = loadPredFile(file_name)
    else:
        # predicted_joints is inited by [None], so we must assign the variable again to
        # get the right shape
        tmp = np.zeros((test_num, joint_size / dim, dim))
        tmp = predicted_joints
        predicted_joints = tmp
        print predicted_joints.shape
        file_name = None

    return predicted_joints, file_name


if __name__ == '__main__':

    # test NYU dataset
    gt_file = '../dataset/ICVL/test_seq_1.txt'
    gt3D = loadGt(gt_file)

    if DEBUG:
        print 'gt3D.shape = ', gt3D.shape

    model = []
    weight_num = []
    pred_joints = []
    hpe = []
    eval_prefix = []
    # predict joint by ourselves in xyz coordinate
    model.append('baseline')
    #model.append('lstm_small_frame_size_no_concate')
    model.append('lstm_small_frame_size')
    weight_num.append('150000')
    #weight_num.append('200000')
    weight_num.append('200000')
    assert len(model) == len(weight_num), 'length is not equal!'
    assert len(model) == len(weight_num), 'length is not equal!'

    for ind in xrange(len(model)):
        joints, file_name = predictJoints(model[ind], weight_num[ind])
        pred_joints.append(joints)
        eval_prefix.append('ICVL_' + model[ind] + '_' + weight_num[ind])
        if not os.path.exists('../eval/'+eval_prefix[ind]+'/'):
            os.makedirs('../eval/'+eval_prefix[ind]+'/')

        if DEBUG:
            print 'joints.shape = ', joints.shape
            print 'joints[0] = ', joints[0]
            print 'type(joints[0]) = ', type(joints[0])
            print 'type(joints[0][0] = ', type(joints[0][0])

        hpe.append(ICVLHandposeEvaluation(gt3D, joints))
        hpe[ind].subfolder += eval_prefix[ind]+'/'
        mean_error = hpe[ind].getMeanError()
        max_error = hpe[ind].getMaxError()
        print("Test on {}_{}".format(model[ind], weight_num[ind]))
        print("Mean error: {}mm, max error: {}mm".format(mean_error, max_error))
        print("MD score: {}".format(hpe[ind].getMDscore(80)))

        print("{}".format([hpe[ind].getJointMeanError(j) for j in range(joints[0].shape[0])]))
        print("{}".format([hpe[ind].getJointMaxError(j) for j in range(joints[0].shape[0])]))

    print "Testing baseline"
    #################################
    # BASELINE
    # Load the evaluation
    di = ICVLImporter('../dataset/ICVL/', cacheDir='../dataset/cache/')
    data_baseline = di.loadBaseline('../dataset/ICVL/Results/LRF_Results_seq_1.txt')

    hpe_base = ICVLHandposeEvaluation(gt3D, data_baseline)
    hpe_base.subfolder += eval_prefix[0]+'/'
    print("Mean error: {}mm".format(hpe_base.getMeanError()))

    plot_list = zip(model, hpe)
    hpe_base.plotEvaluation(eval_prefix[0], methodName='Tang et al.', baseline=plot_list)

    Seq2_1 = di.loadSequence('test_seq_1')
    Seq2_2 = di.loadSequence('test_seq_2')
    testSeqs = [Seq2_1, Seq2_2]

    for index in xrange(len(hpe)):
        ind = 0
        for i in testSeqs[0].data:
            if ind % 20 != 0:
                ind += 1
                continue
            jt = pred_joints[index][ind]
            jtI = di.joints3DToImg(jt)
            for joint in range(jt.shape[0]):
                t=transformPoint2D(jtI[joint], i.T)
                jtI[joint, 0] = t[0]
                jtI[joint, 1] = t[1]
            hpe[index].plotResult(i.dpt, i.gtcrop, jtI, "{}_{}".format(eval_prefix[index], ind))
            ind+=1