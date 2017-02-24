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

from util.handpose_evaluation import NYUHandposeEvaluation,ICVLHandposeEvaluation
from data.importers import NYUImporter,ICVLImporter

DEBUG = True

def loadGt(gt_file):
    """
    load ground truth joint
    loadGt(str) -> np.array
    :param gt_file: path to ground truth file
    :return: joint in xyz coordinate
    """
    #gt_file = '/mnt/data/NYU-Hands-v2/test/joint_data.mat'
    data = scio.loadmat(gt_file)
    kinect_index = 0
    joint_uvd = data['joint_uvd'][kinect_index, :, :, :]
    joint_xyz = data['joint_xyz'][kinect_index, :, :, :]
    if 'NYU' in gt_file:
        restrictedJoint = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
        joint_name = data['joint_names'].reshape(36, 1)
        return joint_xyz[:, restrictedJoint]

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

def predictJoints(model_name, weights_num, store=True, dataset='NYU', gpu_or_cpu='gpu'):
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
    frame_size, seq_size, joint_size = net.blobs['pred_joint'].data.shape
    dim = 3 # estimate 3 dimension x, y and z

    # recognize different dataset
    if dataset == 'NYU':
        test_num = 8252
    elif dataset == 'ICVL':
        test_num = 0
    else:
        assert 0, 'unknow dataset {}'.format(dataset)

    if gpu_or_cpu == 'gpu':
        caffe.set_mode_gpu()
        caffe.set_device(0)

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
    predicted_joints = np.array([None] * test_num)

    t_start = time.time()
    for i in xrange(np.int(np.ceil(float(test_num) / (frame_size * seq_size)))):
        net.forward()
        print 'test iter = ', i
        for j, ind in enumerate(net.blobs['inds'].data):
            row = j / seq_size
            col = j % seq_size
            if predicted_joints[int(ind) - 1] == None:  # add this sentence make run slow
                if model_name == 'baseline':
                    if ind <= 2440:
                        predicted_joints[int(ind) - 1] = (net.blobs['joint_pred'].data[j].reshape(14, 3) * \
                                                          300 / 2 + net.blobs['com'].data[j].reshape(1, 3))
                    else:
                        predicted_joints[int(ind) - 1] = (net.blobs['joint_pred'].data[j].reshape(14, 3) * \
                                                          300 * 0.87 / 2 + net.blobs['com'].data[j].reshape(1, 3))
                else:
                    predicted_joints[int(ind) - 1] = \
                        (net.blobs['pred_joint'].data[row][col].reshape(joint_size / dim, dim) \
                        * net.blobs['config'].data[j][0] / 2 \
                        + net.blobs['com'].data[j].reshape(1, 3)).copy()
    t_end = time.time()
    print 'time elapse {}'.format((t_end - t_start) / test_num)

    if store:
        print 'write the result in {}'.format(file_name)
        with open(file_name, 'w') as f:
            for i in xrange(predicted_joints.shape[0]):
                for item in predicted_joints[i].reshape(14 * 3):
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
    gt_file = '../dataset/NYU/test/joint_data.mat'
    gt3D = loadGt(gt_file)

    if DEBUG:
        print 'gt3D.shape = ', gt3D.shape

    # predict joint by ourselves in xyz coordinate
    model = 'lstm_small_frame_size'
    weight_num = '200000'
    joints, file_name = predictJoints(model, weight_num)

    eval_prefix = 'NYU_' + model + '_' + weight_num
    if not os.path.exists('../eval/'+eval_prefix+'/'):
        os.makedirs('../eval/'+eval_prefix+'/')

    if DEBUG:
        print 'joints.shape = ', joints.shape
        print 'joints[0] = ', joints[0]
        print 'type(joints[0]) = ', type(joints[0])
        print 'type(joints[0][0] = ', type(joints[0][0])

    hpe = NYUHandposeEvaluation(gt3D, joints)
    hpe.subfolder += eval_prefix+'/'
    mean_error = hpe.getMeanError()
    max_error = hpe.getMaxError()
    #print("Train samples: {}, test samples: {}".format(train_data.shape[0], len(gt3D)))
    print("Mean error: {}mm, max error: {}mm".format(mean_error, max_error))
    print("MD score: {}".format(hpe.getMDscore(80)))

    print("{}".format([hpe.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    #################################
    # BASELINE
    # Load the evaluation
    di = NYUImporter('../dataset/NYU/')
    data_baseline = di.loadBaseline('../dataset/NYU/test/test_predictions.mat', np.asarray(gt3D))

    hpe_base = NYUHandposeEvaluation(gt3D, data_baseline)
    hpe_base.subfolder += eval_prefix+'/'
    print("Mean error: {}mm".format(hpe_base.getMeanError()))

    hpe.plotEvaluation(eval_prefix, methodName='Our lstm',baseline=[('Tompson et al.',hpe_base)])

    # ind = 0
    # for i in testSeqs[0].data:
    #     if ind % 20 != 0:
    #         ind += 1
    #         continue
    #     jt = joints[ind]
    #     jtI = di.joints3DToImg(jt)
    #     for joint in range(jt.shape[0]):
    #         t=transformPoint2D(jtI[joint], i.T)
    #         jtI[joint, 0] = t[0]
    #         jtI[joint, 1] = t[1]
    #     hpe.plotResult(i.dpt, i.gtcrop, jtI, "{}_{}".format(eval_prefix, ind))
    #     ind+=1