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
from layers.data_input_layer import *
from data.transformations import transformPoint2D
from util.handpose_evaluation import NYUHandposeEvaluation
from data.importers import NYUImporter
from data.dataset import NYUDataset

DEBUG = False

# @deprected
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

def predictJoints(model, store=True, dataset='NYU', gpu_or_cpu='gpu'):
    """
    predict joints in xyz coordinate
    predictJoints(str, str, bool, str, str) -> (np.array, str)

    :param model_name: prototxt name e.g. lstm
    :param weights_num: caffemodel number e.g. 160000
    :param store: store the predicted joints in file
    :return: predicted_joints: predicted joints
             file_name: file name store the joints
    """
    model_name = model[0]
    weights_num = model[1]
    model_def = '../models/' + dataset + '/hand_' + model_name + '/hand_' + model_name + '.prototxt'
    model_weights = '../weights/' + dataset + '/hand_' + model_name + '/hand_' + model_name + '_iter_' + weights_num + '.caffemodel'

    assert os.path.isfile(model_def), '{} is not a file!'.format(model_def)
    assert os.path.isfile(model_weights), '{} is not a file!'.format(model_weights)

    print 'load prototxt from {}'.format(model_def)
    print 'load weights from {}'.format(model_weights)

    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # extract seq_size (video num), frame_size (frames in video) and joint_size (dimension need to regress) from the blob
    if 'baseline' in model_name:
        frame_size, joint_size = net.blobs['joint_pred_baseline'].data.shape
        seq_size = 1
    elif '3D' in model_name:
        if '3D_and_depth' in model_name:
            frame_size, joint_size = net.blobs['joint_pred_depth_3D'].data.shape
        else:
            frame_size, joint_size = net.blobs['joint_pred_3D'].data.shape
        seq_size = 1
    elif 'mix' in model_name:
        frame_size, joint_size = net.blobs['joint_pred_mix'].data.shape
        seq_size = 1
    else: # lstm
        frame_size, seq_size, joint_size = net.blobs['pred_joint_lstm'].data.shape
    dim = 3 # estimate 3 dimension x, y and z

    # recognize different dataset
    if dataset == 'NYU':
        test_num = 8252
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
    predicted_joints = np.zeros((test_num, joint_size / dim, dim))

    t_start = time.time()
    for i in xrange(np.int(np.ceil(float(test_num) / (frame_size * seq_size)))):
        net.forward()
        print 'test iter = ', i
        for j, ind in enumerate(net.blobs['inds'].data):
            row = j / seq_size
            col = j % seq_size
            if 'baseline' in model_name:
                predicted_joints[int(ind) - 1] = (net.blobs['joint_pred_baseline'].data[j].reshape(joint_size / dim, dim) * \
                                                      net.blobs['config'].data[j][0] / 2 + net.blobs['com'].data[j].reshape(1, dim))
            elif '3D' in model_name:
                if '3D_and_depth' in model_name:
                    predicted_joints[int(ind) - 1] = (net.blobs['joint_pred_depth_3D'].data[j].reshape(joint_size / dim, dim) * \
                                                      net.blobs['config'].data[j][0] / 2 + net.blobs['com'].data[j].reshape(1, dim))
                else:
                    predicted_joints[int(ind) - 1] = (net.blobs['joint_pred_3D'].data[j].reshape(joint_size / dim, dim) * \
                                                      net.blobs['config'].data[j][0] / 2 + net.blobs['com'].data[j].reshape(1, dim))
            elif 'mix' in model_name:
                predicted_joints[int(ind) - 1] = (net.blobs['joint_pred_mix'].data[j].reshape(joint_size / dim, dim) * \
                                                      net.blobs['config'].data[j][0] / 2 + net.blobs['com'].data[j].reshape(1, dim))
            else:
                predicted_joints[int(ind) - 1] = \
                    (net.blobs['pred_joint_lstm'].data[row][col].reshape(joint_size / dim, dim) \
                    * net.blobs['config'].data[j][0] / 2 \
                    + net.blobs['com'].data[j].reshape(1, dim)).copy()
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

    return predicted_joints, file_name


if __name__ == '__main__':
    # ablation or comparison
    ABLATION = False # ablation or comparison
    # test NYU dataset
    di = NYUImporter('../dataset/NYU/', cacheDir='../dataset/cache/')
    gt3D = []
    Seq2_1 = di.loadSequence('test_1',docom=True)
    Seq2_2 = di.loadSequence('test_2',docom=True)
    testSeqs = [Seq2_1, Seq2_2]
    for seq in testSeqs:
        gt3D.extend([j.gt3Dorig for j in seq.data])

    gt3D = np.array(gt3D)

    if DEBUG:
        print 'gt3D.shape = ', gt3D.shape

    model = []
    weight_num = []
    pred_joints = []
    hpe = []
    eval_prefix = []
    # predict joint by ourselves in xyz coordinate
    if ABLATION == True:
        model.append(('baseline','100000'))
        #model.append(('3D', '100000'))
        model.append(('3D_and_depth', '100000')) 
        model.append(('lstm','100000')) 
        model.append(('mix', '100000'))
    else:
        model.append(('mix', '100000'))

    for ind in xrange(len(model)):
        joints, file_name= predictJoints(model[ind])

        pred_joints.append(joints)
        eval_prefix.append('NYU_' + model[ind][0] + '_' + model[ind][1])
        if not os.path.exists('../eval/'+eval_prefix[ind]+'/'):
            os.makedirs('../eval/'+eval_prefix[ind]+'/')

        if DEBUG:
            print 'joints.shape = ', joints.shape
            print 'joints[0] = ', joints[0]

        hpe.append(NYUHandposeEvaluation(gt3D, joints))
        hpe[ind].subfolder += eval_prefix[ind]+'/'
        mean_error = hpe[ind].getMeanError()
        max_error = hpe[ind].getMaxError()
        print("Test on {}_{}".format(model[ind][0], model[ind][1]))
        print("Mean error: {}mm, max error: {}mm".format(mean_error, max_error))
        print("MD score: {}".format(hpe[ind].getMDscore(80)))

        print("{}".format([hpe[ind].getJointMeanError(j) for j in range(joints[0].shape[0])]))
        print("{}".format([hpe[ind].getJointMaxError(j) for j in range(joints[0].shape[0])]))

    #################################
    # BASELINE
    # Load the evaluation
    if ABLATION:
        # plot list
        plot_list = zip(['Basic', 'Spatial', 'Temporal', 'Final'], hpe)

        # plot ablation result
        # NOTE: figure is in mix_100000 folder
        hpe[-1].plotEvaluation('ablation', methodName='Final', baseline=plot_list[:-1])
    else:
        # plot list
        plot_list = []

        # Comparison
        # siggraph
        data_siggrpah = di.loadBaseline('../dataset/NYU/test/test_predictions.mat', np.asarray(gt3D))

        hpe_siggrpah = NYUHandposeEvaluation(gt3D, data_siggrpah)
        hpe_siggrpah.subfolder += 'comparison/'
        print("Tompson et al. Siggraph 2014")
        print("Mean error: {}mm".format(hpe_siggrpah.getMeanError()))
        plot_list.append(('HeatMap', hpe_siggrpah))

        # cvww15 deeppriori
        data_deeppriori = di.loadBaseline('../result/CVWW15/CVWW15_NYU_Prior-Refinement.txt')
        hpe_deeppriori = NYUHandposeEvaluation(gt3D, data_deeppriori)
        hpe_deeppriori.subfolder += 'comparison/'
        print("Oberweger et al. CVWW 2015")
        print("Mean error: {}mm".format(hpe_deeppriori.getMeanError()))
        plot_list.append(('DeepPrior', hpe_deeppriori))

        # iccv15 deep loopback
        data_loopback = di.loadBaseline('../result/ICCV15/ICCV15_NYU_Feedback.txt')
        hpe_loopback = NYUHandposeEvaluation(gt3D, data_loopback)
        hpe_loopback.subfolder += 'comparison/'
        print("Oberweger et al. ICCV 2015")
        print("Mean error: {}mm".format(hpe_loopback.getMeanError()))
        plot_list.append(('Feedback', hpe_loopback))
        
        # ijcai16 deepmodel
        data_deepmodel = di.loadBaseline('../result/IJCAI16/IJCAI16_NYU.txt')
        hpe_deepmodel = NYUHandposeEvaluation(gt3D, data_deepmodel)
        hpe_deepmodel.subfolder += 'comparison/'
        print("Zhou et al. IJCAI 2016")
        print("Mean error: {}mm".format(hpe_deepmodel.getMeanError()))
        plot_list.append(('DeepModel', hpe_deepmodel))
        
        # ijcv lie-x 
        data_lie = di.loadBaseline('../result/IJCV16/IJCV16_lie_hand_jnts_estm_result_uvd.txt')
        hpe_lie = NYUHandposeEvaluation(gt3D, data_lie)
        hpe_lie.subfolder += 'comparison/'
        print("Xu et al. IJCV 2017")
        print("Mean error: {}mm".format(hpe_lie.getMeanError()))
        plot_list.append(('Lie-X', hpe_lie))

        # plot comparison result
        # NOTE: figure is in mix_100000 folder
        hpe[0].plotEvaluation('comparison', methodName='Ours', baseline=plot_list)

    # sample image
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
            hpe[index].plotResult(i.dpt, i.gtcrop, jtI, "{}_{}".format(eval_prefix[index], ind), showGT=False)
            ind+=1
