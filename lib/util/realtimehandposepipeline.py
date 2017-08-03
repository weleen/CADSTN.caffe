"""Class for running the handpose estimation pipeline in realtime.

RealtimeHandposePipeline provides interface for running the pose estimation.
It is made of detection, image cropping and further pose estimation.

Copyright 2015 Markus Oberweger, ICG,
Graz University of Technology <oberweger@icg.tugraz.at>

This file is part of DeepPrior.

DeepPrior is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DeepPrior is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with DeepPrior.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
sys.path.insert(0, '../../caffe/python')
import caffe
from multiprocessing import Process, Queue, Value
import cv2
import time
import numpy
from handdetector import HandDetector

class RealtimeHandposePipeline(object):
    """
    Realtime pipeline for handpose estimation
    """

    # states of pipeline
    STATE_IDLE = 0
    STATE_INIT = 1
    STATE_RUN = 2

    # different hands
    HAND_LEFT = 0
    HAND_RIGHT = 1

    def __init__(self, di, config, netPath, netWeight):
        """
        Initialize data
        :param poseNet: network for pose estimation
        :param config: configuration
        :param di: depth importer
        :param comrefNet: refinement network from center of mass detection
        :return: None
        """

        # handpose CNN
        self.importer = di
        # configuration
        self.config = config
        self.initialconfig = config
        # synchronization between threads
        self.stop = Value('b', False)
        # for calculating FPS
        self.lastshow = time.time()
        # hand left/right
        self.hand = self.HAND_LEFT
        # initial state
        self.state = self.STATE_IDLE
        # hand size estimation
        self.handsizes = []
        self.numinitframes = 50
        # hand tracking or detection
        self.tracking = False
        self.lastcom = (0, 0, 0)

        # caffe net init
        self.net = caffe.Net(netPath, netWeight, caffe.TEST)
        caffe.set_mode_gpu()
        caffe.set_device(0)
        self.frameSize = self.net.blobs['depth'].shape[0]

    def processFiles(self, filenames):
        """
        Run detector from files
        :param filenames: filenames to load
        :return: None
        """
        #initilize cv2.VideoWriter
        fourcc = cv2.cv.CV_FOURCC('I','4','2','0')
        writer = cv2.VideoWriter('res.avi', fourcc, 25.0, (640, 480))

        allstart = time.time()
        if not isinstance(filenames, list):
            raise ValueError("Files must be list of filenames.")

        ind = 0
        result = numpy.array([None] * len(filenames))
        cm = numpy.ones((self.frameSize, 1, 1, 1))
        cm[0] = 0.
        # sequence size
        for i in range(numpy.int(numpy.ceil(len(filenames) / float(self.frameSize)))):
            if self.stop.value:
                break
                
            start = time.time()
            frames = numpy.array([None] * self.frameSize)
            for j in xrange(self.frameSize): # frame size
                if ind < len(filenames):
                    f = filenames[ind]
                    ind += 1
                else:
                    f = filenames[ind - 1]
                print('load {}'.format(f))
                frames[j] = self.importer.loadDepthMap(f)
            print('{}ms load {} frames'.format((time.time() - start)*1000., self.frameSize))
                              
            started = time.time()
            crop = []
            M = []
            com3D = []
            for j in xrange(self.frameSize):
                cropj, Mj, com3Dj = self.detect(frames[j].copy())
                crop.append(cropj)
                com3D.append(com3Dj)
                M.append(Mj)
            print('{}ms detect {} frames'.format((time.time() - started)*1000., self.frameSize))
                              

            crop = numpy.array(crop)
            crop.shape = (crop.shape[0], 1, crop.shape[1], crop.shape[2])
            com3D = numpy.array(com3D)
            com3D.shape = (com3D.shape[0], 1, com3D.shape[1])
            startp = time.time()
            self.net.blobs['depth'].data[...] = crop
            self.net.blobs['clip_markers'].data[...] = cm.reshape(16)
            self.net.forward()
            pred = self.net.blobs['joint_pred_mix'].data
            print pred.shape
            print com3D.shape
            print self.config['cube'][2]
            poses = pred.reshape(pred.shape[0], pred.shape[1] / com3D.shape[2], com3D.shape[2]) * self.config['cube'][2] / 2 + com3D
            assert poses.shape[0] == self.frameSize, 'size mismatch! {} not equal to {}'.format(poses.shape[0], self.frameSize)
            print('{}ms pose {} frames'.format((time.time() - startp)*1000., self.frameSize))
                              
            # Display the resulting frame
            for j in xrange(self.frameSize):
                starts = time.time()
                img = self.show(frames[j], poses[j])
                writer.write(img)
                img = self.addStatusBar(img)
                cv2.imshow('frame', img)
                self.lastshow = time.time()
                cv2.imshow('crop', crop[j][0])
                self.processKey(cv2.waitKey(1) & 0xFF)
                print("{}ms display".format((time.time() - starts)*1000.))

            print("-> {}ms per {} frames".format((time.time() - start)*1000., self.frameSize))
        writer.release()
        print("DONE in {}s".format((time.time() - allstart)))
        cv2.destroyAllWindows()

    def detect(self, frame):
        """
        Detect the hand
        :param frame: image frame
        :return: cropped image, transformation, center
        """

        hd = HandDetector(frame, self.config['fx'], self.config['fy'], importer=self.importer)
        doHS = (self.state == self.STATE_INIT)
        if self.tracking and not numpy.allclose(self.lastcom, 0):
            loc, handsz = hd.track(self.lastcom, self.config['cube'], doHandSize=doHS)
        else:
            loc, handsz = hd.detect(size=self.config['cube'], doHandSize=doHS)

        self.lastcom = loc

        if self.state == self.STATE_INIT:
            self.handsizes.append(handsz)
            print numpy.median(numpy.asarray(self.handsizes), axis=0)
        else:
            self.handsizes = []

        if self.state == self.STATE_INIT and len(self.handsizes) >= self.numinitframes:
            self.config['cube'] = tuple(numpy.median(numpy.asarray(self.handsizes), axis=0).astype('int'))
            self.state = self.STATE_RUN
            self.handsizes = []

        if numpy.allclose(loc, 0):
            return numpy.zeros((self.net.blobs['depth'].data.shape[2], self.net.blobs['depth'].data.shape[3]), dtype='float32'), numpy.eye(3), loc
        else:
            crop, M, com = hd.cropArea3D(loc, size=self.config['cube'], dsize=(self.net.blobs['depth'].data.shape[2], self.net.blobs['depth'].data.shape[3]))
            com3D = self.importer.jointImgTo3D(com)
            crop[crop == 0] = com3D[2] + (self.config['cube'][2] / 2.)
            crop[crop >= com3D[2] + (self.config['cube'][2] / 2.)] = com3D[2] + (self.config['cube'][2] / 2.)
            crop[crop <= com3D[2] - (self.config['cube'][2] / 2.)] = com3D[2] - (self.config['cube'][2] / 2.)
            crop -= com3D[2]
            crop /= (self.config['cube'][2] / 2.)
            return crop, M, com3D

    def estimatePose(self, crop):
        """
        Estimate the hand pose
        :param crop: cropped hand depth map
        :return: joint positions
        """

        # mirror hand if left/right changed
        if self.hand == self.HAND_LEFT:
            inp = crop[None, None, :, :]
        else:
            inp = crop[None, None, :, ::-1]

        jts = self.poseNet.computeOutput(inp)

        # mirror pose if left/right changed
        if self.hand == self.HAND_LEFT:
            return jts[0].reshape(self.poseNet.cfgParams.numJoints, 3)
        else:
            jj = jts[0].reshape(self.poseNet.cfgParams.numJoints, 3)
            # mirror coordinates
            jj[:, 0] *= (-1.)
            return jj

    def show(self, frame, pose):
        """
        Show depth with overlayed joints
        :param frame: depth frame
        :param pose: joint positions
        :return: image
        """

        # plot depth image with annotations
        imgcopy = frame.copy()
        # display hack to hide nd depth
        msk = numpy.logical_and(32001 > imgcopy, imgcopy > 0)
        msk2 = numpy.logical_or(imgcopy == 0, imgcopy == 32001)
        min = imgcopy[msk].min()
        max = imgcopy[msk].max()
        imgcopy = (imgcopy - min) / (max - min) * 255.
        imgcopy[msk2] = 255.
        imgcopy = imgcopy.astype('uint8')
        imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_GRAY2BGR)

        jtI = self.importer.joints3DToImg(pose)
        for i in range(jtI.shape[0]):
            cv2.circle(imgcopy, (jtI[i, 0], jtI[i, 1]), 3, (255, 0, 0), -1)

        import matplotlib
        if pose.shape[0] == 16:
            jointConnections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [0, 10],
                                 [10, 11], [11, 12], [0, 13], [13, 14], [14, 15]]
            jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1]]]))[0, 0]]
        elif pose.shape[0] == 14:
            jointConnections = [[13, 1], [1, 0], [13, 3], [3, 2], [13, 5], [5, 4], [13, 7], [7, 6], [13, 10],
                                     [10, 9], [9, 8], [13, 11], [13, 12]]
            jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 1]]]))[0, 0]]
        else:
            raise ValueError("Invalid number of joints")

        for i in range(len(jointConnections)):
            cv2.line(imgcopy, (jtI[jointConnections[i][0], 0], jtI[jointConnections[i][0], 1]),
                     (jtI[jointConnections[i][1], 0], jtI[jointConnections[i][1], 1]), 255.*jointConnectionColors[i], 2)

        return imgcopy

    def addStatusBar(self, img):
        """
        Add status bar to image
        :param img: image
        :return: image with status bar
        """
        barsz = 20
        retimg = numpy.ones((img.shape[0]+barsz, img.shape[1], img.shape[2]), dtype='uint8')*255

        retimg[barsz:img.shape[0]+barsz, 0:img.shape[1], :] = img

        # FPS text
        fps = 1./(time.time()-self.lastshow)
        cv2.putText(retimg, "FPS {0:2.2f}".format(fps), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # hand text
        cv2.putText(retimg, "Left" if self.hand == self.HAND_LEFT else "Right", (80, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # hand size
        cv2.putText(retimg, "({0:d}, {1:d}, {2:d})".format(self.config['cube'][0], self.config['cube'][1], self.config['cube'][2]), (120, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # hand tracking mode, tracking or detection
        cv2.putText(retimg, "T" if self.tracking else "D", (200, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # status symbol
        if self.state == self.STATE_IDLE:
            col = (0, 0, 255)
        elif self.state == self.STATE_INIT:
            col = (0, 255, 255)
        elif self.state == self.STATE_RUN:
            col = (0, 255, 0)
        else:
            col = (0, 0, 255)
        cv2.circle(retimg, (5, 5), 5, col, -1)
        return retimg

    def processKey(self, key):
        """
        Process key
        :param key: key value
        :return: None
        """

        if key == ord('q'):
            self.stop.value = True
        elif key == ord('h'):
            if self.hand == self.HAND_LEFT:
                self.hand = self.HAND_RIGHT
            else:
                self.hand = self.HAND_LEFT
        elif key == ord('+'):
            lst = list(self.config['cube'])
            lst[0] += 10
            lst[1] += 10
            lst[2] += 10
            self.config['cube'] = tuple(lst)
        elif key == ord('-'):
            lst = list(self.config['cube'])
            lst[0] -= 10
            lst[1] -= 10
            lst[2] -= 10
            self.config['cube'] = tuple(lst)
        elif key == ord('r'):
            self.reset()
        elif key == ord('i'):
            self.state = self.STATE_INIT
        elif key == ord('t'):
            self.tracking = not self.tracking
        else:
            pass

    def reset(self):
        """
        Reset stateful parts
        :return: None
        """
        self.state = self.STATE_IDLE
        self.config = self.initialconfig
