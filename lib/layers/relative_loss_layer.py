#!/usr/bin/env python

__author__ = 'WuYiming'

import _init_paths
import caffe
import numpy as np
import yaml

class RelativeLossLayer(caffe.Layer):
    
    def relative_loss(self, gt, pred):
        try:
            joint_num, dim = gt.shape
        except:
            raise ValueError("input dimension is not 2")

        assert self.joint_num == joint_num and dim == 3, \
                "dimension is wrong, self.joint_num vs. joint_num = {} {}, dim = {}" \
                .format(self.joint_num, joint_num, dim)

        for i in xrange(joint_num):
            if joint_num == 14: # NYU dataset
                p1, p2, r1, r2, m1, m2, i1, i2, t1, t2, t3, w1, w2, c = pred[:]    
                r_p1 = p1 - p2
                r_p2 = p2 - c
                r_r1 = r1 - r2
                r_r2 = r2 - c
                r_m1 = m1 - m2
                r_m2 = m2 - c
                r_i1 = i1 - i2
                r_i2 = i2 - c
                r_t1 = t1 - t2
                r_t2 = t2 - t3
                r_t3 = t3 - c
                r_w1 = w1 - c
                r_w2 = w2 - c
                r = np.array([r_p1, r_p2, r_r1, r_r2, r_m1, r_m2, r_i1, r_i2, \
                        r_t1, r_t2, r_t3, r_w1, r_w2])
                p1t, p2t, r1t, r2t, m1t, m2t, i1t, i2t, t1t, t2t, t3t, w1t, w2t, ct = gt[:]
                r_p1_t = p1t - p2t
                r_p2_t = p2t - ct
                r_r1_t = r1t - r2t
                r_r2_t = r2t - ct
                r_m1_t = m1t - m2t
                r_m2_t = m2t - ct
                r_i1_t = i1t - i2t
                r_i2_t = i2t - ct
                r_t1_t = t1t - t2t
                r_t2_t = t2t - t3t
                r_t3_t = t3t - ct
                r_w1_t = w1t - ct
                r_w2_t = w2t - ct
                r_t = np.array([r_p1_t, r_p2_t, r_r1_t, r_r2_t, r_m1_t, r_m2_t, r_i1_t, \
                        r_i2_t, r_t1_t, r_t2_t, r_t3_t, r_w1_t, r_w2_t])
                # calculate derivative
                diff = np.array([r_p1 - r_p1_t,
                                -(r_p1 - r_p1_t) + (r_p2 - r_p2_t),
                                r_r1 - r_r1_t,
                                -(r_r1 - r_r1_t) + (r_r2 - r_r2_t),
                                r_m1 - r_m1_t,
                                -(r_m1 - r_m1_t) + (r_m2 - r_m2_t),
                                r_i1 - r_i1_t,
                                -(r_i1 - r_i1_t) + (r_i2 - r_i2_t),
                                r_t1 - r_t1_t,
                                -(r_t1 - r_t1_t) + (r_t2 - r_t2_t),
                                -(r_t2 - r_t2_t) + (r_t3 - r_t3_t),
                                r_w1 - r_w1_t,
                                r_w2 - r_w2_t,
                                -(r_p2 - r_p2_t) - (r_r2 - r_r2_t) - (r_m2 -r_m2_t) - \
                                (r_i2 - r_i2_t) - (r_t3 - r_t3_t) - (r_w1 - r_w1_t) - \
                                (r_w2 - r_w2_t)])
                
            elif joint_num == 16: # ICVL dataset
                raise NotImplementedError 
            else:
                raise Exception("joint_num is not 14 or 16!")
            return (r - r_t).reshape((joint_num-1)*dim), diff.reshape(joint_num*dim)
            

    def setup(self, bottom, top):
        # check bottom size
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance")


    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have same dimension.")

        layer_param = yaml.load(self.param_str)
        self.joint_num = int(layer_param['joint_num'])

        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
    
    def forward(self, bottom, top):
        # split the joint

        pred = bottom[0].data.reshape(bottom[0].num, self.joint_num, 3)
        gt = bottom[1].data.reshape(bottom[1].num, self.joint_num, 3)
      
        self.loss = np.zeros((bottom[0].num, (self.joint_num - 1) * 3))
        for i in xrange(bottom[0].num):
            self.loss[i], self.diff[i] = self.relative_loss(gt[i], pred[i])
        top[0].data[...] = np.sum(self.loss**2) / bottom[0].num / 2

    def backward(self, top, propagate_down, bottom):
        for i in xrange(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
