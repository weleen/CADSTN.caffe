#!/usr/bin/env python
"""
augment the dataset, scale, translation, rotation, flip.
"""

import cv2
import numpy as np


def flip(data):
    """
    flip the dataset
    :param data: dict, keys: com, inds, config, depth, joint
    :return: extended dataset
    """
    pass


def rotation(data):
    """
    rotate the depth image
    :param data: dict
    :return:
    """
    pass

def translation(data):
    """
    translate the depth image
    :param data: dict
    :return:
    """
    pass

def rad2deg(rad):
    """
    radians to degree
    :param rad:
    :return: degree
    """
    return rad * 180.0 /np.pi

def deg2rad(deg):
    """
    degree to radians
    :param deg:
    :return: radians
    """
    return deg * np.pi / 180.0

