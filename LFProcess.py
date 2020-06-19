__author__ = 'Eisa Hedayati'

import os
import math
import numpy as np
import cv2
import scipy as sp


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('Successfully created: ' + directory)
        return 0

    print(directory + ' already existed')
    return 0


def must_dir(directory):
    assert os.path.exists(directory), 'Configuration ' + directory + ' does not exist'


def must_file(fileName):
    assert os.path.isfile(fileName), 'Configuration ' + fileName + ' does not exist'


def readConfig(confPath, lookAtPath):
    configFile = open(confPath, 'r')
    config = configFile.read().split(',')
    configFile.close()
    lookAtFile = open(lookAtPath, 'r')
    lookAt = lookAtFile.read().split(',')
    lookAtFile.close()
    aperture = float(config[0])
    focalDist = float(config[1])
    camArray = int(config[2]), int(config[3])

    # calculating camera distance to the focus point
    p1 = [float(lookAt[0]), float(lookAt[1]), float(lookAt[2])]
    p2 = [float(lookAt[3]), float(lookAt[4]), float(lookAt[5])]
    dist = distance(p1, p2)
    microLensArray = int(config[4]), int(config[5])

    return focalDist, aperture, camArray, microLensArray, dist


def distance(p1, p2):
    dist = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) +
                     (p1[1] - p2[1]) * (p1[1] - p2[1]) +
                     (p1[2] - p2[2]) * (p1[2] - p2[2])
                     )
    return dist


# Read all images taken by cameras in the array and converts them to a single
# (or sliced in case its too big) light field
def readImages(dirPath, camGrid, sensorDim, lfRawPath, useHDD):
    if useHDD:
        subLF = np.zeros([sensorDim[0], sensorDim[1], camGrid[1], 3], np.dtype(np.uint8))
    else:
        LF = np.zeros([sensorDim[0], sensorDim[1], camGrid[0], camGrid[1], 3], np.dtype(np.uint8))

    subName = lfRawPath + 'subLF_'

    for i in range(0, camGrid[0]):
        for j in range(0, camGrid[1]):
            imPath = dirPath + str(i) + '_' + str(j) + '.png'
            im = cv2.imread(imPath)
            if useHDD:
                subLF[:, :, j, :] = im
            else:
                LF[:, :, i, j, :] = im
        if useHDD:
            np.save(subName + str(i) + '.npy', subLF)

    if not useHDD:
        lfName = lfRawPath + 'LF.npy'
        np.save(lfName, LF)
        print ('Success in creating single file')
        print(LF.shape)
        return lfName
    else:
        print('Success in creating multiple files')
        return subName

class LFConfig:
    def __init__(self, configFileAddress, lookAtFileAddress):
        self.focalDist, self.aperture, self.camGrid, self.microLensArray, self.camDist = \
            readConfig(configFileAddress, lookAtFileAddress)


class LF:

    def __init__(self, projectName):
        self.Name = projectName
        self.lfRawPath = self.Name + '/raw/'
        ensure_dir(self.lfRawPath)

        self.configDir = self.Name + '/config'
        must_dir(self.configDir)

        self.camConfigFile = self.configDir + '/cameraConfig'
        must_file(self.camConfigFile)

        self.lookAtFile = self.configDir + '/lookAt'
        must_file(self.lookAtFile)

        self.gridImagesPath = self.Name + '/camImagesFlat/'
        must_dir(self.gridImagesPath)
        self.config = LFConfig(self.camConfigFile, self.lookAtFile)
        self.splitInHDD = False

        self.lf = np.empty([self.config.microLensArray[0], self.config.microLensArray[1],
                            self.config.camGrid[0], self.config.camGrid[1], 3], np.dtype(np.uint8))
        self.refocusedLf = np.empty([self.config.microLensArray[0], self.config.microLensArray[1],
                                     self.config.camGrid[0], self.config.camGrid[1], 3], np.dtype(np.uint8))
        self.aperture = np.ones([self.config.camGrid[0], self.config.camGrid[1]])
        self.lfName = ""
        self.subName = ""

    def storeLf(self, sizeLimit, restore=False):
        if (self.config.camGrid[0] * self.config.camGrid[1] * self.config.microLensArray[0] *
                self.config.microLensArray[1] * 3) / (1024 * 1024) > sizeLimit:
            self.splitInHDD = True

        if os.path.isfile(self.lfRawPath + 'LF.npy') and not restore:
            self.lfName = self.lfRawPath + 'LF.npy'
            print('single file already existed')
        elif os.path.isfile(self.lfRawPath + 'subLF_' + str(self.config.camGrid[0] - 1) + '.npy') and not restore:
            self.lfName = self.lfRawPath + 'subLF_'
            print('multiple files already existed')
        else:
            self.lfName = readImages(self.gridImagesPath, self.config.camGrid, self.config.microLensArray,
                                     self.lfRawPath, self.splitInHDD)

    def loadLf(self):
        if not self.splitInHDD:
            try:
                self.lf = np.load(self.lfName)
            except:
                print("Failed to load, the" + self.lfName + " does not exists.")
        else:
            self.subName = self.lfName

    def aperture_shape(self, radii):
        yy, xx = np.meshgrid(np.arange(self.aperture.shape[0]) - ((self.aperture.shape[0] - 1.0) / 2.0),
                             np.arange(self.aperture.shape[1]) - ((self.aperture.shape[1] - 1.0) / 2.0),
                             indexing='ij')

        for i in range(self.aperture.shape[0]):
            for j in range(self.aperture.shape[1]):
                if yy[i, j].astype(float) ** 2 + xx[i, j].astype(float) ** 2 < radii ** 2:
                    self.aperture[i, j] = 1.0

    # all-in-focus is middle view of light field
    def allInFocusImage(self):
        if not self.splitInHDD:
            aif = self.lf[:, :, self.config.camGrid[0] // 2, self.config.camGrid[1] // 2, :]
        else:
            aifSubName = self.subName + str(self.config.camGrid[0] // 2) + '.npy'
            aifSub = np.load(aifSubName)
            aif = aifSub[:, :, self.config.camGrid[1] // 2, :]
        return aif

    # Conventional camera DOF
    # (refocused = True + splitHDD) is not implemented
    def depthOfFieldImage(self, apertureShape, refocused=False, highRam=True):
        if not refocused:
            lf = self.lf
        else:
            lf = self.refocusedLf

        if not self.splitInHDD:
            if highRam:
                shape_tile = np.tile(apertureShape[np.newaxis, np.newaxis, :, :, np.newaxis],
                                     (lf.shape[0], lf.shape[1], 1, 1, 3))
                dof = np.sum(shape_tile * lf, axis=(2, 3)) / np.sum(apertureShape)
            else:
                apertureShapeRGB = np.tile(apertureShape[np.newaxis, :, :, np.newaxis], (lf.shape[1], 1, 1, 3))

                dof = np.zeros([lf.shape[0], lf.shape[1], lf.shape[4]])
                for i in range(lf.shape[0]):
                    dof[i, :, :] = np.sum(apertureShapeRGB * lf[i, :, :, :, :], axis=(1, 2)) / np.sum(
                        apertureShape)
        else:
            subDof = []
            for i in range(self.config.camGrid[0]):
                subLFnumpyFile = self.subName + str(i) + '.npy'
                subLF = np.load(subLFnumpyFile)
                shape_tile = np.tile(apertureShape[np.newaxis, np.newaxis, :, i, np.newaxis],
                                     (subLF.shape[0], subLF.shape[1], 1, 3))
                subDof.append(np.sum(shape_tile * subLF, axis=2))

            dof = np.zeros([subDof[0].shape[0], subDof[0].shape[1], 3])
            for i in range(self.config.camGrid[0]):
                dof = dof + subDof[i]
            dof = dof / np.sum(apertureShape)

        return dof

    def refocus(self, alpha):
        # refocus light field and integrate across aperture to form shallow depth-of-field refocused to "alpha"
        # alpha is the disparity to focus to
        # (alpha=0 keeps original focus from when light field captured)
        # alpha can be calibrated to correspond to metric depth using camera parameters
        # lf_refocused[x,y,u,v] = lf[x+u*alpha,y+u*alpha,u,v]
        lfSize = self.lf.shape
        lf_refocused = np.zeros_like(self.lf)
        yValues = np.arange(lfSize[0]).astype(float)
        xValues = np.arange(lfSize[1]).astype(float)
        yy, xx = np.meshgrid(yValues, xValues, indexing='ij')
        for v in range(lfSize[2]):
            yy_t = np.clip(yy + (v - ((lfSize[2] - 1) / 2.0)) * alpha, 0, lfSize[0])
            for u in range(lfSize[3]):
                xx_t = np.clip(xx + (u - ((lfSize[3] - 1) / 2.0)) * alpha, 0, lfSize[1])
                interp_coords = np.stack([np.reshape(yy_t, [-1]), np.reshape(xx_t, [-1])], axis=-1)
                for c in range(3):
                    interp_values = sp.interpolate.interpn([yValues, xValues], self.lf[:, :, v, u, c], interp_coords,
                                                           method='linear', bounds_error=False, fill_value=None)
                    lf_refocused[:, :, v, u, c] = np.reshape(interp_values, [lfSize[0], lfSize[1]])
        self.refocusedLf = lf_refocused

    def downSampler(self, ratio):
        camGrid = np.divide(self.config.camGrid, ratio)

        downSampledDir = self.Name + str(camGrid[0]) + 'x' + str(camGrid[1])
        imageDir = downSampledDir + '/camImagesFlat/'
        if not os.path.exists(downSampledDir):
            os.makedirs(downSampledDir)
            os.makedirs(imageDir)
            os.makedirs(downSampledDir + '/config/')

        aperture = np.zeros([self.config.camGrid[0], self.config.camGrid[1]])
        ones = np.ones([ratio, ratio])
        zeros = np.zeros([ratio, ratio])
        for i in range(camGrid[0]):
            for j in range(camGrid[1]):
                aperture[i * ratio:(i + 1) * ratio, j * ratio:(j + 1) * ratio] = ones
                dof = self.depthOfFieldImage(aperture)
                cv2.imwrite(imageDir + str(i) + '_' + str(j) + '.png', dof)
                aperture[i * ratio:(i + 1) * ratio, j * ratio:(j + 1) * ratio] = zeros
        print('Down sampled LF with ratio of ' + str(ratio) + ' has been created and saved in ' + downSampledDir)
