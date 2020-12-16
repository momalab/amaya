import os.path
from os import path
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import pickle
import os
import random
import math, scipy
from scipy.interpolate import interp1d
import pandas as pd

# IMPORTANT --------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTANT *************
datasetMalBasePath = '<PLEASE MENTION EXTRACTED FEATURE DIR FOR MALWARE PATH HERE>'
datasetGoodBasePath = '<PLEASE MENTION EXTRACTED FEATURE DIR FOR GOODWARE PATH HERE>'
resultPath = '<PLEASE MENTION THE RESULT DIR>'

def SaveDataStructure(filePath, dataStructure):
    if not path.exists(filePath):
        with open(filePath, 'wb') as handle:
            pickle.dump(dataStructure, handle, protocol=pickle.HIGHEST_PROTOCOL)

def LoadDataStructure(filePath):
    with open(filePath, 'rb') as handle:
        return(pickle.load(handle))

def CreateDirIfNotExist(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

def SaveList(filePath, tempList):
    listPath = filePath + 'sample_list.txt'
    if not path.exists(listPath):
        fileHandle = open(listPath, 'w')
        for sampleName in tempList:
            fileHandle.write(sampleName + '\n')
    else:
        print('Sample list already backed up ...')

# Utilize the Entropy_Image folder to get file names
def GetSampleName(basePath):
    tempList = []
    dirPath = basePath + 'entropy_image/'
    fileNameList = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]

    for fileName in fileNameList:
        tempList.append(fileName.split('_')[0])
    return tempList

def ReadEntropyImage(basePath, fileNameList):
    featureDict = dict()
    for fileName in fileNameList:
        featureDict[fileName] = np.array(Image.open(basePath + 'entropy_image/' + fileName + '_original.png'))

    for downSampleRate in range(2, 17):
        tempDict = dict()
        fileSampleRateName = str(downSampleRate * downSampleRate)
        CreateDirIfNotExist(basePath + 'compressed-features/' + fileSampleRateName)
        entropyBackupPath = basePath + 'compressed-features/' + fileSampleRateName + '/entropy_image.pkl'

        if not path.exists(entropyBackupPath):
            for fileName in fileNameList:
                tempDict[fileName] = CompressAndFlatten(True, True, downSampleRate, downSampleRate, featureDict[fileName])
            SaveDataStructure(entropyBackupPath, tempDict)
        else:
            print('Saved data structure exists at %s ...'%(entropyBackupPath))
        del tempDict
        print('Processed %s ...'%(entropyBackupPath))
    del featureDict

def ReadRawEntropy(basePath, fileNameList):
    featureDict = dict()
    for fileName in fileNameList:
        featureDict[fileName] = np.loadtxt(basePath + '/entropy_raw_2d/' + fileName + '_raw.txt', delimiter=',')

    for downSampleRate in range(2, 17):
        tempDict = dict()
        fileSampleRateName = str(downSampleRate * downSampleRate)
        CreateDirIfNotExist(basePath + 'compressed-features/' + fileSampleRateName)
        entropyBackupPath = basePath + 'compressed-features/' + fileSampleRateName + '/entropy_raw.pkl'

        if not path.exists(entropyBackupPath):
            for fileName in fileNameList:
                tempDict[fileName] = CompressAndFlatten(True, True, downSampleRate, downSampleRate, featureDict[fileName])
            SaveDataStructure(entropyBackupPath, tempDict)
        else:
            print('Saved data structure exists at %s ...'%(entropyBackupPath))
        del tempDict
        print('Processed %s ...' % (entropyBackupPath))
    del featureDict

def ReadStringInfo(basePath, fileNameList):
    featureDict = dict()
    for fileName in fileNameList:
        featureDict[fileName] = np.loadtxt(basePath + '/string_info/' + fileName + '_strinfo.txt', delimiter=',')

    for downSampleRate in range(2, 17):
        tempDict = dict()
        fileSampleRateName = str(downSampleRate * downSampleRate)
        CreateDirIfNotExist(basePath + 'compressed-features/' + fileSampleRateName)
        strBackupPath = basePath + 'compressed-features/' + fileSampleRateName + '/string_info.pkl'

        if not path.exists(strBackupPath):
            for fileName in fileNameList:
                tempDict[fileName] = CompressAndFlatten(False, False, downSampleRate * downSampleRate, None, featureDict[fileName])
            SaveDataStructure(strBackupPath, tempDict)
        else:
            print('Saved data structure exists at %s ...'%(strBackupPath))
        del tempDict
        print('Processed %s ...' % (strBackupPath))
    del featureDict

def ReadSyscallInfo(basePath, fileNameList):
    featureDict = dict()
    for fileName in fileNameList:
        featureDict[fileName] = np.loadtxt(basePath + '/syscall_info/' + fileName + '_syscall.txt', delimiter=',')

    for downSampleRate in range(2, 17):
        tempDict = dict()
        fileSampleRateName = str(downSampleRate * downSampleRate)
        CreateDirIfNotExist(basePath + 'compressed-features/' + fileSampleRateName)
        syscallBackupPath = basePath + 'compressed-features/' + fileSampleRateName + '/syscall_info.pkl'

        if not path.exists(syscallBackupPath):
            for fileName in fileNameList:
                tempDict[fileName] = CompressAndFlatten(False, False, downSampleRate * downSampleRate, None, featureDict[fileName])
            SaveDataStructure(syscallBackupPath, tempDict)
        else:
            print('Saved data structure exists at %s ...'%(syscallBackupPath))
        del tempDict
        print('Processed %s ...' % (syscallBackupPath))
    del featureDict

def Downsample1D(originalArray, downSampleRate):
    padSize = math.ceil(float(originalArray.size) / downSampleRate) * downSampleRate - originalArray.size
    originalArrayPadded = np.append(originalArray, np.zeros(padSize))

    batchSize = int(originalArrayPadded.size / downSampleRate)
    return np.mean(originalArrayPadded.reshape(-1, batchSize), axis=1)


def CompressAndFlatten(is2D, doFlatten, row, column, originalArray):
    compressedArray = None
    flattenArray = None

    if is2D:
        compressedArray = np.array(Image.fromarray(originalArray).resize((row, column), Image.NEAREST))
    else:
        compressedArray = Downsample1D(originalArray, row)

    if doFlatten:
        flattenArray = compressedArray.flatten()
    else:
        flattenArray = compressedArray

    return flattenArray

if __name__ == '__main__':
    CreateDirIfNotExist(datasetMalBasePath + 'compressed-features/')
    malwareList = GetSampleName(datasetMalBasePath)
    SaveList(datasetMalBasePath + 'compressed-features/', malwareList)
    ReadEntropyImage(datasetMalBasePath, malwareList)
    ReadRawEntropy(datasetMalBasePath, malwareList)
    ReadStringInfo(datasetMalBasePath, malwareList)
    ReadSyscallInfo(datasetMalBasePath, malwareList)

    CreateDirIfNotExist(datasetGoodBasePath + 'compressed-features/')
    goodwareList = GetSampleName(datasetGoodBasePath)
    SaveList(datasetGoodBasePath + 'compressed-features/', goodwareList)
    ReadEntropyImage(datasetGoodBasePath, goodwareList)
    ReadRawEntropy(datasetGoodBasePath, goodwareList)
    ReadStringInfo(datasetGoodBasePath, goodwareList)
    ReadSyscallInfo(datasetGoodBasePath, goodwareList)

    # Create text file for x86
    malDictBasePath = datasetMalBasePath + 'compressed-features/'
    goodDictBasePath = datasetGoodBasePath + 'compressed-features/'

    downSampleRateList = ['4', '9', '16', '25', '36', '49', '64', '81', '100', '121', '144', '169', '196', '225', '256']
    for downSampleRate in downSampleRateList:
        malPath = malDictBasePath + downSampleRate + '/'
        goodPath = goodDictBasePath + downSampleRate + '/'

        x86MalEntropy = LoadDataStructure(malPath + 'entropy_raw.pkl')
        x86GoodEntropy = LoadDataStructure(goodPath + 'entropy_raw.pkl')

        x86MalStrInfo = LoadDataStructure(malPath + 'string_info.pkl')
        x86GoodStrInfo = LoadDataStructure(goodPath + 'string_info.pkl')

        x86MalSyscall = LoadDataStructure(malPath + 'syscall_info.pkl')
        x86GoodSyscall = LoadDataStructure(goodPath + 'syscall_info.pkl')

        tempMalList = []
        tempMalDf = pd.DataFrame()
        for hash in x86MalEntropy:
            tempMalList = [hash] + x86MalEntropy[hash].tolist() + x86MalStrInfo[hash].tolist() + x86MalSyscall[hash].tolist() + ['1']
            tempMalDf = tempMalDf.append([tempMalList], ignore_index=True)

        tempGoodList = []
        tempGoodDf = pd.DataFrame()
        for hash in x86GoodEntropy:
            tempGoodList = [hash] + x86GoodEntropy[hash].tolist() + x86GoodStrInfo[hash].tolist() + x86GoodSyscall[hash].tolist() + ['0']
            tempGoodDf = tempGoodDf.append([tempGoodList], ignore_index=True)

        finalDataFrame = pd.concat([tempMalDf, tempGoodDf], ignore_index=True, sort=False)
        finalDataFrame = finalDataFrame.sample(frac=1).reset_index(drop=True)
        finalDataFrame.to_csv(resultPath + 'features_' + downSampleRate + '.txt')
        print('\nCreated Dataset for Downsampling Rate of %s'%(downSampleRate))
        print(finalDataFrame)