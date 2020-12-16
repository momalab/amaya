from utils import *
import math
from PIL import Image, ImageDraw
from numpy import asarray
import numpy as np

import r2pipe
from math import log
import shutil
from os import listdir
from os.path import isfile, join
import hashlib
import os

import zipfile

# IMPORTANT --------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTANT *************
DATASET_PATH = '<PLEASE REPLACE ME with DATASET PATH>'
TEMP_PATH = 'temp/'


# Hilbert byte entropy helper functions from
# https://github.com/cortesi/scurve/tree/master/scurve
def hilbert_point(dimension, order, h):
    """
        Convert an index on the Hilbert curve of the specified dimension and
        order to a set of point coordinates.
    """
    #    The bit widths in this function are:
    #        p[*]  - order
    #        h     - order*dimension
    #        l     - dimension
    #        e     - dimension
    hwidth = order*dimension
    e, d = 0, 0
    p = [0]*dimension # create a dummy result position vector [0, 0] for 2 dimensions
    for i in range(order):
        w = bitrange(h, hwidth, i*dimension, i*dimension+dimension)
        l = graycode(w)
        l = itransform(e, d, dimension, l)
        for j in range(dimension):
            b = bitrange(l, dimension, j, j+1)
            p[j] = setbit(p[j], order, i, b)
        e = e ^ lrot(entry(w), d+1, dimension)
        d = (d + direction(w, dimension) + 1)%dimension
    return p


def hilbert_index(dimension, order, p):
    h, e, d = 0, 0, 0
    for i in range(order):
        l = 0
        for x in range(dimension):
            b = bitrange(p[dimension-x-1], order, i, i+1)
            l |= b<<x
        l = transform(e, d, dimension, l)
        w = igraycode(l)
        e = e ^ lrot(entry(w), d+1, dimension)
        d = (d + direction(w, dimension) + 1)%dimension
        h = (h<<dimension)|w
    return h


class Hilbert:
    def __init__(self, dimension, order):
        # default order = 8, and dimension = 2
        self.dimension, self.order = dimension, order

    @classmethod
    def fromSize(self, dimension, size):
        """
            Size is the total number of points in the curve. Finds number of points on the curve. Always will be 16 points for our settings.
        """
        x = math.log(size, 2)

        if not float(x)/dimension == int(x)/dimension:
            raise ValueError("Size does not fit Hilbert curve of dimension %s."%dimension)
        return Hilbert(dimension, int(x/dimension))

    def __len__(self):
        return 2**(self.dimension*self.order)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        return self.point(idx)

    def dimensions(self):
        """
            Size of this curve in each dimension.
        """
        return [int(math.ceil(len(self)**(1/float(self.dimension))))]*self.dimension

    def index(self, p): 
        return hilbert_index(self.dimension, self.order, p)

    def point(self, idx):
        return hilbert_point(self.dimension, self.order, idx)


def entropy(data, blocksize, offset, symbols=256):
    """
        Returns local byte entropy for a location in a file.
        This code has a window of 32 bytes and moves the window by just 1 byte.
    """
    if len(data) < blocksize:
        raise ValueError("Data length must be larger than block size.") 
    if offset < blocksize/2:
        start = 0
    elif offset > len(data)-blocksize/2:
        start = len(data)-blocksize/2
    else:
        start = offset-blocksize/2
    hist = {}
    start = int(start)
    blocksize = int(blocksize)

    for i in data[start:start+blocksize]:
        hist[i] = hist.get(i, 0) + 1
        
   
    base = min(blocksize, symbols)
    entropy = 0

    for i in hist.values():
       
        p = i/float(blocksize)
        # If blocksize < 256, the number of possible byte values is restricted.
        # In that case, we adjust the log base to make sure we get a value
        # between 0 and 1.
        entropy += (p * math.log(p, base))
    return -entropy


class _Color:
    def __init__(self, data, block):
        self.data, self.block = data, block
        s = list(set(data)) # set() finds all the unique bytes present in the binary. From 0x00 to 0xff (255). Then convert set to list.
        s.sort()
        self.symbol_map = {v : i for (i, v) in enumerate(s)}

    def __len__(self):
        return len(self.data)

    def point(self, x):
        if self.block and (self.block[0]<=x<self.block[1]):
            return self.block[2]
        else:
            return self.getPoint(x)


class ColorEntropy(_Color):
    def getPoint(self, x):
        e = entropy(self.data, 32, x, len(self.symbol_map))
        # http://www.wolframalpha.com/input/?i=plot+%284%28x-0.5%29-4%28x-0.5%29**2%29**4+from+0.5+to+1
        def curve(v):
            f = (4*v - 4*v**2)**4
            f = max(f, 0)
            return f
        r = curve(e-0.5) if e > 0.5 else 0
        b = e**2
        
        return [int(255*r), 0, int(255*b)], e


def drawmap_unrolled(map, size, csource, name):
	# For raw entropy values
    entropyArray = np.empty((size*4, size), float)
    entropy1DArray = list()

    map = Hilbert.fromSize(2, size**2)

    # For color entropy values
    c = Image.new("RGB", (size, size*4))
    cd = ImageDraw.Draw(c)
    step = len(csource)/float(len(map)*4)

    sofar = 0
    for quad in range(4):
        for i, p in enumerate(map):
            off = (i + (quad * size**2))
            color, entropy = csource.point(int(off * step))

            x, y = tuple(p)
            entropyArray[y + (size * quad), x] = entropy
            entropy1DArray.append(entropy)
            cd.point((x, y + (size * quad)), fill=tuple(color))

            sofar += 1

    RemoveIfFileExist('entropy_image/' + name + '_original.png')
    RemoveIfFileExist('entropy_raw_2d/' + name + '_raw.txt')
    c.save('entropy_image/' + name + '_original.png')
    np.savetxt('entropy_raw_2d/' + name + '_raw.txt', entropyArray, delimiter=',')

    # Save 1D entropy array, sequential entropy information. No location information
    RemoveIfFileExist('entropy_raw_1d/' + name + '_raw.txt')
    np.savetxt('entropy_raw_1d/' + name + '_raw.txt', np.asarray(entropy1DArray), delimiter=',')


# The binary is traversed sequentially, but the image is filled according to hilbert curve
# So, using the 2d numpy entropy array is a better choice than 1d array. 
# Compress / downsample it to reduce the input size
def drawmap_square(map, size, csource, name):
	# For raw entropy values
    entropyArray = np.empty((size, size), float)
    entropy1DArray = list()

    # Increase size**2 value to change the number of points on the hilber curve. Ex, size**8
    # map is a Hilber object, with dimension = 2 and order = log(size**2)/dimension (16/2 = 8 in default case)
    map = Hilbert.fromSize(2, size**2)

    c = Image.new("RGB", map.dimensions()) # Get square dimension for the selected size [256, 256] by default
    cd = ImageDraw.Draw(c)
    step = len(csource)/float(len(map)) # csource is the object that represents the binary data and its meta information

    # enumerate() calls __getitem__() in Hilbert class and then point and finally hilbert_point
    for i, p in enumerate(map):
        # point first calls point inside '_Color' and then getPoint in ColorEntropy
        # p has the coordinates for the hilbert curve, i is the index as a counter for p

        color, entropy = csource.point(int(i*step))
        entropyArray[tuple(p)] = entropy
        entropy1DArray.append(entropy)
        cd.point(tuple(p), fill=tuple(color))

    RemoveIfFileExist('entropy_image/' + name + '_original.png')
    RemoveIfFileExist('entropy_raw_2d/' + name + '_raw.txt')
    c.save('entropy_image/' + name + '_original.png')
    np.savetxt('entropy_raw_2d/' + name + '_raw.txt', entropyArray, delimiter=',')

    # Save 1D entropy array, sequential entropy information. No location information
    RemoveIfFileExist('entropy_raw_1d/' + name + '_raw.txt')
    np.savetxt('entropy_raw_1d/' + name + '_raw.txt', np.asarray(entropy1DArray), delimiter=',')


# String and syscall helper functions

def RemoveIfFileExist(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)


def StringHistogram(filestrings, name):
    histogram = [[0 for i in range(16)] for j in range(16)]
    for section in filestrings:
        for string in filestrings[section]:
            if len(string) >= 6:
                if log(len(string), 1.25) > 188:
                    bin1 = 15
                else:
                    bin1 = int((log(len(string), 1.25) - 8)//12)

                bin2 = int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16) % 16

                histogram[bin1][bin2] += 1

    feature_array = histogram[0].copy()
    for i in range(1,16):
        feature_array.extend(histogram[i])

    RemoveIfFileExist('string_info/' + name + '_strinfo.txt')
    np.savetxt('string_info/' + name + '_strinfo.txt', feature_array, delimiter=',')


def SyscallHistogram(syscallList, name):
    histogram = [[0 for i in range(16)] for j in range(16)]
    for syscall in syscallList:
        if log(len(syscall), 1.25) > 188:
            bin1 = 15
        else:
            bin1 = int((log(len(syscall), 1.25) - 8)//12)

        bin2 = int(hashlib.md5(syscall.encode('utf-8')).hexdigest(), 16) % 16
        histogram[bin1][bin2] += 1

    feature_array = histogram[0].copy()
    for i in range(1,16):
        feature_array.extend(histogram[i])

    RemoveIfFileExist('syscall_info/' + name + '_syscall.txt')
    np.savetxt('syscall_info/' + name + '_syscall.txt', feature_array, delimiter=',')

def CompletedList(dirPath):
    completedSamples = []
    allfiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
    for sample in allfiles:
        completedSamples.append(sample.split('_')[0] + '.zip')

    return completedSamples

if __name__ == '__main__':
    counter = 0

    CompletedList = CompletedList('archive_syscall_info/')
    allfiles = [f for f in listdir(DATASET_PATH) if isfile(join(DATASET_PATH, f))]
    totalLength = len(allfiles)

    for filename in allfiles:
        # Copy to Temp directory for processing
        counter += 1
        saveFilename = filename.split('.')[0]

        if filename in CompletedList:
            print('Completed %s ... (%i/%i)'%(saveFilename, counter, totalLength))
            continue

        shutil.copy(DATASET_PATH + filename, TEMP_PATH + filename)

        if zipfile.is_zipfile(TEMP_PATH + filename):
            # Unzip the file
            with zipfile.ZipFile(TEMP_PATH + filename, 'r') as zipRef:
                zipRef.extractall(TEMP_PATH)
        os.remove(TEMP_PATH + filename)
        tempFilePath = TEMP_PATH + saveFilename

        #filename = filename.rstrip()
        print('Processing %s ... (%i/%i)'%(saveFilename, counter, totalLength))

        # Processing for hilbert byte entropy and saving raw values and image
        with open(tempFilePath, "rb") as f:
            d = f.read()

        block = None
        size = 256
        map = 'hilbert'
        dst = saveFilename

        csource = ColorEntropy(d, block)
        drawmap_square(map, size, csource, dst)

        # Processing for string histogram and saving the output in file
        #r = r2pipe.open(tempFilePath, flags=['-a', 'arm'])
        r = r2pipe.open(tempFilePath, flags=['-a', 'x86'])

        f = open(TEMP_PATH + saveFilename + '_strinfo.txt', 'w+')
        f.write(r.cmd('izz'))
        f.close()

        f = open(TEMP_PATH + saveFilename + '_syscall.txt', 'w+')
        f.write(r.cmd('asl'))
        f.close()

        RemoveIfFileExist('string_info/' + saveFilename + '_strinfo.txt')
        RemoveIfFileExist('syscall_info/' + saveFilename + '_syscall.txt')

        # Parsing string in the binary and creating 16 x 16 histogram
        filestrings = {}
        with open(TEMP_PATH + saveFilename +'_strinfo.txt') as rf:
            for _ in range(3):
                next(rf)
            for l in rf:
                line = l.split()
                parsedLine = line[5:]

                if parsedLine[0][0] == '.':
                    if parsedLine[0] in filestrings:
                        filestrings[parsedLine[0]].append(' '. join(parsedLine[2:]))
                    else:
                        filestrings[parsedLine[0]] = [' '. join(parsedLine[2:])]
                else:
                    
                    if 'unclassified' in filestrings:
                        filestrings['unclassified'].append(' '. join(parsedLine[1:]))
                    else:
                        filestrings['unclassified'] = [' '. join(parsedLine[1:])]

        StringHistogram(filestrings, saveFilename)
        shutil.move(TEMP_PATH + saveFilename + '_strinfo.txt', 'archive_string_info/' + saveFilename + '_strinfo.txt')

        # Parsing syscall entries found the binary and creating 16 x 16 histogram
        syscallList = []
        with open(TEMP_PATH + saveFilename +'_syscall.txt') as rf:
            content = rf.readlines()
        content = [x.strip() for x in content]

        for syscallLine in content:
            syscall = syscallLine.split()[0]
            syscallList.append(syscall)

        SyscallHistogram(syscallList, saveFilename)
        shutil.move(TEMP_PATH + saveFilename + '_syscall.txt', 'archive_syscall_info/' + saveFilename + '_syscall.txt')
        RemoveIfFileExist(tempFilePath)