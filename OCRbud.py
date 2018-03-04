import os
import sys
import string
import argparse
import numpy as np
import tensorflow as tf
from TFANN import ANNC
from skimage.io import imread
from scipy.stats import mode as Mode
from sklearn.model_selection import ShuffleSplit

pc = len(string.ascii_letters + string.digits + ' ')                                                              #possible characters pc
max_ch = 64                                                                                                       #max characters per block max_ch
img_size = [18, 640, 3]                                                                                           #CNN image size img_size
nba = 5                                                                                                           #number of neural nets, 5 nn using bagging nba
tf_img = tf.placeholder("float", [None] + img_size, name = 'tf_img')                                              #tensorflow placeholders for image dataset and target data tf_img, tf_targ
tf_targ = tf.placeholder("float", [None, max_ch, pc], name = 'tf_targ')

def DvdIntoSubImg(I):
    '''
    Division of images into sub parts to give as input to OCR
    '''
    h, w, c = I.shape
    H, W = h // img_size[0], w // img_size[1]
    HP = img_size[0] * H
    WP = img_size[1] * W
    I = I[0:HP, 0:WP]                                                                                            #removal of all the extra pixels
    return I.reshape(H,
                     img_size[0], -1,
                     img_size[1], c).swapaxes(1, 2).reshape(-1, img_size[0],
                                                            img_size[1], c)

def SubImgShape(I):
    '''
        Get no. of rows, cols of image sub-divs
        '''
    h, w, c = I.shape
    return h // img_size[0], w // img_size[1]

    
def MakeNet(nn = 'ocrnet'):
    #neural net arch
    #convolution layers reduce input volume to shape of output
    #18 / 2 * 3 * 3 = 1 and 640 / 2 * 5 = 64 output.shape
    ws = [('C', [5, 5,  3, pc // 2], [1, 2, 2, 1]), ('AF', 'relu'),
          ('C', [4, 4, pc // 2, pc], [1, 3, 1, 1]), ('AF', 'relu'),
          ('C', [3, 5, pc,      pc], [1, 3, 5, 1]), ('AF', 'relu'),
          ('R', [-1, max_ch, pc])]
    #creation of neural network in TensorFlow
    return ANNC(img_size,
                ws,
                batchSize = 512,
                learnRate = 2e-5,
                maxIter = 64,
                name = nn,
                reg = 1e-5,
                tol = 1e-2,
                verbose = True,
                X = tf_img,
                Y = tf_targ)

def LdNets():
    cnn = [MakeNet('ocrnet' + str(i)) for i in range(nba)]
    if not cnn[-1].RestoreModel('Tensorflow_model/', 'graph'):
        A, Y, T, FN = LdData()
        for CNNi in cnn:
            FitModel(CNNi, A, Y, T, FN)
        cnn[-1].SaveModel(os.path.join('Tensorflow_model', 'graph'))
        with open('Tensorflow_model/_classes.txt', 'w') as F:
            F.write('\n'.join(cnn[-1]._classes))
    else:
        with open('Tensorflow_model/_classes.txt') as F:
            cl = F.read().splitlines()
        for CNNi in cnn:
            CNNi.RestoreClasses(cl)
    return cnn

def LdData(OCRpath = '.'):
    '''
        Load OCR dataset.
        A: matrix of images (NIMG, Height, Width, Channel).
        Y: matrix of characters (NIMG, MAX_CHAR)
        OCRpath: OCR data path
        return: Data Matrix, Target Matrix, Target Strings
        '''
    trn_path = os.path.join(OCRpath, 'Trn.csv')
    A, Y, T, FN = [], [], [], []
    with open(trn_path) as F:
        for i, Li in enumerate(F):
            FNi, Yi = Li.strip().split(',')                                                                   #filename,string
            T.append(Yi)
            A.append(imread(os.path.join(OCRpath, FNi))[:, :, :3])                                            #read image and discard alpha channel
            Y.append(list(Yi) + [' '] * (max_ch - len(Yi)))                                                   #provide string padding with spaces
            FN.append(FNi)
    return np.stack(A), np.stack(Y), np.stack(T), np.stack(FN)
    
def FitModel(cnnc, A, Y, T, FN):
    print('Model is being fit....')
    ss = ShuffleSplit(n_splits = 1)
    trn, tst = next(ss.split(A))
    cnnc.fit(A[trn], Y[trn])                                                                                    #Fit network
    YH = []                                                                                                     #preds as seq of char indices
    for i in np.array_split(np.arange(A.shape[0]), 32): 
        YH.append(cnnc.predict(A[i]))
    YH = np.vstack(YH)
    PS = np.array([''.join(YHi) for YHi in YH])                                                                 #convert seq of char indices to str
    #Compute the accuracy
    S1 = SAcc(PS[trn], T[trn])
    S2 = SAcc(PS[tst], T[tst])
    print('Train: ' + str(S1))
    print('Test: ' + str(S2))
    for PSi, Ti, FNi in zip(PS, T, FN):
        if np.random.rand() > 0.99:                                                                             #print random rows
            print(FNi + ': ' + Ti + ' ----> ' + PSi)
    print('Being fit with CV data....')
    #Fit remainder
    cnnc.SetMaxIter(4)
    cnnc.fit(A, Y)
    return cnnc
    
def JoinStr(YH, ss):
    '''
    Arrange substrings as per order of image sub-divisions
    '''
    YH = np.array([''.join(YHi) for YHi in YH]).reshape(ss)
    return '\n'.join(''.join(YHij for YHij in YHi) for YHi in YH)

def ImgToStr(I):
    '''
        OCR converts image to string
        '''
    SI = DvdIntoSubImg(I)
    ss = SubImgShape(I)
    YH = FuseRes(TFS.run(YHL, feed_dict = {TFIM: SI}))
    return JoinStr(CNN[-1]._classes[YH], ss)

def SAcc(T, PS):
    return sum(sum(i == j for i, j in zip(S1, S2)) / len(S1) for S1, S2 in zip(T, PS)) / len(T)

def FuseRes(T):                                                                                                #take mode result from 5 networks
    return (Mode(np.stack([Ti.ravel() for Ti in T]))[0]).reshape(T[0].shape)



    
cnn = LdNets()                                                                                                #Convolutional Neural networks loaded
YHL = [CNNi.YHL for CNNi in cnn]                                                                              #Prediction placeholders
TFS = cnn[-1].GetSes()                                                                                        #Tensorflow getses() method to get Tensorflow session
if __name__ == "__main__":
    p = argparse.ArgumentParser(description = 'OCRbud: Deep learning based OCR')
    p.add_argument('-f', action = 'store_true', help = 'Force model training')
    p.add_argument('Img', metavar = 'I', type = str, nargs = '+', help = 'Image files')
    pa = p.parse_args()
    if pa.f:
        FitModel()
    for img in pa.Img:
        I = imread(img)[:, :, :3]                                                                             #read image and discard alpha
        S = ImgToStr(I)
        print(S)
