import os
import string
import numpy as np
import matplotlib.pyplot as mpl
from numpy.random import randint, choice
from PIL import Image, ImageFont, ImageDraw

    
def GenBlock(min_c, max_c, s = (100, 100), o = (0, 0)):
    '''
    Generates a single image block. This is the image size the CNN uses
    '''
    single_img = ''.join(choice(poss_char, randint(min_c, max_c)))
    img = Image.new('RGB', s, "black")
    draw = ImageDraw.Draw(img)
    draw.text(o, single_img, (255, 255, 255), font = TF)
    return np.array(img), single_img
        
def GenImage(rows, cols, img_size, min_c = 10, max_c = 64, img_gen = 128, OCRpath = 'Trn'):
    '''
    rows:   Number of row blocks
    cols:   Number of column blocks
    img_size:   Image size
    min_c: Minimum number of characters per line
    max_c: Maximum number of characters per line
    img_gen: Number of images to generate
    OCRpath:   Directory path to write images
    '''
    Y = []
    MS = GetFontSize(max_c)
    blocks = rows * cols                                                                                                #total number of blocks = Rows * Cols
    for i in range(img_gen):                                                                                            #Write images to ./Out/ directory
        Im, Ym = MergeBlock(rows, cols, [GenBlock(min_c, max_c, img_size) for _ in range(blocks)])
        FNi = os.path.join(OCRpath, '{:05d}.png'.format(i))
        mpl.imsave(FNi, Im)
        Y.append(FNi + ',' + Ym)
    with open(OCRpath + '.csv', 'w') as F:                                                                              #Write CSV file
        F.write('\n'.join(Y))
   
def GetFontSize(max_c):
    '''
    Gets the maximum size of an image containing characters in poss_char
    of maximum length max_c
    '''
    img = Image.new('RGB', (1, 1), "black")
    draw = ImageDraw.Draw(img)
    h, w = 0, 0
    for poss_chari in poss_char:                                                                                        #Get max height and width possible characters poss_char
        tsi = draw.textsize(poss_chari * max_c, font = TF)
        h = max(tsi[0], h)
        w = max(tsi[1], w)
    return (h, w)   
    
def MergeBlock(rows, cols, genb_out):
    '''
    Merges blocks into combined images that are NR blocks tall and NC blocks wide
    rows:  Number of row blocks
    cols:  Number of column blocks
    genb_out:   List of outputs from GenBlock
    ret: Merged image, Merged string
    '''
    B = np.array([t[0] for t in genb_out])
    Y = np.array([t[1] for t in genb_out])
    n, r, c, _ = B.shape
    return Unblock(B, r * rows, c * cols), '@'.join(''.join(Yi) for Yi in Y.reshape(rows, cols))
     
def Unblock(I, h, w):
    '''
    I:   Array of shape (n, rows, cols, c)
    h:   Height of new array
    w:   Width of new array
    ret: Array of shape (h, w, c)
    '''
    n, rows, cols, c = I.shape
    return I.reshape(h // rows, -1, rows, cols, c).swapaxes(1, 2).reshape(h, w, c)
       
TF = ImageFont.truetype('consola.ttf', 18)
#Possible characters to use
poss_char = list(string.ascii_letters + string.digits + ' ')

if __name__ == "__main__":
    minc, maxc = 10, 64
    ms = GetFontSize(maxc)
    print('CNN Image Size: ' + str(ms))
    GenImage(1, 1, ms, minc, maxc, img_gen = 32768, OCRpath = 'Trn')                                                      #Training data
    GenImage(4, 2, ms, minc, maxc, img_gen = 256,   OCRpath = 'Tst')                                                      #Testing data
