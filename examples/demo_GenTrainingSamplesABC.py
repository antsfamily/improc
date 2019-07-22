from matplotlib import pyplot as plt
import improc as imp
import numpy as np
import os

import scipy.io as scio

patchSize = [480, 480, 4]
# patchSize = [240, 240, 4]
# patchSize = [32, 32, 4]
numPatches = 200
numSelPtcs = 100
sortway = 'ascent'
sortway = 'descent'
# sortway = None
startid = 0
noise = 'wgn'
# noise = None
SNR = 30

# datasetname = 'RSSRAI2019TRAIN'
datasetname = 'RSSRAI2019VAL'

# --------------------------------------
folderIN = '/mnt/d/DataSets/oi/rsi/RSSRAI2019/train/train/'
folderOUT = '/mnt/d/DataSets/oi/rsi/RSSRAI2019/train/samples/'
folderIN = '/mnt/d/DataSets/oi/rsi/RSSRAI2019/val/val/'
folderOUT = '/mnt/d/DataSets/oi/rsi/RSSRAI2019/val/samples/'

num = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 20]
num = [15]

folderA = 'img_2017'
folderB = 'img_2018'
folderC = 'mask'


folderAin = os.path.join(folderIN, folderA)
folderBin = os.path.join(folderIN, folderB)
folderCin = os.path.join(folderIN, folderC)

folderAout = os.path.join(folderOUT, folderA)
folderBout = os.path.join(folderOUT, folderB)
folderCout = os.path.join(folderOUT, folderC)

os.makedirs(folderAout, exist_ok=True)
os.makedirs(folderBout, exist_ok=True)
os.makedirs(folderCout, exist_ok=True)


imageNameA = 'image_2017_960_960_'
imageNameB = 'image_2018_960_960_'
imageNameC = 'mask_2017_2018_960_960_'
imgspathA = []
imgspathB = []
imgspathC = []

for n in num:
    imgspathA.append(folderAin + '/' + imageNameA + str(n) + '.tif')
    imgspathB.append(folderBin + '/' + imageNameB + str(n) + '.tif')
    imgspathC.append(folderCin + '/' + imageNameC + str(n) + '.tif')

A = []
B = []
C = []
cc = np.zeros((960, 960, 4), dtype='uint8')
for n in range(len(num)):
    A.append(imp.imreadadv(imgspathA[n]))
    B.append(imp.imreadadv(imgspathB[n]))
    c = imp.imreadadv(imgspathC[n])
    cc[:, :, 0] = c
    cc[:, :, 1] = c
    cc[:, :, 2] = c
    cc[:, :, 3] = c
    print(cc.min(), cc.max())
    C.append(cc)

# N-H-W-C --> H-W-C-N
A = np.transpose(np.array(A), (1, 2, 3, 0))
B = np.transpose(np.array(B), (1, 2, 3, 0))
C = np.transpose(np.array(C), (1, 2, 3, 0))


print(A.shape, B.shape, C.shape)
print(A.min(), A.max())
print(B.min(), B.max())
print(C.min(), C.max())


# data = {'C': C}
# scio.savemat('./C.mat', data)

patchesA, patchesB, patchesC = imp.imgsABC2ptcs(A, B, C, patchSize, numPatches)

if noise is not None:
    PEAK = imp.peakvalue(patchesA)
    patchesA = imp.matnoise(patchesA, noise=noise, PEAK=PEAK)
    PEAK = imp.peakvalue(patchesB)
    patchesB = imp.matnoise(patchesB, noise=noise, PEAK=PEAK)

print(patchesA.dtype, patchesA.shape)
print(patchesB.dtype, patchesB.shape)
print(patchesC.dtype, patchesC.shape)

# show
imp.showblks(patchesA[:, :, 0:3, :100], rcsize=[10, 10], stride=(0, 0))
imp.showblks(patchesB[:, :, 0:3, :100], rcsize=[10, 10], stride=(0, 0))
imp.showblks(patchesC[:, :, 0:3, :100], rcsize=[10, 10], stride=(0, 0))

print(patchesA.min(), patchesA.max())
print(patchesB.min(), patchesB.max())
print(patchesC.min(), patchesC.max())

# H-W-C-N   ->   N-H-W-C
patchesA = np.transpose(patchesA, (3, 0, 1, 2))
patchesB = np.transpose(patchesB, (3, 0, 1, 2))
patchesC = np.transpose(patchesC, (3, 0, 1, 2))

cnt = startid
for patchA, patchB, patchC in zip(patchesA, patchesB, patchesC):
    # imgA = patchA[:, :, 0]
    # imgB = patchB[:, :, 0]
    # imgC = patchC[:, :, 0]
    imgA = patchA
    imgB = patchB
    imgC = patchC[:, :, 0]
    # print(patchA.shape, imgA.shape)
    # print(patchB.shape, imgB.shape)
    # print(patchC.shape, imgC.shape)
    cnt = cnt + 1
    outfilename = "%06d" % cnt + ".tif"
    outfileA = os.path.join(folderAout, outfilename)
    outfileB = os.path.join(folderBout, outfilename)
    outfileC = os.path.join(folderCout, outfilename)
    # print(outfileA)
    # print(outfileB)
    # print(outfileC)
    imp.imwriteadv(outfileA, imgA)
    imp.imwriteadv(outfileB, imgB)
    imp.imwriteadv(outfileC, imgC)


# ------------write to file

npA, hpA, wpA, cpA = patchesA.shape
npB, hpB, wpB, cpB = patchesB.shape
npC, hpC, wpC, cpC = patchesC.shape

print("npA, hpA, wpA, cpA: ", npA, hpA, wpA, cpA)
print("npB, hpB, wpB, cpB: ", npB, hpB, wpB, cpB)
print("npC, hpC, wpC, cpC: ", npC, hpC, wpC, cpC)

data = {}
data['T1'] = patchesA
data['T2'] = patchesB
data['GT'] = patchesC[:, :, :, 0]
data['info'] = '(N, H, W, C) --> T1(%d, %d, %d, %d); T2(%d, %d, %d, %d); GT(%d, %d, %d)' % (
    npA, hpA, wpA, cpA, npB, hpB, wpB, cpB, npC, hpC, wpC)

FMT = '.pkl'
filename = datasetname + str(numPatches) + 'SamplesWithPatchSize' + str(
    patchSize[0]) + 'x' + str(patchSize[1]) + 'x' + str(patchSize[2]) + FMT
outfile = os.path.join(folderOUT, filename)
imp.save(data, outfile)
data = imp.load(outfile)

print(data['info'])
print(data['T1'].shape)
print(data['T2'].shape)
print(data['GT'].shape)

plt.show()
