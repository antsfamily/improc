import os
import improc as imp
import numpy as np
import matplotlib.pyplot as plt


blksizeA = (480, 480, 4)
blksizeB = (480, 480, 4)
blksizeC = (480, 480, 1)

startid = 0


folderIN = '/mnt/d/DataSets/oi/rsi/RSSRAI2019/val/val/'
folderOUT = '/mnt/d/DataSets/oi/rsi/RSSRAI2019/val/blocks/'

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


num = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 20]
num = [15]
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
    C.append(imp.imreadadv(imgspathC[n]))

A = np.array(A)
B = np.array(B)
C = np.array(C)
N, H, W = C.shape
C = np.reshape(C, (N, H, W, 1))
print(A.shape, B.shape, C.shape)
# N-H-W-C --> H-W-C-N
A = np.transpose(A, (1, 2, 3, 0))
B = np.transpose(np.array(B), (1, 2, 3, 0))
C = np.transpose(C, (1, 2, 3, 0))


print(A.shape, B.shape, C.shape)
print(A.min(), A.max())
print(B.min(), B.max())
print(C.min(), C.max())

blksA, imgshapeA = imp.imgs2blks(A, blksizeA)
blksB, imgshapeB = imp.imgs2blks(B, blksizeB)
blksC, imgshapeC = imp.imgs2blks(C, blksizeC)

print("imgshapeA, imgshapeB, imgshapeC: ", imgshapeA, imgshapeB, imgshapeC)
print("blksA.shape, blksB.shape, blksC.shape: ",
      blksA.shape, blksB.shape, blksC.shape)

imp.showblks(blksA[:, :, 0:3, :4], rcsize=[2, 2], stride=(1, 1))
imp.showblks(blksB[:, :, 0:3, :4], rcsize=[2, 2], stride=(1, 1))
imp.showblks(blksC[:, :, 0:3, :4], rcsize=[2, 2], stride=(1, 1))

imgsA = imp.blks2imgs(blksA, [imgshapeA])
imgsB = imp.blks2imgs(blksB, [imgshapeB])
imgsC = imp.blks2imgs(blksC, [imgshapeC])

print("imgsA[0], imgsB[0], imgsC[0]: ", imgsA[
      0].shape, imgsB[0].shape, imgsC[0].shape)


plt.figure()
plt.subplot(131)
plt.imshow(imgsA[0][:, :, 0:3])
plt.subplot(132)
plt.imshow(imgsB[0][:, :, 0:3])
plt.subplot(133)
plt.imshow(imgsC[0][:, :])
plt.show()

# H-W-C-N   ->   N-H-W-C
blksA = np.transpose(blksA, (3, 0, 1, 2))
blksB = np.transpose(blksB, (3, 0, 1, 2))
blksC = np.transpose(blksC, (3, 0, 1, 2))

cnt = startid
for blkA, blkB, blkC in zip(blksA, blksB, blksC):
    # imgA = blkA[:, :, 0]
    # imgB = blkB[:, :, 0]
    # imgC = blkC[:, :, 0]
    imgA = blkA
    imgB = blkB
    imgC = blkC[:, :, 0]
    # print(blkA.shape, imgA.shape)
    # print(blkB.shape, imgB.shape)
    # print(blkC.shape, imgC.shape)
    cnt = cnt + 1
    outfilename = "%06d" % cnt + ".tif"
    outfileA = os.path.join(folderAout, outfilename)
    outfileB = os.path.join(folderBout, outfilename)
    outfileC = os.path.join(folderCout, outfilename)
    # print(outfileA)
    # print(outfileB)
    # print(outfileC)
    imp.imsaveadv(outfileA, imgA)
    imp.imsaveadv(outfileB, imgB)
    imp.imsaveadv(outfileC, imgC)
