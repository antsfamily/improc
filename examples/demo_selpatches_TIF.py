import os
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from improc import imgs2blks, imgs2ptcs, showblks, selptcs, geocluptcs

sortway = 'descent'

numtoshow = 100
numpatches = 2000

patchsize = [8, 8, 3]
# patchsize = [32, 32, 1]

imgspathes = [
    '/mnt/d/DataSets/oi/nsi/classical/LenaRGB.bmp',
    '/mnt/d/DataSets/oi/nsi/classical/BaboonRGB.bmp',
    '/mnt/d/DataSets/oi/nsi/classical/PeppersRGB.bmp',

]

# patchsize = [8, 8, 3]
# imgspathes = [
#     # '/mnt/d/DataSets/oi/nsi/byme/goldenfish.jpg',
#     '/mnt/d/DataSets/oi/nsi/classical/BaboonRGB.bmp',
#     '/mnt/d/DataSets/oi/nsi/classical/LenaRGB.tif',
# ]

# blks, imgsInfo = imgs2blks(imgspathes, blkSize, 'symmetric')

patches = imgs2ptcs(imgspathes, patchsize, numpatches)

bS = patches.shape
print('shape of patches: ', patches.shape)
print('dtype of patches: ', patches.dtype)

patchesres = np.reshape(patches, (np.prod(bS[0:3]), bS[3]))  # H*W*C-N
std_patchesres = np.std(patchesres, 0)
print('std shape of patchesres', std_patchesres.shape)
print('---------------------------------')
print('min std:', np.min(std_patchesres), 'max std: ', np.max(std_patchesres))
print('============')

# selpatches, idxsel, selptcs_scores = selptcs(
#     patches, method='std', sort=sortway)

# just sort
sortpatches_std, idx, std_patchesres = selptcs(
    patches, method='std', sort=sortway)
print(std_patchesres)
print(idx)
print(type(std_patchesres))
orig = showblks(patches, plot=False)
sort = showblks(sortpatches_std, plot=False)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(orig, cm.gray)
plt.subplot(1, 2, 2)
plt.imshow(sort, cm.gray)



print(patches.shape)
print(type(patches))

# classify patches
imgs = np.zeros((4, 4, 1, 3), 'uint8')
imgs[:, :, 0, 0] = np.array(
    [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
imgs[:, :, 0, 1] = np.array(
    [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
imgs[:, :, 0, 2] = np.array(
    [[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
imgs = imgs * 255
smoothptcs, dominantptcs, stochasticptcs, smoothptcs_idx, dominantptcs_idx, stochasticptcs_idx = geocluptcs(
    patches)
print(smoothptcs.shape)
print(stochasticptcs.shape)
showblks(smoothptcs, plot=True, title='Smooth patches')

print('++++++====')
showblks(stochasticptcs, plot=True, title='Stochasticptcs patches')

print('=====================')

plt.figure()
for k in range(0, 6):
    print(k)
    plt.subplot(2, 3, k + 1)
    if dominantptcs[k].size == 0:
        toshow = np.ones((100, 100), 'uint8') * 255
    else:
        toshow = showblks(dominantptcs[k], rcsize=(12, 12), plot=False)
    plt.imshow(toshow, cm.gray)
    plt.title('dominantptcs, angle: ' + str(k * 30))

plt.show()
