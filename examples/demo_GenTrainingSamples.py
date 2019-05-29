from matplotlib import pyplot as plt
from scipy.misc import imsave as scipyimsave

import improc as imp
import numpy as np
import os

patchSize = [128, 128, 1]
numPatches = 3000
numSelPtcs = 300
sortway = 'ascent'
# sortway = 'descent'
sortway = None
startid = 0

# --------------------------------------
infolder = '../data/imgs/'
outfolder = '../data/samples/'

samplefolder = str(patchSize[0]) + 'x' + \
    str(patchSize[1]) + 'x' + str(patchSize[2])
os.makedirs(outfolder + samplefolder, exist_ok=True)

files = os.listdir(infolder)
imgspathes = []
for imgsfile in files:
    imgspathes.append(infolder + imgsfile)


patches = imp.imgs2ptcs(imgspathes, patchSize, numPatches)

# select patches
sortpatches_std, idx, std_patchesres = imp.selptcs(
    patches, numsel=numSelPtcs, method='std', sort=sortway)

patches = sortpatches_std
numptcs = patches.shape[3]
print(patches.dtype, patches.shape)
# show
imp.showblks(patches[:, :, :, :100], rcsize=[10, 10])
imp.showblks(patches[:, :, :, -100:], rcsize=[10, 10])


# H-W-C-N   ->   N-H-W-C
patches = np.transpose(patches, (3, 0, 1, 2))
print(patches.shape)

cnt = startid
for patch in patches:
    img = patch[:, :, 0]
    print(patch.shape, img.shape)
    cnt = cnt + 1
    outfilename = "%06d" % cnt + ".png"
    outfile = os.path.join(outfolder, samplefolder, outfilename)
    print(outfile)
    scipyimsave(outfile, img)

plt.show()
