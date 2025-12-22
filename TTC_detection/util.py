import cv2 as cv
import numpy as np
import skimage
from skimage import feature
BRIEF_RATIO = 0.8
FAST_THEOHOLD=120

def computePixel(img, idx1, idx2, width, center):
    halfWidth = width // 2
    col1 = idx1 % width - halfWidth
    row1 = idx1 // width - halfWidth
    col2 = idx2 % width - halfWidth
    row2 = idx2 // width - halfWidth
    return 1 if img[int(center[0]+row1)][int(center[1]+col1)] < img[int(center[0]+row2)][int(center[1]+col2)] else 0

def makeTestPattern(patchWidth, nbits):
    np.random.seed(0)
    compareX = patchWidth*patchWidth * np.random.random((nbits,1))
    compareX = np.floor(compareX).astype(int)
    np.random.seed(1)
    compareY = patchWidth*patchWidth * np.random.random((nbits,1))
    compareY = np.floor(compareY).astype(int)
    return (compareX, compareY)

def computeBrief(img, locs):
    patchWidth = 9
    nbits = 256
    compareX, compareY = makeTestPattern(patchWidth,nbits)
    m, n = img.shape
    halfWidth = patchWidth//2
    locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))
    desc = np.array([list(map(lambda x: computePixel(img, x[0], x[1], patchWidth, c), zip(compareX, compareY))) for c in locs])
    return desc, locs


def briefMatch(desc1, desc2, ratio=BRIEF_RATIO):
    matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True,max_ratio=ratio)
    return matches
    #bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    #matches = bf.match(desc1, desc2)
    #matches = sorted(matches, key=lambda x: x.distance)
    #return matches

def detect_corners(image):
    """
    Detect corners in the given image using FAST detector.
    keypoints: coordinates of key points arranged in a Nx2 matrix. (row, col)
    """
    # alternative: cv2.goodFeaturesToTrack(): Harris corner detector
    fast = cv.FastFeatureDetector_create(threshold=FAST_THEOHOLD, type=2)  # TYPE_9_16
    #keypoints = np.asarray(cv.KeyPoint_convert(fast.detect(image, None)), dtype=int)  # (col, row)
    #print(type(keypoints))

    keypoints = fast.detect(image, None)
    keypoints_2 = np.asarray(cv.KeyPoint_convert(fast.detect(image, None)), dtype=int)
    #print(keypoints_2)
    #k.response = strength of the keypoint, k.pt = (x, y) = (col, row)
    # show key points on image
    img2 = cv.drawKeypoints(image, keypoints=keypoints, outImage=None, color=(255,0,0))
    # cv.imshow('cam1_time0', img2)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    keypoints_2[:, [0, 1]] = keypoints_2[:, [1, 0]]  # swap columns: (col, row) -> (row, col)
    
    return keypoints_2


if __name__ == '__main__':
    image=cv.imread('static_img.jpg')
    detect_corners(image)
