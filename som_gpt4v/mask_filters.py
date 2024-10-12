import numpy as np
from PIL import Image

def corner_filter(mask):
    '''If the mask contains a corner, return False'''
    return not(mask[0, 0] or mask[0, -1] or mask[-1, 0] or mask[-1, -1])

def area_filter(mask, area_thresh):
    '''If the mask area is smaller than area_thresh, return False'''
    return mask.sum() > area_thresh

def intersection_filter(mask1, mask2, intersection_thresh):
    '''If the intersection of mask1 and mask2 is smaller than intersection_thresh, return False'''
    # resize mask2 to the same size as mask1
    mask2 = np.array(Image.fromarray(mask2).resize(mask1.shape[1::-1]))
    # calculate the intersection area
    intersection = np.logical_and(mask1, mask2).sum()
    return intersection / mask1.sum() > intersection_thresh

def get_mask_filter(corner, area, intersection, area_thresh=100, mask2=None, intersection_thresh=0.1):
    def mask_filter(output):
        mask = output['segmentation']
        if corner and not corner_filter(mask):
            return False
        if area and not area_filter(mask, area_thresh):
            return False
        if intersection and not intersection_filter(mask, mask2, intersection_thresh):
            return False
        return True
    return mask_filter


if __name__ == '__main__':
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:8, 5:15] = True
    output = {'segmentation': mask}
    mask2 = np.zeros((20, 40), dtype=bool)
    mask2[4:16, 20:40] = True
    flt = get_mask_filter(True, True, True, area_thresh=10, mask2=mask2)
    print(flt(output))