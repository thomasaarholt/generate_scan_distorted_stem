import numpy as np
from scipy.optimize import curve_fit
from scipy.signal.windows import tukey
from numba import jit

try:
    #raise ValueError() # Uncomment this to test numpy-version on cupy-enabled systems
    import cupy as cp
    def asnumpy(arr):
        try:
            return cp.asnumpy(arr)
        except:
            return arr
    from cupyx.scipy.ndimage import affine_transform, zoom

except:
    print('No CuPy')
    cp = np
    def asnumpy(arr):
        return np.array(arr)
    from scipy.ndimage import affine_transform, zoom


def normalize_many(imgs, window):
    "Normalize images for hybrid correlation"
    axes = (-2,-1)
    return window * (
        imgs - np.expand_dims(
            (imgs * window).mean(axis=axes), axes) / np.mean(window)
    )

def normalize_one(imgs, window):
    "Normalize image for hybrid correlation"
    return window * (
        imgs - (imgs * window).mean() / np.mean(window)
    )

def tukey_window(shape, alpha=0.1):
    filt1 = tukey(shape[0], alpha=alpha, sym=True)
    if shape[0] == shape[1]:
        filt2 = filt1
    else:
        filt2 = tukey(shape[1], alpha=alpha, sym=True)
    return filt1[:, None] * filt2


def Gaussian2DFunc(YX, A=1, x0=0, y0=0, sigma=5, offset=0):
    y, x = YX
    return A*np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2)) + offset

def estimate_center(difference_img, shift1, shift2, subpixel_radius):
    "Gaussian curve fitting of difference image using in subpixel correlation"
    S1, S2 = np.meshgrid(shift1, shift2)
    parameters, _ = curve_fit(
        Gaussian2DFunc, 
        (S1.ravel(), S2.ravel()), 
        difference_img.ravel(), 
        p0 = (-1, 0, 0, subpixel_radius/2, difference_img.max()))
    _, p1, p2,*_ = parameters
    return (p1, p2), parameters

def make_zero_zero(arr, eps=1e-15):
    "cos(np.deg2rad(90)) will return a nonzero value. This sets such small values to zero."
    if isinstance(arr, (np.ndarray, cp.ndarray)):
        arr[np.abs(arr) < eps] = 0
    elif isinstance(arr, float):
        if abs(arr) < eps:
            arr = 0
    return arr

def swap_transform_standard(T):
    """Swaps the transformation matrix order from the regular one
    expected by skimage.transform.warp to the YXZ one expected by 
    scipy.ndimage.affine_transform
    """
    T = T.copy()
    T[:2,:2] = T[:2,:2][::-1,::-1]
    T[:2, 2] = T[:2, 2][::-1]
    return T


@jit(nopython=True)
def bilinear_weighting(points, intensities):
    size = len(points)
    topleft = np.zeros((size, 2), dtype=np.int64)
    xyindices = np.zeros((size, 2, 2, 2), dtype=np.int64)
    xyweights = np.zeros((size, 2,2))
    xyints = np.zeros((size, 2,2))
    for i, point in enumerate(points):
        x = point[0]
        y = point[1]
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1, y1 = x0 + 1, y0 + 1

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        topleft[i] = [x0, y0]
        xyindices[i] = np.array([[(x0,y0), (x1, y0)], [(x0, y1), (x1, y1)]])
        xyweights[i] = np.array([[wa, wc], [wb, wd]])
    xyints = xyweights * np.reshape(intensities, (-1, 1, 1))
    return xyints, xyweights, topleft, xyindices

@jit(nopython=True)
def bilinear_interpolation(indices, intensities):
    xyints, xyweights, topleft, xyindices = bilinear_weighting(indices, intensities)
    start_indices = xyindices.reshape(-1, 2).T
    start_indices = np.array([start_indices[0].min(), start_indices[1].min()])
    end_indices = xyindices.reshape(-1, 2).T
    end_indices = np.array([end_indices[0].max(), end_indices[1].max()])
    shape = end_indices - start_indices + 1
    shape = (shape[0], shape[1])
    
    intimg = np.zeros(shape)
    weightimg = np.zeros(shape)

    for tl, square_ints, square_weights in zip(topleft - start_indices, xyints, xyweights):
        xi, yi = tl
        intimg[xi:xi+2, yi:yi+2] += square_ints
        weightimg[xi:xi+2, yi:yi+2] += square_weights

    return intimg, weightimg