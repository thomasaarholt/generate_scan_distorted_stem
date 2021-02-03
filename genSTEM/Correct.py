import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from .utils import (
    cp, asnumpy, affine_transform, make_zero_zero, swap_transform_standard, tukey_window,
    normalize_one, normalize_many, estimate_center, Gaussian2DFunc)
from .Model import transform_points
from tqdm.auto import tqdm
from time import time
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.spatial import Delaunay
from sklearn.neighbors import KNeighborsRegressor
try:
    from interpolation.splines import eval_linear, eval_cubic, eval_cubic_numba
except:
    pass


def create_final_stack_and_average(images, scan_angles, best_str, best_angle, normalize_correlation = False, print_shifts = False, subpixel=False):
    """From an estimated drift speed and angle, produce a stack of transformed images and the stack average.

    # First warp images
    # Correct Shifts
    # 
    
    Parameters
    ----------
    images: ndarray
        Stack of images to use for reconstruction.
    scan_angles: list or array of float, int
        Scan rotation angles for the stack of images in degrees.
    best_str:
        
    best_angle:
        
    normalize_correlation:
        
    print_shifts:
        
    subpixel:
        
    """
    warped_images_with_nan = cp.stack([
        warp_image(
            img, 
            scanangle, 
            best_str, 
            best_angle,
            nan=True)
        for img, scanangle in zip(images, scan_angles)])

    warped_images_nonan = warped_images_with_nan.copy()
    warped_images_nonan[cp.isnan(warped_images_nonan)] = 0

    shifts, m = hybrid_correlation_images(warped_images_nonan)

    corrected_images = warped_images_with_nan.copy()
    corrected_images[1:] = cp.stack([translate(img, shift[::-1], cval=cp.nan) for img, shift in zip(warped_images_with_nan[1:], shifts[1:])])
    
    if subpixel:
        subpixel_images = []
        img1 = warped_images_nonan[0].copy()
        for img_nonan, img_nan, rough_shift in zip(warped_images_nonan[1:], warped_images_with_nan[1:], shifts[1:]):
            shift = cp.asarray(subpixel_correlation(img1, img_nonan))
            img2b = translate(img_nan, shift + rough_shift[::-1], cval=cp.nan)
            subpixel_images.append(img2b)
        warped_images_with_nan[1:] = cp.stack(subpixel_images)
        corrected_images = warped_images_with_nan
    mean_image = cp.nanmean(corrected_images, axis=0)

    return asnumpy(mean_image), shifts
    
def estimate_drift(images, scan_angles, tolerancy_percent=1, normalize_correlation=True, debug=False, correlation_power=0.8, low_drift=False, update_gcf=True):
    """Estimates the global, constant drift on an image using affine transformations.
    Takes as input a list of images and scan angles. Only tested with square images.

    Explores the space of possible transformations in 360 degree directions and drift 
    speeds that don't result in shears or stretches out of the image.

    Begins by testing drift at 0, 90, 180 and 270 degree intervals, and converges
    towards the best fit.
    """
    t1 = time()
    angle_steps = 8
    angle_low, angle_high = 0, 360 - 360/angle_steps

    xlen = images.shape[1]
    str_steps = 8
    if low_drift:
        str_low, str_high = 0, (1 / xlen) / 100000
    else:
        str_low, str_high = 0, 1 / xlen

    print("Getting fit with no drift first")
    all_maxes, pairs = warp_and_correlate(
        images, 
        scan_angles, 
        [0], [0], 
        correlation_power=correlation_power)

    best_angle, best_str, fit_value = 0, 0, all_maxes.max()
    history = [(best_angle, best_str, fit_value)]

    old_values = fit_value
    new_values = np.inf
    i = 0
    j = 0 # "extra" iterations when something just needed readjusting
    converged = False
    converged_for_two_iterations = False
    
    try:
        while i < 2 or not converged_for_two_iterations:
            print()
            print("Iteration #{}: Best guess: Angle: {:.2f}°, Speed: {:.2e}. Fit value: {:.2e}".format(
                i+j, best_angle, best_str, fit_value)
                + "\nLimits: [{:.1f}°, {:.1f}°], [{:.2e}, {:.2e}]".format(
                    angle_low % 360, angle_high % 360, str_low, str_high))
            old_values = new_values

            drift_angles = np.linspace(angle_low, angle_high, angle_steps)

            angle_diff = (drift_angles[1] - drift_angles[0])

            drift_speeds = np.linspace(str_low, str_high, str_steps)
            str_diff = (drift_speeds[1] - drift_speeds[0])

            best_angle, best_str, fit_value = get_best_angle_speed_shifts(
                images,
                scan_angles,
                drift_angles, 
                drift_speeds,
                correlation_power=correlation_power,
                image_number=i+j,
                update_gcf=update_gcf,
            )
            history.append((best_angle, best_str, fit_value))
            new_values = fit_value

            if debug:
                warped_images_nonan = [
                warp_image(
                    img, 
                    scanangle, 
                    best_str, 
                    best_angle,
                    nan=False) for img, scanangle in zip(images, scan_angles)]
                fig = plt.figure(figsize=(6,3))
                fig.clear()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(asnumpy(warped_images_nonan[0]))
                ax2.imshow(asnumpy(warped_images_nonan[1]))
                fig.tight_layout()
                fig.canvas.draw()
                
            if best_str in drift_speeds[:2]:
                print("Initial drift speed intervals were too large. Reducing.")
                str_low = 0
                str_high = drift_speeds[2].item()
                j += 1
                continue


            if best_str in drift_speeds[-2:]:
                print("Initial drift speed intervals were too small. Increasing.")
                str_low = drift_speeds[-2].item()
                str_high = 10 * str_high
                j += 1
                continue
            
            if i > 0 and best_angle in drift_angles[:1]:
                print("Best angle is lower angle regime. Rotating drift by three steps.")
                angle_low = (angle_low - 3*angle_diff)
                angle_high =  (angle_high - 3*angle_diff)
                j += 1
                continue

            if i > 0 and best_angle in drift_angles[-2:]:
                print("Best angle is upper angle regime. Rotating drift by three steps.")
                angle_low = (angle_low + 3*angle_diff)
                angle_high =  (angle_high + 3*angle_diff)
                j += 1
                continue

            if i % 2:
                print("Adjusting drift speed limits")
                str_low = best_str - 2*str_diff
                str_high = best_str + 2*str_diff

                if str_low < 0:
                    str_low = 0
            else:
                print("Adjusting drift angle limits")
                angle_low = best_angle - 2*angle_diff
                angle_high = best_angle + 2*angle_diff



            i += 1
            converged_for_two_iterations = converged 
            converged = drift_values_converged(old_values, new_values, tolerancy_percent=tolerancy_percent)
            converged_for_two_iterations = converged_for_two_iterations and converged

    except KeyboardInterrupt:
        print(f"Aborted after {i} iterations")
        pass
    print("Final iteration: Final guess: Angle: {}°, Speed: {:.2e}".format(
            np.round(best_angle, 2).item(), best_str.item()))
    print(f"\nTook {round(time() - t1, 1)} seconds")
        
    return best_angle, best_str, np.array(history)




def hybrid_correlation(
    img1, 
    img2,
    p=0.8,
    normalize = True, 
    window="tukey",
    window_strength=0.1,
    fit_only = False,
    ):
    """Performs hybrid correlation on two images.

    fit_only will only return the correlation maximum value, 
    not the required shift.
    """

    if window == "tukey":
        window = cp.asarray(tukey_window(img1.shape, alpha=window_strength))
    else:
        window = 1

    if normalize:
        img1, img2 = normalize_many(cp.stack([img1, img2]), window)
    else:
        img1 = img1 * window
        img2 = img2 * window

    padsize = img1.shape[0] // 4
    img1 = cp.pad(img1, padsize)
    img2 = cp.pad(img2, padsize)

    fftimg1 = cp.fft.rfft2(img1)
    fftimg2 = cp.fft.rfft2(img2)

    
    m = fftimg1 * cp.conj(fftimg2)
    corr =  cp.fft.irfft2(cp.abs(m)**p * cp.exp(1j * cp.angle(m)))

    if fit_only:
        return corr.max()

    corr = cp.fft.fftshift(corr)
    translation = cp.array(cp.unravel_index(corr.argmax(), corr.shape))
    center = cp.array(corr.shape) // 2
    return [x.item() for x in translation - center], corr.max()

def get_tukey_window(shape, alpha=0.1):
    return cp.asarray(tukey_window(shape, alpha=alpha))

def prepare_correlate(images, window="tukey", window_strength=0.1):
    """Prepare images for hybrid correlation by windowing, normalizing, padding
     and taking the fft
    """
    if window == "tukey":
        window = get_tukey_window(images.shape[-2:], window_strength)
    elif window is None:
        window = 1
    # Else window can be a finished window object

    images = normalize_many(images, window)
    padsize1 = images.shape[1] // 4
    padsize2 = images.shape[2] // 4

    leading_dims = len(images.shape) - 2

    images = cp.pad(images, leading_dims * ((0,0),) + ((padsize1, padsize1), (padsize2, padsize2)))
    fftimages = cp.fft.rfft2(images)
    return fftimages

def hybrid_correlation_prefft_all(fftimages, p=0.8, fit_only=False):
    """Fastest implementation, compares first image to rest. Could consider performing it 
    across all permutations of images.
    """
    fimg0 = fftimages[0]
    restimages = fftimages[1:]
    m = fimg0 * cp.conj(restimages)
    corr =  cp.fft.irfft2(cp.abs(m)**p * cp.exp(1j * cp.angle(m)))
    score = corr.max(axis=(-2, -1))

    if fit_only:
        return score.mean(0)
    else:
        corr = cp.fft.fftshift(corr, axes=(-2,-1))
        center = cp.array(corr.shape[1:]) // 2
        max_coord = cp.array(cp.unravel_index(corr.argmax(axis=(-2, -1)), corr[0].shape), dtype=int).T
        shift = max_coord - center
        shifts = cp.zeros((len(fftimages), 2))
        shifts[1:] = shift
        return shifts, score.mean()

def hybrid_correlation_images(images, p=0.8, fit_only=False, window="tukey", window_strength=0.1):
    images_fft = prepare_correlate(images, window=window, window_strength=window_strength)
    result = hybrid_correlation_prefft_all(images_fft, p=p, fit_only=fit_only)
    if fit_only:
        score = result
        return score
    else:
        shift, score = result
        return shift, score

def hybrid_correlation_prefft(
    fftimg1, 
    fftimg2,
    p=0.8,
    fit_only = False,
    ):
    """Performs hybrid correlation on two images.

    fit_only will only return the correlation maximum value, 
    not the required shift.
    """

    m = fftimg1 * cp.conj(fftimg2)
    corr =  cp.fft.irfft2(cp.abs(m)**p * cp.exp(1j * cp.angle(m)))

    if fit_only:
        return corr.max()

    corr = cp.fft.fftshift(corr)
    translation = cp.array(cp.unravel_index(corr.argmax(), corr.shape))
    center = cp.array(corr.shape) // 2
    return [x.item() for x in translation - center], corr.max()

def translate(img, shift, cval=0):
    sx, sy = shift
    if cval is True:
        cval = np.nan
    return affine_transform(
        img,
        cp.asarray(
            np.linalg.inv(
                swap_transform_standard(
                    Affine2D().translate(sx, sy).get_matrix()))), 
        order=1,
        cval=cval)

def transform_drift_scan(scan_angle=0, drift_angle=0, drift_speed=0, xlen=100):
    '''Generates the transformation matrix, T, which includes the transformation from scan rotation, Tr, and drift, Tt.
    
    Parameters
    ----------
    scan_angles: float, int
        Scan rotation angles for the stack of images in degrees.
    drift_angle: float, int
        Angle of drift in degrees.
    drift_speed: float, int
        
    '''
    arr = np.zeros(np.shape(drift_speed) + np.shape(drift_angle) + (3,3))
    if np.shape(drift_speed):
        drift_speed = drift_speed[:, None, None]
    angle = np.deg2rad(drift_angle + scan_angle)
    arr[...] = np.eye(3)
    s_sin = drift_speed*(np.sin(angle)) 
    s_cos = drift_speed*(np.cos(angle))
    arr[..., 0,0] = 1 - s_cos
    arr[..., 0,1] = -s_cos*xlen
    arr[..., 0,2] = -s_cos
    arr[..., 1,0] = -s_sin
    arr[..., 1,1] = 1 - s_sin*xlen
    arr[..., 1,2] = -s_sin
    return arr.squeeze()

def warp_images(images, scan_angles, drift_speed=0, drift_angle=0, nan=False):
    warped_images = [warp_image(img, scan, drift_speed, drift_angle, nan=nan) for img, scan in zip(images, scan_angles)]
    return cp.array(warped_images)

def warp_and_shift_images(images, scan_angles, drift_speed=0, drift_angle=0, nan=False):
    warped_images = warp_images(images, scan_angles, drift_speed, drift_angle, nan=False)
    shifts, _ = hybrid_correlation_images(warped_images)
    if nan:
        warped_images = warp_images(images, scan_angles, drift_speed, drift_angle, nan=nan)

    translated_images = cp.zeros(images.shape)
    for i, (img, shift) in enumerate(zip(warped_images, shifts)):
        translated_images[i] = translate(img, shift[::-1], cval=nan)
    return translated_images, shifts

def add_shifts_and_rotation_to_transform(transform, image_shape, scan_rotation_deg, post_shifts=[0,0]):
    shift1, shift2 = (np.array(image_shape) - 1) / 2
    post_shift1, post_shift2 = post_shifts[::-1]

    T = (
        Affine2D().translate(-shift1, -shift2)
        + Affine2D(transform)
        .rotate_deg(scan_rotation_deg)
        .translate(shift1, shift2)
        .translate(post_shift1, post_shift2)
    )
    return T


def add_shifts_only_to_transform(transform, image_shape, scan_rotation_deg, ):#post_shifts=[0,0]):
    shift1, shift2 = (np.array(image_shape) - 1) / 2
    #post_shift1, post_shift2 = post_shifts[::-1]

    T = (
        Affine2D().translate(-shift1, -shift2)
        + Affine2D(transform)
        #.rotate_deg(scan_rotation_deg)
        .translate(shift1, shift2)
        #.translate(post_shift1, post_shift2)
    )
    return T

def warp_image(img, scanrotation, drift_speed, drift_angle, nan=False, post_shifts=[0,0]):
    drift_transform = transform_drift_scan(
         -scanrotation,
         drift_angle, 
         drift_speed, 
         img.shape[-2])
    T = add_shifts_and_rotation_to_transform(drift_transform, img.shape, scanrotation, post_shifts)
    cval = cp.nan if nan else 0 # value of pixels that were outside the image border
    T2 = swap_transform_standard(T.get_matrix()).copy()
    T3 = cp.array(T2)
    T4 = cp.linalg.inv(T3)
    return affine_transform(
        img, 
        T4,
        order=1,
        cval=cval)

def warp_and_correlate(images, scan_angles, drift_angles, speeds, correlation_power=0.8):
    scores = cp.zeros((len(drift_angles), len(speeds)))
    pairs = []

    for ai, drift_angle in tqdm(
        enumerate(drift_angles), 
        desc="Iterating through drift angles", 
        total=len(drift_angles),
        leave=True):
        
        for si, speed in tqdm(enumerate(speeds), desc="Iterating through drift speeds", 
            total=len(speeds), leave=False):
            wimages = warp_images(images, scan_angles, speed, drift_angle)
            score = hybrid_correlation_images(wimages, p=correlation_power, fit_only=True)
            scores[ai, si] = score
            pairs.append((drift_angle, speed))
    return asnumpy(scores), np.array(pairs)

def get_best_angle_speed_shifts(images, scan_angles, drift_angles, drift_speeds, correlation_power=0.8, image_number=0, update_gcf=True):
    
    all_maxes, pairs = warp_and_correlate(images, scan_angles, drift_angles, drift_speeds, correlation_power=correlation_power)
    angle_fit = all_maxes.mean(axis=(1))
    str_fit = all_maxes.mean(axis=(0))

    s, a = np.meshgrid(drift_speeds*images[0].shape[0], np.deg2rad(drift_angles))
    
    m = all_maxes

    maxes = all_maxes.flatten()
    i = maxes.argmax() # take the mean over each set of images
    best_drift_angle, best_drift_speed = pairs[i]

    if update_gcf:
        fig = plt.gcf()
        fig.clear()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133, projection="polar")
        
        ax1.scatter(drift_angles, angle_fit)
        ax2.plot(drift_speeds*images.shape[1], str_fit)
        ax3.pcolormesh(a,s,m, shading="nearest", cmap='jet')

        x, y = np.deg2rad(best_drift_angle), drift_speeds[0].item() * images.shape[1]
        dx, dy = 0, best_drift_speed* images.shape[1] - y
        ax1.set(title='Testing angles', xlabel='Angles (°)', ylabel='Fit (Average over drift speeds)')
        ax2.set(title='Testing drift speeds', xlabel='Drift Speed (A.U.)', ylabel='Fit (Average over drift angles)')
        ax3.set(title='Radial fit representation', ylim = (s.min(), s.max()), yticklabels=[])
        ax3.annotate("", xy=(x+dx, y+dy), xytext=(x, y), arrowprops=dict(color="k"))
        fig.tight_layout()
        fig.canvas.draw()
    return best_drift_angle, best_drift_speed, maxes[i]

def drift_values_converged(old_values, new_values, tolerancy_percent=1):
    tol = tolerancy_percent/100
    return old_values * (1 + tol) >= new_values >= old_values * (1 - tol)

def subpixel_correlation_shift(img1, img2, rough_shift=(0,0), subpixel_radius=1, steps=11, window=True, window_strength=0.8, debug=False):
    "Returns the value of shift needed to be fed to shift img2 onto img1 using Correct.translate"
    rough1, rough2 = rough_shift
    if window:
        window = cp.asarray(tukey_window(img1.shape, window_strength))
    else:
        window = 1

    img1, img2 = normalize_many(cp.stack([img1, img2]), window)
    
    shift1 = np.linspace(-subpixel_radius, subpixel_radius, steps)
    shift2 = np.linspace(-subpixel_radius, subpixel_radius, steps)

    difference_img = cp.zeros((shift1.shape + shift2.shape))
    for i1, d1 in enumerate(shift1):
        for i2, d2 in enumerate(shift2):
            img2_b = translate(img2, (rough1 + d1, rough2 + d2), cval=cp.nan)
            mask = ~np.isnan(img2_b) # only non-nan indices
            diff = np.abs(img1 - img2_b)[mask]
            difference_img[i1, i2] = np.mean(diff)
            
    difference_img = asnumpy(difference_img)

    # Tried using 2D gaussian fitting on the difference image, but just choosing the point of
    # least difference is more stable and good enough
    #(p1, p2), parameters = estimate_center(difference_img, shift1, shift2, subpixel_radius)
    i = np.argmin(difference_img)
    i1, i2 = np.unravel_index(i, difference_img.shape)
    p1, p2 = shift1[i1], shift2[i2]
    shift = (rough1 + p1, rough2 + p2)
    if debug:
        S1, S2 = np.meshgrid(shift1, shift2)
        plt.figure()
        plt.imshow(difference_img, extent=(-subpixel_radius, subpixel_radius, subpixel_radius, -subpixel_radius))
        plt.contour(S1, S2, Gaussian2DFunc((S1, S2), *parameters), 8, colors='w')
        print(f"Subpixel offsets: p1={round(p1,2)}, p2={round(p2, 2)}")
    return shift

def subpixel_correlation(img1, img2, subpixel_radius=2.5, steps=11, window=True, window_strength=0.8):
    "Estimates pixel shift using hybrid correlation, then manually compares the differences by subpixel translation of the image"
    rough_shift, m = hybrid_correlation(img1, img2)
    shift = subpixel_correlation_shift(img1, img2, rough_shift=rough_shift, subpixel_radius = subpixel_radius, steps=steps, window=window, window_strength=window_strength)
    return shift

def interpolate_image_to_new_position(img: "(Y, X)", points: "(N, 2) or (Y, X, 2)", fill_value=np.nan, mode='linear'):
    """Warp an image to new positions given by a list of coordinates that has the same length 
    as the image has pixels
    
    Parameters
    ----------
    img: Image of shape (Y, X)
    points: Array of points of shape (N, 2), where the last two indices are in traditional (X,Y) convention
    fill_value: Value of points interpolated outside the grid of the image.
        False or float or np.nan
        False follows interpolate.py behaviour.
    """
    # Grid probably becomes a linspace'd array:
    grid = ((0, img.shape[0]-1, img.shape[0]), (0, img.shape[1]-1, img.shape[1]))
    points = points[::-1] # Swap coordinates to (Y, X) convention
    points = points.reshape((2, -1)).T
    if not points.flags.c_contiguous:
        points = np.ascontiguousarray(points)
    if mode == 'linear':
        interpolated_values = eval_linear(grid, img, points)

    elif mode == 'cubic':
        from interpolation.splines import filter_cubic
        coeff = filter_cubic(grid, img) # a 12x12 array
        interpolated_values = eval_cubic(grid, coeff, points)
    else:
        raise ValueError('That mode is wrong.')
    if fill_value is not False and fill_value is not None:
        mask = np.any((points >= img.shape) | (points < 0.), axis=1)
        interpolated_values[mask] = fill_value
    return interpolated_values.reshape(img.shape)


def bilinear_bincount_numpy(points, intensities):
    """Bilinear weighting of points onto a grid.
    Extent of grid given by min and max of points in each dimension
    points should have shape (N, 2)
    intensity should have shape (N,)
    """
    floor = np.floor(points)
    ceil = floor + 1
    floored_indices = np.array(floor, dtype=int)
    low0, low1 = floored_indices.min(0)
    high0, high1 = floored_indices.max(0)
    floored_indices = floored_indices - (low0, low1)
    shape = (high0 - low0 + 2, high1-low1 + 2)

    upper_diff = ceil - points
    lower_diff = points - floor

    w1 = np.prod((upper_diff), axis=1)
    w2 = upper_diff[:,0]*lower_diff[:,1]
    w3 = lower_diff[:,0]*upper_diff[:,1]
    w4 = np.prod((lower_diff), axis=1)

    shifts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    indices = floored_indices[:, None] + shifts
    indices = (indices * (shape[1], 1)).sum(-1)
    weights = np.array([w1, w2, w3, w4]).T

    weight_bins = np.bincount(indices.flatten(), weights=weights.flatten(), minlength = np.prod(shape))
    intens_bins = np.bincount(indices.flatten(), weights=(intensities[:, None]*weights).flatten(), minlength = np.prod(shape))

    weight_image = weight_bins.reshape(shape)
    intens_image = intens_bins.reshape(shape)
    return intens_image, weight_image

def bilinear_bincount_cupy(points, intensities, subpixel=1):
    """Bilinear weighting of points onto a grid.
    Extent of grid given by min and max of points in each dimension
    points should be a cupy array of shape (N, 2)
    intensity should be a cupy array of shape (N,)
    """

    points = subpixel * points
    floor = cp.floor(points)
    ceil = floor + 1
    floored_indices = cp.array(floor, dtype=int)
    low0, low1 = floored_indices.min(0)
    high0, high1 = floored_indices.max(0)
    floored_indices = floored_indices - cp.array([low0, low1])
    shape = cp.array([high0 - low0 + 2, high1-low1 + 2])

    upper_diff = ceil - points
    lower_diff = points - floor

    w1 = upper_diff[:, 0] * upper_diff[:, 1]
    w2 = upper_diff[:, 0] * lower_diff[:, 1]
    w3 = lower_diff[:, 0] * upper_diff[:, 1]
    w4 = lower_diff[:, 0] * lower_diff[:, 1]

    shifts = cp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    indices = floored_indices[:, None] + shifts
    indices = (indices * cp.array([shape[1].item(), 1])).sum(-1)
    weights = cp.array([w1, w2, w3, w4]).T

    weight_bins = np.bincount(indices.flatten(), weights=weights.flatten(), minlength = np.prod(shape).item())
    intens_bins = np.bincount(indices.flatten(), weights=(intensities[:, None]*weights).flatten(), minlength = np.prod(shape).item())

    weight_image = weight_bins.reshape(shape.get())
    intens_image = intens_bins.reshape(shape.get())
    return intens_image, weight_image

def get_indices_of_non_parallel_images(scanangles):
    "For each scan rotation, get the indices of the other scanangles that are not parallel to it"
    images_non_parallel_indices = []
    for current_angle in scanangles:
        non_parallel_indices = []
        for index, angle in enumerate(scanangles):
            if angle != current_angle and angle != (current_angle + 180) % 360:
                non_parallel_indices.append(index)
        images_non_parallel_indices.append(non_parallel_indices)
    images_non_parallel_indices = np.array(images_non_parallel_indices)
    return images_non_parallel_indices

def abs_difference(original, reference):
    """
    Get absolute difference between two images/rows that may contain nans
    should not be used on individual pixels, since one pixel may be nan
    and the other not
    """
    shape = reference.shape
    reference = reference.reshape((-1, shape[-1]))
    return np.nanmean(np.abs(original - reference), axis=-1).reshape(shape[:-1])

def make_final_image(images, transforms, corrected_indices, subpixel_factor=1, offset=(0,0), method="linear", neighbors=8, neighbor_weights="distance"):
    "Should add option to specify which positions one wants to interpolate"
    points = []
    for i in range(len(images)):
        points.append(transform_points(corrected_indices[i], swap_transform_standard(transforms[i])))
    points = np.array(points)
    points = np.swapaxes(points, 0, 1).reshape((2,-1)).T
    intensities = asnumpy(images.flatten())
    func = get_interpolation_function(points, intensities, method=method, neighbors=neighbors, neighbor_weights=neighbor_weights)

    indices = np.mgrid[
        offset[0]: offset[0] + images.shape[1]-1/subpixel_factor:subpixel_factor*images.shape[1]*1j, 
        offset[1]: offset[1] + images.shape[2]-1/subpixel_factor:subpixel_factor*images.shape[2]*1j
    ]
    final_img = func(indices.reshape((2, -1)).T).reshape(indices.shape[1:])
    return final_img




def get_row_shifts(images, interpolation_functions, transforms, image_indices=None, max_pixel_shift=1, steps=11):
    if image_indices is None:
        indices = np.indices(images.shape[1:])
        image_indices = np.stack(len(images)*[indices]).astype(float)
    deltarange = np.linspace(-max_pixel_shift, max_pixel_shift, steps)

    image_row_shifts = []
    for i in tqdm(range(len(images)), desc="Calculating row shift"):
        row_shifts = []
        for row_index in np.arange(images.shape[-2]):
            original_image_row = asnumpy(images[i, row_index])
            diffs = []
            for deltaRow in deltarange:
                row_indices = transform_points(image_indices[i, :, row_index] + np.array([0, deltaRow])[:, None], swap_transform_standard(transforms[i])).T
                reference_image_row = interpolation_functions[i](row_indices)
                diff = abs_difference(original_image_row, reference_image_row)
                diffs.append(diff)
            if np.isnan(diffs).all():
                row_shifts.append(0.)
                continue
            min_index = np.nanargmin(diffs)
            shift = deltarange[min_index]
            row_shifts.append(shift)
        image_row_shifts.append(row_shifts)
    image_row_shifts = np.array(image_row_shifts)
    return image_row_shifts


def get_row_col_shifts(images, interpolation_functions, transforms, image_indices=None, max_pixel_shift=1, steps=11):
    if image_indices is None:
        indices = np.indices(images.shape[1:])
        image_indices = np.stack(len(images)*[indices]).astype(float)
    deltarange = np.linspace(-max_pixel_shift, max_pixel_shift, steps)

    image_row_shifts = []
    for i in tqdm(range(len(images)), desc="Calculating row shift"):
        row_shifts = []
        for row_index in np.arange(images.shape[-2]):
            original_image_row = asnumpy(images[i, row_index])
            diff_row = []
            for delta_row in deltarange:
                diff_col = []
                for delta_col in deltarange:
                    row_indices = transform_points(image_indices[i, :, row_index] + np.array([delta_col, delta_row])[:, None], swap_transform_standard(transforms[i])).T
                    reference_image_row = interpolation_functions[i](row_indices)
                    diff = abs_difference(original_image_row, reference_image_row)
                    diff_col.append(diff)
                diff_row.append(diff_col)
            diff_row = np.array(diff_row)
            if np.isnan(diff_row).all():
                row_shifts.append([0., 0.])
                continue
            min_index = np.nanargmin(diff_row)
            irow, icol = np.unravel_index(min_index, diff_row.shape)
            shift = [deltarange[icol], deltarange[irow]]
            row_shifts.append(shift)
        image_row_shifts.append(row_shifts)
    image_row_shifts = np.array(image_row_shifts)
    return np.swapaxes(image_row_shifts, -1, -2)

def get_row_col_shifts2(images, interpolation_functions, transforms, image_indices=None, max_pixel_shift=1, steps=11):
    if image_indices is None:
        indices = np.indices(images.shape[1:])
        image_indices = np.stack(len(images)*[indices]).astype(float)
    deltarange = np.linspace(-max_pixel_shift, max_pixel_shift, steps)

    image_row_shifts = np.zeros((len(images), 2, images.shape[-2]))
    for i, image in enumerate(tqdm(asnumpy(images), desc="Calculating row shift")):
        #image = cp.asarray(image) # move to GPU if available, for faster differencing below
        row_shifts = []
        for row_index in np.arange(image.shape[-2]):
            original_image_row = image[row_index]

            shifted_indices = []
            for delta_row in deltarange:
                shifted_indices_cols = []
                for delta_col in deltarange:
                    shifted_indices_cols.append(image_indices[i, :, row_index] + np.array([delta_col, delta_row])[:, None])
                shifted_indices.append(shifted_indices_cols)
            shifted_indices = np.array(shifted_indices)
            shifted_indices = np.swapaxes(shifted_indices, -1, -2)
            shape = shifted_indices.shape

            shifted_indices = shifted_indices.reshape((-1, 2)).T
            row_indices = asnumpy(transform_points(shifted_indices, swap_transform_standard(transforms[i])).T) ###
            reference_image_rows = interpolation_functions[i](row_indices).reshape(shape[:-1])
            diff = abs_difference(original_image_row, reference_image_rows)
            nan_mask = np.isnan(diff)
            # If the difference between real image and reference image, without shift == nan
            # Or if all differences are nan
            # Then don't shift the row
            if nan_mask[(steps - 1) // 2, (steps - 1) // 2] or np.all(nan_mask):
                shift = [0.,0.]
            else:
                min_index = np.nanargmin(diff)
                irow, icol = np.unravel_index(min_index.item(), diff.shape)
                shift = [deltarange[icol], deltarange[irow]]
            image_row_shifts[i, 0, row_index] = shift[0]
            image_row_shifts[i, 1, row_index] = shift[1]

    return asnumpy(image_row_shifts)


def get_row_col_shifts3(images, interpolation_functions, transforms, image_indices=None, max_pixel_shift=1, steps=3):
    indices = np.indices(images.shape[1:])
    if image_indices is None:
        image_indices = np.stack(len(images)*[indices]).astype(float)
    deltarange = np.linspace(-max_pixel_shift, max_pixel_shift, steps)

    image_row_shifts = np.zeros((len(images), 2, images.shape[-2]))
    for i, image in enumerate(tqdm(images, desc="Calculating row shift")):
        #image = cp.asarray(image) # move to GPU if available, for faster differencing below
        shi = np.zeros((2,) + (steps, steps) + images.shape[1:])
        ref = np.zeros((steps, steps) + images.shape[1:])

        shi = shi + image_indices[i][:, None, None].astype(float) + np.array(np.meshgrid(deltarange, deltarange))[..., None, None]        

        for row_index in np.arange(image.shape[-2]):

            shifted_indices = []
            for delta_row in deltarange:
                shifted_indices_cols = []
                for delta_col in deltarange:
                    shifted_indices_cols.append(image_indices[i, :, row_index] + np.array([delta_col, delta_row])[:, None])
                shifted_indices.append(shifted_indices_cols)
            shifted_indices = np.array(shifted_indices)
            shifted_indices = np.swapaxes(shifted_indices, -1, -2)
            shape = shifted_indices.shape

            shifted_indices = shifted_indices.reshape((-1, 2)).T
            row_indices = asnumpy(transform_points(shifted_indices, swap_transform_standard(transforms[i])).T) ###
            reference_image_rows = interpolation_functions[i](row_indices).reshape(shape[:-1])
            diff = abs_difference(original_image_row, reference_image_rows)
            nan_mask = np.isnan(diff)
            # If the difference between real image and reference image, without shift == nan
            # Or if all differences are nan
            # Then don't shift the row
            if nan_mask[(steps - 1) // 2, (steps - 1) // 2] or np.all(nan_mask):
                shift = [0.,0.]
            else:
                min_index = np.nanargmin(diff)
                irow, icol = np.unravel_index(min_index.item(), diff.shape)
                shift = [deltarange[icol], deltarange[irow]]
            image_row_shifts[i, 0, row_index] = shift[0]
            image_row_shifts[i, 1, row_index] = shift[1]

    return asnumpy(image_row_shifts)


def get_interpolated_functions_and_transforms(images, scanangles, best_angle, best_speed, post_shifts, image_indices=None, method="linear", neighbors=8):
    other_indices = get_indices_of_non_parallel_images(scanangles)
    indices = np.indices(images.shape[1:])
    if image_indices is None:
        image_indices = np.stack(len(images)*[indices]).astype(float)
    image_indices = np.asarray(image_indices)
    forward_transformed_coords = []
    intensities = []
    funcs = []
    transforms = []

    for i, image in enumerate(images):
        T = transform_drift_scan(-scanangles[i], best_angle, best_speed, images.shape[-2])
        T = add_shifts_and_rotation_to_transform(
            T, image.shape, scan_rotation_deg=scanangles[i], post_shifts=post_shifts[i]).get_matrix()

        forward_coords = transform_points(image_indices[i], swap_transform_standard(T))
        transforms.append(T)
 
        # input to interpolation must be numpy not cupy
        forward_transformed_coords.append(asnumpy(forward_coords))
        intensities.append(asnumpy(image))
    transforms = np.array(transforms)

    forward_transformed_coords = np.array(forward_transformed_coords)
    intensities = np.array(intensities)

    for i in tqdm(range(len(images)), desc='Creating interpolation functions'):
        coords = np.zeros((len(other_indices[i]),) + (2,) + images.shape[1:])
        intens = np.zeros((len(other_indices[i]),) + images.shape[1:] )
        for index, j in enumerate(other_indices[i]):
            coords[index] = forward_transformed_coords[j]
            intens[index] = intensities[j]
        points = np.swapaxes(coords, 0, 1).reshape((2,-1)).T
        func = get_interpolation_function(points, intens, method, neighbors)
        funcs.append(func)
    return funcs, transforms

def get_interpolation_function(points, intensities, method='linear', neighbors=8, neighbor_weights="distance"):
    if method == 'linear':
        tesselation = Delaunay(points)
        func = LinearNDInterpolator(tesselation, intensities.flatten(), fill_value=np.nan)
    elif method == 'nearest':
        func = NearestNDInterpolator(points, intensities.flatten())
    elif method == 'cubic':
        tesselation = Delaunay(points)
        func = CloughTocher2DInterpolator(tesselation, intensities.flatten(), fill_value=np.nan)
    elif method == 'neighbor':
        nn = KNeighborsRegressor(n_neighbors=neighbors, weights=neighbor_weights)
        reg = nn.fit(points, intensities.flatten())
        func = reg.predict
    return func

def bilinear_2(points, intensity):
    # Create empty matrices, starting from 0 to p.max
    w = np.zeros((points[:, 0].max().astype(int) + 2, points[:, 1].max().astype(int) + 2))
    i = np.zeros_like(w)
    
    # Calc weights
    floor = np.floor(points)
    ceil = floor + 1
    upper_diff = ceil - points
    lower_diff = points - floor
    w1 = upper_diff[:, 0] * upper_diff[:, 1]
    w2 = upper_diff[:, 0] * lower_diff[:, 1]
    w3 = lower_diff[:, 0] * upper_diff[:, 1]
    w4 = lower_diff[:, 0] * lower_diff[:, 1]
    
    # Get indices
    ix, iy = floor[:, 0].astype(int), floor[:, 1].astype(int)

    # Use np.add.at. See, https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
    np.add.at(w, (ix, iy), w1)
    np.add.at(w, (ix, iy+1), w2)
    np.add.at(w, (ix+1, iy), w3)
    np.add.at(w, (ix+1, iy+1), w4)

    np.add.at(i, (ix, iy), w1 * intensity)
    np.add.at(i, (ix, iy+1), w2 * intensity)
    np.add.at(i, (ix+1, iy), w3 * intensity)
    np.add.at(i, (ix+1, iy+1), w4 * intensity)
    
    # Clip (to accomodate image size to be the same as your bilinear function)
    iix, iiy = points[:, 0].min().astype(int), points[:, 1].min().astype(int)
    i, w = i[iix:, iiy:], w[iix:, iiy:]
    return i, w