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

try:
    from interpolation.splines import eval_linear, eval_cubic, eval_cubic_numba
except:
    print("Scanline development only: Missing the package 'interpolation.py' - currently do `conda install interpolation numba=0.49` due to some dev issues on their part")


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

    warped_imgA = warped_images_nonan[0]
    shifts = [(0,0)] # shift of the first image, which the rest are relative to
    for warped_imgB in warped_images_nonan[1:]:
        shift, m = hybrid_correlation(warped_imgA, warped_imgB, normalize=normalize_correlation)
        shifts.append(shift)
    shifts = cp.array(np.array(shifts))


    corrected_images = warped_images_with_nan.copy()
    corrected_images[1:] =  cp.stack([translate(img, shift[::-1], cval=cp.nan) for img, shift in zip(warped_images_with_nan[1:], shifts[1:])])
    
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
    
def estimate_drift(images, scan_angles, tolerancy_percent=1, normalize_correlation=True, debug=False, correlation_power=0.8, low_drift=False):
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
        str_low, str_high = 0, (1 / xlen) / 1000
    else:
        str_low, str_high = 0, 1 / xlen

    print("Getting fit with no drift first")
    all_maxes, pairs = warp_and_correlate(
        images, 
        scan_angles, 
        [0], [0], 
        normalize_correlation=normalize_correlation, 
        correlation_power=correlation_power)
    maxes = all_maxes.reshape((-1, len(images)-1)).mean(1)

    best_angle, best_str, fit_value = 0, 0, maxes.max()
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
            print("Iteration #{}: Best guess: Angle: {}°, Speed: {:.2e}. Fit value: {:.2e}".format(
                i+j, np.round(best_angle, 2), np.round(best_str, 5), fit_value)
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
                normalize_correlation=normalize_correlation,
                correlation_power=correlation_power,
                image_number=i+j,
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
    window=True,
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
        #img1 = (img1 - np.sum(img1 * window) / np.sum(window) ) * window
        #img2 = (img2 - np.sum(img2 * window) / np.sum(window) ) * window
        #img1 = normalize_one(img1, window)
        #img2 = normalize_one(img2, window)
        #img1 = (img1 - img1.mean() * window) * window
        #img2 = (img2 - img2.mean() * window) * window
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

def translate(img, shift, cval=0):
    sx, sy = shift
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
    s_sin = drift_speed*(np.sin(angle)) # used to use make_zero_zero on this, but it breaks it?
    s_cos = drift_speed*(np.cos(angle))
    arr[..., 0,0] = 1 - s_cos
    arr[..., 0,1] = -s_cos*xlen
    arr[..., 0,2] = -s_cos    
    arr[..., 1,0] = -s_sin
    arr[..., 1,1] = 1 - s_sin*xlen
    arr[..., 1,2] = -s_sin
    return arr.squeeze()

def warp_images(images, scan_angles, drift_speed=0, drift_angle=0):
    warped_images = [warp_image(img, scan, drift_speed, drift_angle) for img, scan in zip(images, scan_angles)]
    return warped_images

def warp_and_shift_images(images, scan_angles, drift_speed=0, drift_angle=0):
    warped_images = [warp_image(img, scan, drift_speed, drift_angle) for img, scan in zip(images, scan_angles)]
    translated_images = [warped_images[0]]

    shifts = [hybrid_correlation(warped_images[0], img)[0] for img in warped_images[1:]]
    translated_images += [translate(img, shift[::-1]) for img, shift in zip(warped_images[1:], shifts)]
    translated_images = cp.array(translated_images)
    return translated_images

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

def warp_image(img, scanrotation, drift_speed, drift_angle, nan=False, post_shifts=[0,0]):
    drift_transform = transform_drift_scan(
         -scanrotation, 
         drift_angle, 
         drift_speed, 
         img.shape[-2])
    T = add_shifts_and_rotation_to_transform(drift_transform, img.shape, scanrotation, post_shifts)
    cval = cp.nan if nan else 0 # value of pixels that were outside the image border
    return affine_transform(
        img, 
        cp.linalg.inv(cp.asarray(swap_transform_standard(T.get_matrix()))),
        order=1,
        cval=cval)

def warp_and_correlate(images, scan_angles, drift_angles, speeds, normalize_correlation=False, correlation_power=0.8):
    all_maxes = []
    pairs = []

    for ai, drift_angle in tqdm(
        enumerate(drift_angles), 
        desc="Iterating through drift angles", 
        total=len(drift_angles),
        leave=True):
        
        angle_maxes = []
        for si, speed in tqdm(
            enumerate(speeds), 
            desc="Iterating through drift speeds", 
            total=len(speeds),
            leave=False):
            current_maxes = []
            current_shifts = []
            warped_imgA = warp_image(images[0], scan_angles[0], speed, drift_angle)
            
            speed_maxes = []
            for img, scanangle in zip(images[1:], scan_angles[1:]):
                warped_imgB = warp_image(img, scanangle, speed, drift_angle)
                m = hybrid_correlation(warped_imgA, warped_imgB, fit_only=True, normalize=normalize_correlation, p=correlation_power)
                speed_maxes.append(m.item())
            angle_maxes.append(speed_maxes)
            pairs.append((drift_angle, speed))
        all_maxes.append(angle_maxes)
    return np.array(all_maxes), np.array(pairs)

def get_best_angle_speed_shifts(images, scan_angles, drift_angles, drift_speeds, normalize_correlation=False, correlation_power=0.8, image_number=0):
    
    all_maxes, pairs = warp_and_correlate(images, scan_angles, drift_angles, drift_speeds, normalize_correlation=normalize_correlation, correlation_power=correlation_power)
    angle_fit = all_maxes.mean(axis=(1,2))
    str_fit = all_maxes.mean(axis=(0,2))

    s, a = np.meshgrid(drift_speeds*images[0].shape[0], np.deg2rad(drift_angles))
    
    m = all_maxes.mean(-1)

    maxes = np.mean(all_maxes.reshape((-1, len(images)-1)), 1)
    i = maxes.argmax() # take the mean over each set of images
    best_drift_angle, best_drift_speed = pairs[i]


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
    indices = (indices * (shape[0], 1)).sum(-1)
    weights = np.array([w1, w2, w3, w4]).T

    weight_bins = np.bincount(indices.flatten(), weights=weights.flatten())
    intens_bins = np.bincount(indices.flatten(), weights=(intensities[:, None]*weights).flatten())

    all_weight_bins = np.zeros(np.prod(shape))
    all_intens_bins = np.zeros(np.prod(shape))

    all_weight_bins[:len(weight_bins)] = weight_bins
    all_intens_bins[:len(weight_bins)] = intens_bins

    weight_image = all_weight_bins.reshape(shape)
    intens_image = all_intens_bins.reshape(shape)
    return intens_image, weight_image

def bilinear_bincount_cupy(points, intensities):
    """Bilinear weighting of points onto a grid.
    Extent of grid given by min and max of points in each dimension
    points should be a cupy array of shape (N, 2)
    intensity should be a cupy array of shape (N,)
    """
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
    indices = (indices * cp.array([shape[0].item(), 1])).sum(-1)
    weights = cp.array([w1, w2, w3, w4]).T

    # These bins only fill up to the highest index - not to shape[0]*shape[1]
    weight_bins = cp.bincount(indices.flatten(), weights=weights.flatten())
    intens_bins = cp.bincount(indices.flatten(), weights=(intensities[:, None]*weights).flatten())
    
    # So we create a zeros array that is big enough
    all_weight_bins = cp.zeros(cp.prod(shape).item())
    all_intens_bins = cp.zeros_like(all_weight_bins)
    # And fill it here
    all_weight_bins[:len(weight_bins)] = weight_bins
    all_intens_bins[:len(weight_bins)] = intens_bins

    weight_image = all_weight_bins.reshape(shape.get())
    intens_image = all_intens_bins.reshape(shape.get())
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
    return np.nanmean(np.abs(original - reference))

def make_final_image(images, transforms, corrected_indices, subpixel_factor=1):
    "Should add option to specify which positions one wants to interpolate"
    coords = []
    for i in range(len(images)):
        coords.append(transform_points(corrected_indices[i], swap_transform_standard(transforms[i])))
    coords = np.array(coords)
    tesselation = Delaunay(np.swapaxes(coords, 0, 1).reshape((2,-1)).T)
    func = LinearNDInterpolator(tesselation, asnumpy(images.flatten()), fill_value=np.nan)
    indices = np.mgrid[
        :images.shape[1]-1/subpixel_factor:subpixel_factor*images.shape[1]*1j, 
        :images.shape[2]-1/subpixel_factor:subpixel_factor*images.shape[2]*1j
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


def get_interpolated_functions_and_transforms(images, scanangles, best_angle, best_speed, post_shifts, image_indices=None):
    other_indices = get_indices_of_non_parallel_images(scanangles)
    indices = np.indices(images.shape[1:])
    if image_indices is None:
        image_indices = np.stack(len(images)*[indices]).astype(float)

    forward_transformed_coords = []
    intensities = []
    funcs = []
    transforms = []

    for i, image in enumerate(tqdm(images, desc='Creating transforms')):
        T = transform_drift_scan(-scanangles[i], best_angle, best_speed, images.shape[-2])
        T = add_shifts_and_rotation_to_transform(
            T, image.shape, scan_rotation_deg=scanangles[i], post_shifts=post_shifts[i]).get_matrix()

        forward_coords = transform_points(image_indices[i], T)
        transforms.append(T)

        forward_transformed_coords.append(forward_coords)
        intensities.append(asnumpy(image))

    forward_transformed_coords = np.array(forward_transformed_coords)
    intensities = np.array(intensities)

    for i in tqdm(range(len(images)), desc='Creating interpolation functions'):
        coords = np.zeros((len(other_indices[i]),) + (2,) + images.shape[1:])
        intens = np.zeros((len(other_indices[i]),) + images.shape[1:] )
        for index, j in enumerate(other_indices[i]):
            coords[index] = forward_transformed_coords[j]
            intens[index] = intensities[j]
        tesselation = Delaunay(np.swapaxes(coords, 0, 1).reshape((2,-1)).T)
        func = LinearNDInterpolator(tesselation, intens.flatten(), fill_value=np.nan)
        funcs.append(func)
    return funcs, transforms