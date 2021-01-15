import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from .utils import (
    cp, asnumpy, affine_transform, make_zero_zero, swap_transform_standard, tukey_window,
    normalize_one, normalize_many, estimate_center, Gaussian2DFunc)
from tqdm.auto import tqdm
from time import time


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
    shifts = []
    for warped_imgB in warped_images_nonan[1:]:
        shift, m = hybrid_correlation(warped_imgA, warped_imgB, normalize=normalize_correlation)
        shifts.append(shift)
    shifts = cp.array(np.array(shifts))


    corrected_images = warped_images_with_nan.copy()
    corrected_images[1:] =  cp.stack([translate(img, shift[::-1], cval=cp.nan) for img, shift in zip(warped_images_with_nan[1:], shifts)])
    
    if subpixel:
        subpixel_images = []
        img1 = warped_images_nonan[0].copy()
        for img_nonan, img_nan, rough_shift in zip(warped_images_nonan[1:], warped_images_with_nan[1:], shifts):
            shift = cp.asarray(subpixel_correlation(img1, img_nonan))
            img2b = translate(img_nan, shift + rough_shift[::-1], cval=cp.nan)
            subpixel_images.append(img2b)
        warped_images_with_nan[1:] = cp.stack(subpixel_images)
        corrected_images = warped_images_with_nan
    mean_image = cp.nanmean(corrected_images, axis=0)

    return asnumpy(mean_image)
    
def estimate_drift(images, scan_angles, tolerancy_percent=1, normalize_correlation=True, debug=False, correlation_power=0.8):
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

def warp_and_shift_images(images, scan_angles, drift_speed=0, drift_angle=0):
    warped_images = [warp_image(img, scan, drift_speed, drift_angle) for img, scan in zip(images, scan_angles)]
    translated_images = [warped_images[0]]

    shifts = [hybrid_correlation(warped_images[0], img)[0] for img in warped_images[1:]]
    translated_images += [translate(img, shift[::-1]) for img, shift in zip(warped_images[1:], shifts)]
    translated_images = cp.array(translated_images)
    return translated_images

def add_shifts_and_rotation_to_transform(transform, image_shape, scan_rotation_deg):
    shift1, shift2 = (np.array(image_shape) - 1) / 2
    T = (
        Affine2D().translate(-shift1, -shift2)
        + Affine2D(transform)
        .rotate_deg(scan_rotation_deg)
        .translate(shift1, shift2)
    )
    return T

def warp_image(img, scanrotation, drift_speed, drift_angle, nan=False):
    drift_transform = transform_drift_scan(
         -scanrotation, 
         drift_angle, 
         drift_speed, 
         img.shape[-2])
    T = add_shifts_and_rotation_to_transform(drift_transform, img.shape, scanrotation)
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
    ax3.annotate("", xy=(x+dx, y+dy), xytext=(x, y), arrowprops=dict(color="k"))
    ax3.set_yticklabels([])
    ax3.set_ylim(s.min(), s.max())
    ax1.set_xlabel('Angles (°)')
    ax2.set_xlabel('Drift Speed (A.U.)')
    ax1.set_ylabel('Fit (Average over drift speeds)')
    ax2.set_ylabel('Fit (Average over drift angles)')
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
