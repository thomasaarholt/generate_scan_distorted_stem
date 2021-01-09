import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from matplotlib.transforms import Affine2D
from matplotlib.patches import Arrow
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from time import time
from IPython import display
import gc
from scipy.signal.windows import tukey
from scipy.optimize import curve_fit


try:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform, zoom
    

except:
    print('No CuPy')
    cp = np
    from scipy.ndimage import affine_transform, zoom


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
    
def subpixel_correlation_shift(img1, img2, rough_shift=(0,0), subpixel_radius=2.5, steps=11, debug=False):
    "Returns the value of shift needed to be fed to shift img2 onto img1 using Model.translate"
    rough1, rough2 = rough_shift
    crop1 = abs(rough1) + subpixel_radius
    crop2 = abs(rough1) + subpixel_radius
    
    img1_crop = img1[crop1:-crop1, crop2:-crop2]
    img2_crop = img2[crop1:-crop1, crop2:-crop2]
    
    shift1 = np.linspace(-subpixel_radius, subpixel_radius, steps)
    shift2 = np.linspace(-subpixel_radius, subpixel_radius, steps)

    difference_img = cp.zeros((shift1.shape + shift2.shape))
    for i1, d1 in enumerate(shift1):
        for i2, d2 in enumerate(shift2):
            img2_crop_b = translate(img2_crop, (rough1 + d1, rough2 + d2), cval=cp.nan)
            mask = ~np.isnan(img2_crop_b) # only non-nan indices
            diff = np.abs(img1_crop - img2_crop_b)[mask]
            difference_img[i1, i2] = np.mean(diff)
            
    difference_img = difference_img.get()

    (p1, p2), parameters = estimate_center(difference_img, shift1, shift2, subpixel_radius)
    print(p1, p2)
    i = np.argmin(difference_img)
    i1, i2 = np.unravel_index(i, difference_img.shape)
    S1, S2 = np.meshgrid(shift1, shift2)
    p1, p2 = S1[i1, i2], S2[i1, i2]
    print(p1, p2) 
    plt.figure()
    plt.imshow(difference_img, extent=(-subpixel_radius, subpixel_radius, -subpixel_radius, subpixel_radius))

    shift = (rough1 + p1, rough2 + p2)
    if debug:
        plt.figure()
        plt.imshow(difference_img, extent=(-subpixel_radius, subpixel_radius, subpixel_radius, -subpixel_radius))
        plt.contour(S1, S2, Gaussian2D((S1, S2), *parameters), 8, colors='w')
        print(f"Subpixel offsets: p1={round(p1,2)}, p2={round(p2, 2)}")
    return shift

def subpixel_correlation(img1, img2, subpixel_radius=2.5, steps=11):
    "Estimates pixel shift using hybrid correlation, then manually compares the differences by subpixel translation of the image"
    rough_shift, m = hybrid_correlation(img1, img2)
    shift = subpixel_correlation_shift(img1, img2, rough_shift=rough_shift, subpixel_radius = subpixel_radius, steps=steps)
    return shift

def make_zero_zero(arr, eps=1e-13):
    "cos(np.deg2rad(90)) will return a nonzero value. This sets such small values to zero."
    if isinstance(arr, (np.ndarray, cp.ndarray)):
        arr[np.abs(arr) < eps] = 0
    elif isinstance(arr, float):
        arr = 0
    return arr

def get_atoms():
    from ase.io import read
    from ase.build import make_supercell
    atoms = read("0013687130_v6bxv2_tv0.1bxv0.0_d1.8z_traj.xyz")
    atoms = make_supercell(atoms, np.diag((1,2,1)))
    mask = atoms.positions[:,2] > 35
    del atoms[mask]
    atoms.rotate([1,0,0], (0,0,1))

    atoms[1716].number = 80
    atoms[1766].number = 80
    return atoms

def get(nImages=4, drift_strength=5, pixel_size = 0.1, maxScanAngle=360, minScanAngle=0, drift_angle=None, vacuum=10., jitter=0.,  ):
    atoms = get_atoms()
    random_angle = drift_angle if drift_angle else np.random.random() * 2*np.pi
    drift_vector = [np.cos(random_angle),np.sin(random_angle)]
    centre_drift = True
    
    images = []

    scanangles = cp.linspace(minScanAngle, maxScanAngle, nImages, endpoint=False)
    for scanangle in tqdm(scanangles):
        m = ImageModel(atoms, scan_rotation=scanangle,
                    pixel_size=pixel_size, vacuum=vacuum,
                        drift_strength=drift_strength, 
                        drift_vector=drift_vector,
                        jitter_strength=jitter, 
                        centre_drift=centre_drift,
                        fast=False,
                    )
        img = m.generate_cupy()
        side = np.minimum(*img.shape)
        images.append(img[:side, :side])
    images = cp.stack(images)
    print(f"Size: {images.nbytes / 1e9} GB")
    print(f"Shape: {images.shape}")
    return images, scanangles, random_angle

def create_final_stack_and_average(images, scanangles, best_str, best_angle, normalize_correlation = False, print_shifts = False):
    """From an estimated drift strength and angle, produce a stack of transformed 
    images and the stack average.
    """
    warped_images_with_nan = [
    warp_image(
        img, 
        scanangle, 
        best_str, 
        best_angle,
        nan=True)
    for img, scanangle in zip(images, scanangles)]

    warped_images_nonan = cp.stack(warped_images_with_nan)
    warped_images_nonan[cp.isnan(warped_images_nonan)] = 0

    warped_imgA = warped_images_nonan[0]
    shifts = []
    #mean_image = warped_imgA / len(images)
    for warped_imgB in warped_images_nonan[1:]:
        shift, m = hybrid_correlation(warped_imgA, warped_imgB, normalize=normalize_correlation)
        shifts.append(shift)
        #mean_image += translate(warped_imgB, shift)/len(images)

    warped_images2 = cp.stack(warped_images_with_nan[:1] + [translate(img, shift[::-1]) for img, shift in zip(warped_images_with_nan[1:], shifts)])
    mean_image = cp.nanmean(warped_images2, axis=0)
    #mean_image[cp.isnan(mean_image)] = 0
    
    return mean_image.get()#, warped_images_nonan.get()

def estimate_drift(images, scanangles, tolerancy_percent=1, normalize_correlation=False, debug=False, correlation_power=0.8):
    """Estimates the global, constant drift on an image using affine transformations.
    Takes as input a list of images and scanangles. Only tested with square images.

    Explores the space of possible transformations in 360 degree directions and drift 
    speeds that don't result in shears or stretches out of the image.

    Begins by testing drift at 0, 90, 180 and 270 degree intervals, and converges
    towards the best fit.
    """
    t1 = time()
    angle_steps = 8
    angle_low, angle_high = 0, 360 - 360/angle_steps

    xlen = images.shape[1]
    str_steps = 9
    str_low, str_high = 0, 1 / xlen

    print("Getting fit with no drift first")
    all_maxes, pairs = warp_and_correlate(images, scanangles, cp.array([0]), cp.array([0]), normalize_correlation=normalize_correlation, correlation_power=correlation_power)
    maxes = all_maxes.reshape((-1, len(images)-1)).mean(1)

    best_angle, best_str, fit_value = 0, 0, maxes.max()
    history = [(best_angle, best_str, fit_value)]

    old_values = np.array((best_angle,best_str))
    new_values = np.array([np.inf, np.inf])
    i = 0
    j = 0 # "extra" iterations when something just needed readjusting
    converged = False
    converged_for_two_iterations = False
    
    try:
        while i < 2 or not converged_for_two_iterations:
            print()
            print("Iteration #{}: Best guess: Angle: {}°, Strength: {:.2e}. Fit value: {:.2e}".format(
                i+j, np.round(best_angle, 2), np.round(best_str, 5), fit_value)
                + "\nLimits: [{:.1f}°, {:.1f}°], [{:.2e}, {:.2e}]".format(
                    angle_low % 360, angle_high % 360, str_low, str_high))
            old_values = np.array([best_angle, best_str])

            if angle_low > angle_high:
                print("ANGLE LOW WAS HIGHER")
                angle_low -= 360
            drift_angles = cp.linspace(angle_low, angle_high, angle_steps)
            #angle_low %= 360
            #angle_high %= 360

            angle_diff = (drift_angles[1] - drift_angles[0]).item()

            drift_strengths = cp.linspace(str_low, str_high, str_steps)
            str_diff = (drift_strengths[1] - drift_strengths[0]).item()

            best_angle, best_str, fit_value = get_best_angle_strength_shifts(
                images,
                scanangles,
                drift_angles, 
                drift_strengths,
                normalize_correlation=normalize_correlation,
                correlation_power=correlation_power,
                image_number=i+j,
            )
            history.append((best_angle, best_str, fit_value))
            new_values = np.array([best_angle, best_str])
            if debug:
                warped_images_nonan = [
                warp_image(
                    img, 
                    scanangle, 
                    best_str, 
                    best_angle,
                    nan=False) for img, scanangle in zip(images, scanangles)]
                fig = plt.figure(figsize=(6,3))
                fig.clear()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(warped_images_nonan[0].get())
                ax2.imshow(warped_images_nonan[1].get())
                fig.tight_layout()
                fig.canvas.draw()
                
            if best_str in drift_strengths[:2]:
                print("Initial drift speed intervals were too large. Reducing.")
                str_low = 0
                str_high = drift_strengths[2].item()
                j += 1
                continue


            if best_str in drift_strengths[-2:]:
                print("Initial drift speed intervals were too small. Increasing.")
                str_low = drift_strengths[-2].item()
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
                print("Adjusting drift strength limits")
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
    print("Final iteration: Final guess: Angle: {}°, Strength: {:.2e}".format(
            np.round(best_angle, 2).item(), best_str.item()))
    print(f"\nTook {round(time() - t1, 1)} seconds")
        
    return best_angle, best_str, np.array(history)

def swap_transform_standard(T):
    """Swaps the transformation matrix order from the regular one
    expected by skimage.transform.warp to the YXZ one expected by 
    scipy.ndimage.affine_transform
    """
    T = T.copy()
    T[:2,:2] = T[:2,:2][::-1,::-1]
    T[:2, 2] = T[:2, 2][::-1]
    return T

def normalize_many(imgs, window):
    "Normalize images for hybrid correlation"
    axes = (-2,-1)
    return window * (
        imgs - np.expand_dims(
            (imgs * window).mean(axis=axes), axes) / window.mean()
    )

def normalize_one(imgs, window):
    "Normalize image for hybrid correlation"
    return window * (
        imgs - (imgs * window).mean() / window.mean()
    )

def tukey_window(shape, alpha=0.1):
    filt1 = tukey(shape[0], alpha=0.1, sym=True)
    if shape[0] == shape[1]:
        filt2 = filt1
    else:
        filt2 = tukey(shape[1], alpha=0.1, sym=True)
    return filt1[:, None] * filt2

def hybrid_correlation(
    img1, 
    img2,
    p=0.8,
    normalize = True, 
    window=True,
    fit_only = False,
    ):
    """Performs hybrid correlation on two images.
    for higher performance, allows the option to already
    have performed the fft on the inputs.
    Seems to work fine with the real-input `rfft`.

    fit_only will only return the correlation maximum value, 
    not the required shift.
    """

    if window:
        window = cp.asarray(tukey_window(img1.shape))
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
    corr =  (cp.fft.irfft2(cp.abs(m)**p * cp.exp(1j * cp.angle(m))))

    if fit_only:
        return corr.max()

    corr = cp.fft.fftshift(corr)
    translation = cp.array(cp.unravel_index(corr.argmax(), corr.shape))
    center = (cp.array(corr.shape))// 2
    return np.stack([x.item() for x in translation - center]), corr.max()

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
    
def drift_points(shape=(10,10), drift_deg = 0, drift_strength=0):
    lenX, lenY = shape
    drift_vector = (rotation_matrix(drift_deg) @ [1,0]) * drift_strength
    arr = np.zeros((lenX, lenY, 2))
    drift = np.zeros(2)
    for yi in range(lenY):
        for xi in range(lenX):
            drift += drift_vector
            position = np.array((xi, yi))
            arr[xi, yi] = position - drift
    return arr

def drift_pointsYX(shape=(10,10), drift_deg = 0, drift_strength=0):
    "Rotate drift from along x axis, anticlockwise in degrees"
    lenY, lenX = shape
    drift_vector = (rotation_matrix(drift_deg) @ [1,0]) * drift_strength
    drift = np.zeros(2)
    positions = []
    for yi in range(lenY):
        for xi in range(lenX):
            drift += drift_vector
            position = np.array((xi, yi))
            #arr[yi, xi] = position - drift
            positions.append(position - drift)
    return np.array(positions)

def plot(points, ax, lim=((),())):
    points = points.reshape((-1, 2), order='F')
    for i, xy in enumerate(points):
        rect = plt.Rectangle(xy-0.25, 0.5, 0.5, color=plt.cm.RdYlBu(i))
        ax.add_patch(rect)
    xmin, ymin = points.min(0) - 2
    xmax, ymax = points.max(0) + 3
    if lim == ((),()):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
    else:
        ax.set_xlim(lim[0])
        ax.set_ylim(lim[1])  
        
def extend_3D_ones(arr_of_2d):
    return np.hstack([arr_of_2d, np.ones((len(arr_of_2d),1))])
    
def get_matrix(xy, xyprime):
    xy = extend_3D_ones(xy)
    xyprime = extend_3D_ones(xyprime)
    T, *_ = np.linalg.lstsq(xy, xyprime, rcond=None)
    return T.T

def transform_points(points, transform):
    points = extend_3D_ones(points)
    points_prime = points @ transform
    return points_prime[:, :2]

def arr2fft(img):
    import hyperspy.api as hs
    img = img.copy()
    img[np.isnan(img)] = 0
    s = hs.signals.Signal2D(img)
    s = np.log(s.fft(True, True).amplitude)
    return s

def get_and_plot_peaks(data, average_distance_between_peaks=80, threshold = 1):
    neighborhood_size = average_distance_between_peaks
    
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)

    plt.figure()
    plt.imshow(data)
    plt.plot(x,y, 'ro')
    for i, (xi, yi) in enumerate(zip(x,y)):
        pass#plt.annotate(f"{i}", (xi, yi))
    return x, y

def transform_drift_scan(scan_rotation_deg=0, drift_angle_deg=0, strength=0, xlen=100):
    arr = np.zeros(np.shape(strength) + np.shape(drift_angle_deg) + (3,3))
    if np.shape(strength):
        strength = strength[:, None, None]
    angle = np.deg2rad(drift_angle_deg + scan_rotation_deg)
    arr[...] = np.eye(3)
    s_sin = strength*make_zero_zero(np.sin(angle))
    s_cos = strength*make_zero_zero(np.cos(angle))
    arr[..., 0,0] = 1 - s_cos
    arr[..., 0,1] = -s_cos*xlen
    arr[..., 0,2] = -s_cos    
    arr[..., 1,0] = -s_sin
    arr[..., 1,1] = 1 - s_sin*xlen
    arr[..., 1,2] = -s_sin
    return arr.squeeze()

def warp_and_shift_images(images, scanangles, drift_strength=0, drift_angle=0):
    warped_images = [warp_image(img, scan, drift_strength, drift_angle) for img, scan in zip(images, scanangles)]
    translated_images = [warped_images[0]]

    shifts = [hybrid_correlation(warped_images[0], img)[0] for img in warped_images[1:]]
    translated_images += [translate(img, shift) for img, shift in zip(warped_images[1:], shifts)]
    translated_images = cp.array(translated_images)
    #translated_images[translated_images == 0] = cp.nan
    return translated_images

def warp_image(img, scanrotation_deg, drift_strength, drift_angle_deg, nan=False):
    shift_x, shift_y = (cp.array(img.shape) - 1) / 2
    main_transform = transform_drift_scan(
         -scanrotation_deg.item(), 
         drift_angle_deg, 
         drift_strength, 
         img.shape[-2])
    T = (
    Affine2D().translate(-shift_x, -shift_y)
    + Affine2D(main_transform).rotate_deg(scanrotation_deg).translate(shift_x, shift_y)
    )
    cval = cp.nan if nan else 0 # value of pixels that were outside the image border
    return affine_transform(
        img, 
        cp.linalg.inv(cp.asarray(swap_transform_standard(T.get_matrix()))),
        order=1,
        cval=cval)

def warp_and_correlate(images, scanangles, drift_angles, strengths, normalize_correlation=False, correlation_power=0.8):
    all_maxes = []
    pairs = []

    for ai, drift_angle in tqdm(
        enumerate(drift_angles), 
        desc="Iterating through drift angles", 
        total=len(drift_angles),
        leave=True):
        
        angle_maxes = []
        for si, strength in tqdm(
            enumerate(strengths), 
            desc="Iterating through drift strengths", 
            total=len(strengths),
            leave=False):
            current_maxes = []
            current_shifts = []
            warped_imgA = warp_image(images[0], scanangles[0], strength, drift_angle)
            
            strength_maxes = []
            for img, scanangle in zip(images[1:], scanangles[1:]):
                warped_imgB = warp_image(img, scanangle, strength, drift_angle)
                m = hybrid_correlation(warped_imgA, warped_imgB, fit_only=True, normalize=normalize_correlation, p=correlation_power)
                strength_maxes.append(m.get())
            angle_maxes.append(strength_maxes)
            pairs.append((drift_angle.item(), strength.item()))
        all_maxes.append(angle_maxes)
    return np.array(all_maxes), np.array(pairs)

def get_best_angle_strength_shifts(images, scanangles, drift_angles, drift_strengths, normalize_correlation=False, correlation_power=0.8, image_number=0):
    
    all_maxes, pairs = warp_and_correlate(images, scanangles, drift_angles, drift_strengths, normalize_correlation=normalize_correlation, correlation_power=correlation_power)
    angle_fit = all_maxes.mean(axis=(1,2))
    str_fit = all_maxes.mean(axis=(0,2))

    #s, a = np.meshgrid(drift_strengths.get(), np.deg2rad(np.append(drift_angles.get(), 360)))
    s, a = cp.meshgrid(drift_strengths*images[0].shape[0], np.deg2rad(drift_angles))
    s, a = s.get(), a.get()
    
    m = all_maxes.mean(-1)

    maxes = all_maxes.reshape((-1, len(images)-1)).mean(1)
    i = maxes.argmax() # take the mean over each set of images
    best_drift_angle, best_drift_strength = pairs[i]


    #m = np.append(m, m[0,None], axis=0)
    #fig = plt.figure(figsize=(9,3))
    fig = plt.gcf()
    fig.clear()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection="polar")
    
    ax1.scatter(drift_angles.get(), angle_fit)
    ax2.plot(drift_strengths.get()*images.shape[1], str_fit)
    ax3.pcolormesh(a,s,m, shading="nearest", cmap='jet')

    #ax3.scatter(np.deg2rad(best_drift_angle), best_drift_strength*images.shape[1], c='k', marker="x")
    x, y = np.deg2rad(best_drift_angle), drift_strengths[0].item() * images.shape[1]
    dx, dy = 0, best_drift_strength* images.shape[1] - y
    #print(y, dy, y + dy)
    #plt.arrow(x,y, dx, dy, color='black', lw = 3, head_width=0.3, head_length=0.05)
    ax3.annotate("", xy=(x+dx, y+dy), xytext=(x, y), arrowprops=dict(color="k"))
    #ax3.add_patch(arrow)
    ax3.set_yticklabels([])
    ax3.set_ylim(s.min(), s.max())
    ax1.set_xlabel('Angles (°)')
    ax2.set_xlabel('Drift Speed (A.U.)')
    ax1.set_ylabel('Fit (Average over drift speeds)')
    ax2.set_ylabel('Fit (Average over drift angles)')
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig('thomas_{:02d}.png'.format(image_number))
    return best_drift_angle, best_drift_strength, maxes[i]

def drift_values_converged(old_values, new_values, tolerancy_percent=1):
    tol = tolerancy_percent/100
    return (
        (new_values >= old_values - old_values*tol) 
        & 
        (new_values <= old_values + old_values*tol)
    ).all()

def Gaussian2D(x, y, A, xc, yc, sigma):
    return A*np.exp(
        -(
            (x-xc)**2 + 
            (y-yc)**2
        )/(2*sigma**2))

def Gaussian2DCupy(x, y, A, xc, yc, sigma):
    return A*cp.exp(
        -(
            (x-xc)**2 + 
            (y-yc)**2
        )/(2*sigma**2))

def SympyGaussian2D(x, y, A, xc, yc, sigma):
    return A*sp.exp(
        -(
            (x-xc)**2 + 
            (y-yc)**2
        )/(2*sigma**2))

def rotation_matrix(deg, clockwise_positive=False):
    c = np.cos(np.deg2rad(deg))
    s = np.sin(np.deg2rad(deg))
    arr = np.array([[c, -s],[s, c]])
    if clockwise_positive:
         arr = arr.T
    return arr

def rotation_matrix_cupy(deg):
    c = cp.cos(cp.deg2rad(deg))
    s = cp.sin(cp.deg2rad(deg))
    return cp.array([cp.array([c, -s]), cp.array([s, c])])

def add_ac_noise(shape, strength=0.5, dwelltime=1e-6, ac_freq=50):
    noise = np.zeros((2,) + shape)
    noise[0] = strength * np.sin(
        2*dwelltime / (1/ac_freq) * np.arange(np.prod(shape)) * np.pi
    ).reshape(shape)
    return noise
    
def add_drift(shape, drift_vector = [1,0], strength=1e-4):
    strength /= np.prod(shape)
    vector = -cp.array(drift_vector)
    probe_indices = cp.arange(cp.prod(cp.array(shape))).reshape(shape)
    return (strength * vector * probe_indices.T[..., None]).T

def add_line_jitter(XYshape, strength = 0.3, horizontal=True, vertical=False, ):
    jitter = cp.zeros(XYshape) # Shape is (2, X, Y)
    if type(strength) == tuple:
        strengthx = strength[0]
        strengthy = strength[1]
    else:
        strengthx = strengthy = strength
    if horizontal:
        jitter[0] += strengthx*(2*cp.random.random((XYshape[1])) - 1)[:, None]
    if vertical:
        jitter[1] += strengthy*(2*cp.random.random((XYshape[1])) - 1)[:, None]
    return jitter

class ImageModel:
    """Create a STEM-HAADF like image from a list of positions and atomic numbers, or from
    an ASE atoms object. Can add features like:

    scan_rotation: deg
    jitter_strength: float, shifts each scanline by a random factor
    jitter_horizontal: shift scanline leftright by above
    jitter_vertical: shift scnaline updown by above
    pixel_size: Å, Affects to image resolution
    sigma: standard deviation of 2D gaussian representing atomic columns
    power: HAADF n-factor 
    drift_strength: Automatically divided by image shape - should be 0-10
    drift_vector: Direction of drift
    centre_drift: Shift image borders so drifted image is centered
    square: Make image square
    vacuum: Add whitespace around image
    fast: Only compute one layer of unique atoms
    """

    def __init__(
        self, 
        atoms=None, positions=None, numbers=None,
        scan_rotation = 0, jitter_strength=0,
        jitter_horizontal=True, jitter_vertical=False,
        pixel_size=0.1, sigma=0.4, power=1.8, 
        drift_strength = 0, drift_vector=[1,0], centre_drift=True,
        square = False, vacuum=5.0, fast=False):
        
        if atoms:
            self.atom_positions = atoms.positions[:,:2]
            self.atom_numbers = atoms.numbers
        else:
            if not positions:
                raise AttributeError(
            "You must supply either an ase Atoms object or a list of positions and atomic numbers"
            )
            self.atom_positions = positions
            self.atom_numbers = numbers

        if fast:
            unique = np.unique(np.column_stack([self.atom_positions, self.atom_numbers]), axis=0)

            self.atom_positions = unique[:, :2]
            self.atom_numbers =  unique[:, 2]
                

        self.number_of_atoms = len(self.atom_numbers)
        self.pixel_size = pixel_size
        self.sigma = sigma
        self.power = power

        self.jitter_strength = jitter_strength
        self.jitter_horizontal = jitter_horizontal
        self.jitter_vertical = jitter_vertical

        self.drift_strength = drift_strength
        self.drift_vector = drift_vector
        self.centre_drift = centre_drift
        self.scan_rotation = scan_rotation
        self.square = square
        self.margin = vacuum
        
        #self.create_probe_positions()
        self.create_probe_positions_cupy()
        self.create_parameters()
        self.create_parameters_cupy()
        
    def init_sympy(self):
        xy = sp.symbols('x y')
        parameters = sp.symbols('A xc yc sigma', cls=sp.IndexedBase)
        i,n = sp.symbols("i n", integer=True)
        self.symbols = xy + parameters + (n,)
        A, xc, yc, sigma = parameters
        
        Gauss = SympyGaussian2D(*xy, A[i], xc[i], yc[i], sigma[i])
        model = sp.Sum(Gauss, (i,0,n-1))
        self.model = model

    def create_probe_positions(self):
        xlow, ylow = self.atom_positions.min(0) - self.margin
        xhigh, yhigh = self.atom_positions.max(0) + self.margin
        scale = (xhigh - xlow)/100

        if self.square:
            xlow = ylow = min(xlow, ylow)
            xhigh = yhigh = max(xhigh, yhigh)
        xrange = np.arange(xlow, xhigh+scale, self.pixel_size)
        yrange = np.arange(ylow, yhigh+scale, self.pixel_size)
        XY = np.stack(np.meshgrid(xrange, yrange))
        
        if self.scan_rotation:
            mean = XY.mean(axis=(-1,-2))[:, None] - 0.5
            self.probe_positions = (
                rotation_matrix(self.scan_rotation) @ (
                    XY.reshape((2, -1)) - mean) + mean
            ).reshape((2, *XY.shape[1:]))
        else:
            self.probe_positions = XY
    
    def create_probe_positions_cupy(self):
        xlow, ylow = self.atom_positions.min(0) - self.margin
        xhigh, yhigh = self.atom_positions.max(0) + self.margin
        scale = (xhigh - xlow)/100

        if self.square:
            xlow = ylow = min(xlow, ylow)
            xhigh = yhigh = max(xhigh, yhigh)
        xrange = cp.arange(xlow, xhigh+scale, self.pixel_size)
        yrange = cp.arange(ylow, yhigh+scale, self.pixel_size)
        self.probe_positions_cupy = cp.stack(cp.meshgrid(xrange, yrange))
        XYshape = self.probe_positions_cupy.shape
        
        if self.jitter_strength:
            self.probe_positions_cupy += add_line_jitter(
                XYshape = XYshape, 
                strength=self.jitter_strength, 
                horizontal=self.jitter_horizontal, 
                vertical=self.jitter_vertical)
            
        if self.scan_rotation:
            mean = self.probe_positions_cupy.mean(axis=(-1,-2))[:, None]
            self.probe_positions_cupy = (
                rotation_matrix_cupy(self.scan_rotation) @ (
                    self.probe_positions_cupy.reshape((2, -1)) - mean) + mean
            ).reshape((2, *XYshape[1:]))

        if self.drift_strength:
            #strength = self.drift_strength / np.prod(XYshape[1:])
            drift = add_drift(XYshape[1:], self.drift_vector, self.drift_strength)
            self.probe_positions_cupy += drift
            
            if self.centre_drift:
                driftx, drifty = drift
                offsetx = driftx.max() if driftx.max() > -driftx.min() else driftx.min()
                offsety = drifty.max() if drifty.max() > -drifty.min() else drifty.min()
                self.probe_positions_cupy -= cp.array([offsetx, offsety])[:, None, None] / 2
        
        
    def create_parameters(self):
        xc, yc = self.atom_positions.T
        A = self.atom_numbers ** self.power
        sigma = np.ones(self.number_of_atoms) * self.sigma
        self.parameters = np.array([A, xc, yc, sigma])
        
    def create_parameters_cupy(self):
        self.parameters_cupy = cp.array(self.parameters)
        
    def generate_lambdify(self):
        self.init_sympy()
        func = sp.lambdify(self.symbols, self.model, modules = 'numpy')
        self.func = func
        return func(*self.probe_positions, *self.parameters, self.number_of_atoms)
    
    def generate_lambdify_cupy(self):
        self.init_sympy()
        func = sp.lambdify(self.symbols, self.model, modules = 'cupy')
        self.func_cupy = func
        return func(*self.probe_positions_cupy, *self.parameters_cupy, self.number_of_atoms)
    
    def generate_cupy(self):
        X, Y = self.probe_positions_cupy
        img = cp.zeros(X.shape)
        #As, XC, YC, SIGMA = self.parameters_cupy
        for parameters in self.parameters_cupy.T:
            img += Gaussian2DCupy(X, Y, *parameters)
        return img
 
    def generate_cupy_ram(self):
        X, Y = self.probe_positions_cupy
        img = cp.sum(Gaussian2DCupy(X[..., None], Y[..., None], *self.parameters_cupy), -1)
        return img
    
    def generate_numpy(self):
        img = np.zeros(self.probe_positions[0].shape)
        for parameters in self.parameters.T:
            gauss = Gaussian2D(*self.probe_positions, *parameters)
            img += gauss
        return img
