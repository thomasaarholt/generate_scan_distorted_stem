import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from matplotlib.transforms import Affine2D
from time import time

try:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform

except:
    print('No CuPy')
    cp = np
    from scipy.ndimage import affine_transform

def create_final_stack_and_average(images, scanangles, best_str, best_angle):
    """From an estimated drift strength and angle, produce a stack of transformed 
    images and the stack average.
    """
    warped_images_with_nan = [
    warp_image_cuda(
        img, 
        scanangle, 
        best_str, 
        best_angle,
        nan=False)
    for img, scanangle in zip(images, scanangles)]

    warped_images_nonan = cp.stack(warped_images_with_nan)
    warped_images_nonan[cp.isnan(warped_images_nonan)] = 0

    warped_imgA = warped_images_nonan[0]
    shifts = []
    for warped_imgB in warped_images_nonan[1:]:
        s, m = hybrid_correlation(warped_imgA, warped_imgB)
        shifts.append((int(s[0].get()), int(s[1].get())))

    warped_images2 = cp.stack(warped_images_with_nan[:1] + [translate(img, shift) for img, shift in zip(warped_images_with_nan[1:], shifts)])
    mean_image = cp.nanmean(warped_images2, axis=0)
    mean_image[cp.isnan(mean_image)] = 0
    
    return mean_image.get(), warped_images_nonan.get()

def estimate_drift(images, scanangles, tolerancy_percent=1):
    """Estimates the global, constant drift on an image using affine transformations.
    Takes as input a list of images and scanangles. Only tested with square images.

    Explores the space of possible transformations in 360 degree directions and drift 
    speeds that don't result in shears or stretches out of the image.

    Begins by testing drift at 0, 90, 180 and 270 degree intervals, and converges
    towards the best fit.
    """
    t1 = time()
    angle_limits_step = (0, 360 - 45, 45)
    str_limits_step = (0, 1/images[0].shape[0], 0.2/images[0].shape[0])

    best_angle, best_str = 0,0
    old_values = np.array((best_angle,best_str))
    new_values = np.array([np.inf, np.inf])
    i = 0
    
    while i < 1 or not drift_values_converged(old_values, new_values, tolerancy_percent=1):
        print("Iteration #{}. Best guess: Angle: {}°, Strength: {:.2e}".format(
            i, np.round(best_angle, 2), np.round(best_str, 5)))
        old_values = np.array([best_angle, best_str])
        best_angle, best_str = get_best_angle_strength_shifts(
            images,
            scanangles,
            angle_limits_step, 
            str_limits_step
        )
        new_values = np.array([best_angle, best_str])
        angle_limits_step = best_angle-angle_limits_step[2]/2, best_angle + angle_limits_step[2]/2, angle_limits_step[2] / 3
        str_limits_step = best_str-str_limits_step[2]/2, best_str + str_limits_step[2]/2, str_limits_step[2] / 3
        i += 1
    print("Iteration #{}. Best guess: Angle: {}°, Strength: {:.2e}".format(
            i, np.round(best_angle, 2), np.round(best_str, 5)))
    print(f"\nTook {round(time() - t1, 1)} seconds")
    return best_angle, best_str

def swap_transform_standard(T):
    """Swaps the transformation matrix order from the regular one
    expected by skimage.transform.warp to the YXZ one expected by 
    scipy.ndimage.affine_transform
    """
    T = T.copy()
    T[:2,:2] = T[:2,:2][::-1,::-1]
    T[:2, 2] = T[:2, 2][::-1]
    return T

def hamming(img):
    "Applies a 2D hamming window to an image"
    hamm = cp.hamming(img1.shape[0])
    window = hamm[:, None] * hamm
    return img*window

def hybrid_correlation(
    img1, 
    img2,
    p=0.9,
    already_fft = False, 
    normalize_mean = False, 
    hamming_filter=True,
    fit_only = False,
    ):
    """Performs hybrid correlation on two images.
    for higher performance, allows the option to already
    have performed the fft on the inputs.
    Seems to work fine with the real-input `rfft`.
    """

    if normalize_mean:
        img1 -= img1.mean()
        img2 -= img2.mean()

    if hamming_filter:
        hamm = cp.hamming(img1.shape[0])
        window = hamm[:, None] * hamm
        img1 = img1 * window
        img2 = img2 * window

    if already_fft:
        fftimg1 = img1
        fftimg2 = img2
    else:
        fftimg1 = cp.fft.rfft2(img1)
        fftimg2 = cp.fft.rfft2(img2)
    m = fftimg1 * cp.conj(fftimg2)
    corr =  cp.real(cp.fft.irfft2(cp.abs(m)**p * cp.exp(1j * cp.angle(m))))
    if fit_only:
        return corr.max()

    corr = cp.fft.fftshift(corr)
    translation = cp.array(cp.unravel_index(corr.argmax(), corr.shape))
    center = cp.array(corr.shape) // 2
    return cp.stack([x for x in translation - center]), corr.max()

def hybrid_correlation_numpy(img1, img2, p=0.9, filter=True):
    if filter:
        hamm = np.hamming(img1.shape[0])
        window = hamm[:, None] * hamm
        img1 = img1 * window
        img2 = img2 * window

    fftimg1 = np.fft.fft2(img1)
    fftimg2 = np.fft.fft2(img2)
    m = fftimg1 * np.conj(fftimg2)
    corr =  np.real(np.fft.ifft2(np.abs(m)**p * np.exp(1j * np.angle(m))))
    corr = np.fft.fftshift(corr)
    translation = np.array(np.unravel_index(corr.argmax(), corr.shape))
    center = np.array(corr.shape) // 2
    return tuple(x for x in translation - center), corr.max()

def translate(img, shift):
    return cp.roll(img, shift, axis=(-2,-1))
    
def drift_points(shape=(10,10), drift_deg = 0, drift_strength=0):
    lenX, lenY = shape
    drift_vector = (rotation_matrix(drift_deg) @ [1,0]) * drift_strength
    arr = np.zeros((lenX, lenY, 2))
    drift = np.zeros(2)
    corners = []
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

def transform_from_angle_strength(angle, strength, angle_diff=np.pi/2):
    arr = np.zeros(np.shape(strength) + np.shape(angle) + (2,) + (3,3))
    if np.shape(strength):
        strength = strength[:, None, None]
    angles = np.column_stack([angle, angle]).astype(float)
    angles[:,1] = angle  + angle_diff

    arr[...] = np.eye(3)
    s_sin = strength*np.sin(angles)
    s_cos = strength*np.cos(angles)
    
    arr[..., 0,0] = 1 + s_cos
    arr[..., 0,1] = 10*s_cos
    arr[..., 0,2] = s_cos
    arr[..., 1,0] = s_sin
    arr[..., 1,1] = 1 + 10*s_sin
    arr[..., 1,2] = s_sin
    
    return arr.squeeze()

def transform_from_angle_strength_asd(angle, strength, angle_diff=np.pi/2):
    arr = np.zeros(np.shape(strength) + np.shape(angle) + (2,) + (3,3))
    if np.shape(strength):
        strength = strength[:, None, None]
    angles = np.column_stack([angle, angle]).astype(float)
    angles[:,1] = angle  + angle_diff

    arr[...] = np.eye(3)
    s_sin = strength*np.sin(angles)
    s_cos = strength*np.cos(angles)
    arr[..., 0,0] = 1 - 10*s_sin
    arr[..., 1,0] = 10*s_cos
    arr[..., 1,1] = 1 + s_cos
    arr[..., 1,2] = s_cos
    arr[..., 0,1] = -s_sin
    arr[..., 0,2] = -s_sin
    return arr.squeeze()

def transform_from_angle_strength3(angle, strength, angle_diff=np.pi/2, x_len=100):
    arr = np.zeros(np.shape(strength) + np.shape(angle) + (2,) + (3,3))
    if np.shape(strength):
        strength = strength[:, None, None]
    angles = np.column_stack([angle, angle]).astype(float)
    angles[:,1] = angle  + angle_diff
    arr[...] = np.eye(3)
    s_sin = strength*np.sin(angles)
    s_cos = strength*np.cos(angles)
    arr[..., 0,0] = 1 - s_cos
    arr[..., 0,1] = -s_cos*x_len
    arr[..., 0,2] = -s_cos    
    arr[..., 1,0] = -s_sin
    arr[..., 1,1] = 1 - s_sin*x_len
    arr[..., 1,2] = -s_sin
    return arr.squeeze()

def transform_from_angle_strength4(angle, strength, angle_diff=np.pi/2, x_len=100):
    arr = np.zeros(np.shape(strength) + np.shape(angle) + (2,) + (3,3))
    if np.shape(strength):
        strength = strength[:, None, None]
    angles = np.column_stack([angle, angle]).astype(float)
    angles[:,1] = angle  + angle_diff
    arr[...] = np.eye(3)
    s_sin = strength*np.sin(angles)
    s_cos = strength*np.cos(angles)
    arr[..., 0,0] = 1 - s_cos
    arr[..., 0,1] = -s_cos*x_len
    arr[..., 0,2] = -s_cos    
    arr[..., 1,0] = -s_sin
    arr[..., 1,1] = 1 - s_sin*x_len
    arr[..., 1,2] = -s_sin
    return arr.squeeze()

def transform_from_angle_strength5(angle, strength, angle_diff=np.pi/2, xlen=100):
    arr = np.zeros(np.shape(strength) + np.shape(angle) + (2,) + (3,3))
    if np.shape(strength):
        strength = strength[:, None, None]
    angles = np.column_stack([angle, angle]).astype(float)
    angles[:,1] = angle  + angle_diff
    arr[...] = np.eye(3)
    s_sin = strength*np.sin(angles)
    s_cos = strength*np.cos(angles)
    arr[..., 0,0] = 1 - s_cos
    arr[..., 0,1] = -s_cos*xlen
    arr[..., 0,2] = -s_cos    
    arr[..., 1,0] = -s_sin
    arr[..., 1,1] = 1 - s_sin*xlen
    arr[..., 1,2] = -s_sin
    return arr.squeeze()

def transform_drift_scan(scan_rotation_deg=0, drift_angle_deg=0, strength=0, xlen=100):
    arr = np.zeros(np.shape(strength) + np.shape(drift_angle_deg) + (3,3))
    if np.shape(strength):
        strength = strength[:, None, None]
    angle = np.deg2rad(drift_angle_deg + scan_rotation_deg)
    arr[...] = np.eye(3)
    s_sin = strength*np.sin(angle)
    s_cos = strength*np.cos(angle)
    arr[..., 0,0] = 1 - s_cos
    arr[..., 0,1] = -s_cos*xlen
    arr[..., 0,2] = -s_cos    
    arr[..., 1,0] = -s_sin
    arr[..., 1,1] = 1 - s_sin*xlen
    arr[..., 1,2] = -s_sin
    return arr.squeeze()

def warp_image_cuda(img, scanrotation, strength, drift_angle, nan=False):
    shift_x, shift_y = np.array(img.shape) / 2
    main_transform = transform_drift_scan(
         -scanrotation, 
         drift_angle, 
         strength, 
         img.shape[-2])
    T = (
    Affine2D().translate(-shift_x, -shift_y)
    + Affine2D(main_transform).rotate_deg(scanrotation).translate(shift_x, shift_y)
    )

    cval = cp.nan if nan else 0 # value of pixels that were outside the image border
    return affine_transform(
        img, 
        cp.linalg.inv(cp.array(swap_transform_standard(T.get_matrix()))),
        order=1,
        cval=cval)

def warp_image(img, scanrotation, strength, drift_angle):
    shift_x, shift_y = np.array(img.shape) / 2
    T = transform_drift_scan(
         scanrotation, 
         drift_angle, 
         strength, 
         img.shape[0]) 
    T = (
    Affine2D()
    .translate(-shift_x, -shift_y)
    + Affine2D(T)
    + Affine2D()
    .rotate(-scanrotation)
    .translate(shift_x, shift_y)
    )
    
    return warp(img, np.linalg.inv(T.get_matrix()))


def warp_and_correlate(images, scanangles, drift_angles, strengths):
    all_maxes = []
    pairs = []
    for ai, drift_angle in tqdm(
        enumerate(drift_angles), 
        desc="Iterating through drift angles", 
        total=len(drift_angles),
        leave=False):
        for si, strength in tqdm(
            enumerate(strengths), 
            desc="Iterating through drift strengths", 
            total=len(strengths),
            leave=False):
            current_maxes = []
            current_shifts = []
            warped_imgA = warp_image_cuda(images[0], scanangles[0], strength, drift_angle)
            for img, scanangle in tqdm(
                zip(images[1:], scanangles[1:]), 
                desc="Warping images", 
                total=len(images[1:]),
                leave=False):
                warped_imgB = warp_image_cuda(img, scanangle, strength, drift_angle)
                m = hybrid_correlation(warped_imgA, warped_imgB, fit_only=True)
                current_maxes.append(m.get())
            all_maxes.append(current_maxes)
            pairs.append((drift_angle, strength))
    return all_maxes, pairs

def get_best_angle_strength_shifts(images, scanangles, angle_limits_step, strength_limits_step):
    
    low, high, step = angle_limits_step
    drift_angles = np.arange(low, high+step, step) % 360
    low, high, step = strength_limits_step
    drift_strengths = np.arange(low, high+step, step)
    
    all_maxes, pairs = warp_and_correlate(images, scanangles, drift_angles, drift_strengths)
    i = np.array(all_maxes).mean(1).argmax() # take the mean over each set of images
    best_drift_angle, best_drift_strength = pairs[i]
    return best_drift_angle, best_drift_strength

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

    def __init__(
        self, 
        atoms=None, positions=None, numbers=None,
        scan_rotation = 0, jitter_strength=0,
        jitter_horizontal=True, jitter_vertical=False,
        pixel_size=0.1, sigma=0.4, power=1.8, 
        drift_strength = 0, drift_vector=[1,0], centre_drift=True,
        square = False, vacuum=5.0):
        
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
        
        self.create_probe_positions()
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
            mean = XY.mean(axis=(-1,-2))[:, None]
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
    