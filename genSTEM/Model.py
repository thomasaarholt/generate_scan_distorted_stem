import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from tqdm.auto import tqdm
from matplotlib.transforms import Affine2D
from matplotlib.patches import Arrow
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

from .utils import cp, asnumpy

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

def get(atoms = None, nImages=4, drift_speed=5, pixel_size = 0.1, minScanAngle=0, maxScanAngle=360, drift_angle=None, vacuum=10., jitter=0.,  ):
    "Quickly generate images of a atoms object taken at various scan angles"
    if atoms is None:
        atoms = get_atoms()
    
    if drift_angle is None:
        random_angle = np.random.random() * 2*np.pi
    else:
        random_angle = drift_angle
    drift_vector = [np.cos(random_angle),np.sin(random_angle)]
    centre_drift = True
    
    images = []

    scanangles = np.linspace(minScanAngle, maxScanAngle, nImages, endpoint=False)
    for scanangle in tqdm(scanangles):
        m = ImageModel(atoms, scan_rotation=scanangle,
                    pixel_size=pixel_size, vacuum=vacuum,
                        drift_speed=drift_speed, 
                        drift_vector=drift_vector,
                        jitter_strength=jitter, 
                        centre_drift=centre_drift,
                        fast=False,
                    )
        img = m.generate()
        side = np.minimum(*img.shape)
        images.append(img[:side, :side])
    images = cp.stack(images)
    print(f"Size: {images.nbytes / 1e9} GB")
    print(f"Shape: {images.shape}")
    return images, scanangles, random_angle
    
def drift_points(shape=(10,10), drift_deg = 0, drift_speed=0):
    lenX, lenY = shape
    drift_vector = (rotation_matrix(drift_deg) @ [1,0]) * drift_speed
    arr = np.zeros((lenX, lenY, 2))
    drift = np.zeros(2)
    for yi in range(lenY):
        for xi in range(lenX):
            drift += drift_vector
            position = np.array((xi, yi))
            arr[xi, yi] = position - drift
    return arr

def drift_pointsYX(shape=(10,10), drift_deg = 0, drift_speed=0):
    "Rotate drift from along x axis, anticlockwise in degrees"
    lenY, lenX = shape
    drift_vector = (rotation_matrix(drift_deg) @ [1,0]) * drift_speed
    drift = np.zeros(2)
    positions = []
    for yi in range(lenY):
        for xi in range(lenX):
            drift += drift_vector
            position = np.array((xi, yi))
            positions.append(position - drift)
    return np.array(positions)

def plot(points, ax, lim=((),())):
    points = points.reshape((-1, 2), order='F')
    for i, xy in enumerate(points):
        rect = plt.Rectangle(xy-0.25, 0.5, 0.5, color=matplotlib.cm.get_cmap('RdYlBu')(i))
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



def Gaussian2D(x, y, A, xc, yc, sigma):
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

def add_ac_noise(shape, strength=0.5, dwelltime=1e-6, ac_freq=50):
    noise = np.zeros((2,) + shape)
    noise[0] = strength * np.sin(
        2*dwelltime / (1/ac_freq) * np.arange(np.prod(shape)) * np.pi
    ).reshape(shape)
    return noise
    
def add_drift(shape, drift_vector = [1,0], speed=1e-4):
    speed /= np.prod(shape)
    vector = -cp.array(drift_vector)
    probe_indices = cp.arange(cp.prod(cp.array(shape))).reshape(shape)
    return (speed * vector * probe_indices.T[..., None]).T

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
    an ASE atoms object. 
    Images are generated by placing a 2D gaussian on each atom XY position.
    If a list of positions and numbers, positions should have shape (N, 2) and numbers shape (N,)
    
    Can add features like:

    scan_rotation: deg
    drift_speed: float, Automatically divided by image shape - should be 0-10
    drift_vector: length-2 vector. Direction of drift
    pixel_size: float, Å, Affects to image resolution

    jitter_strength: float, shifts each scanline by a random factor
    jitter_horizontal: bool, shift scanline leftright by above
    jitter_vertical: bool, shift scnaline updown by above
    sigma: float, standard deviation of 2D gaussian representing atomic columns
    power: float, HAADF n-factor - ~1.4-2.0

    centre_drift: bool, Shift image borders so drifted image is centered
    square: bool, Make image square
    vacuum: float, Å, Add whitespace around image
    fast: bool, Only compute one layer of unique atoms
    """

    def __init__(
        self, 
        atoms=None, positions=None, numbers=None,
        scan_rotation = 0, drift_speed = 0, drift_vector=[1,0], 
        pixel_size=0.1, jitter_strength=0,
        jitter_horizontal=True, jitter_vertical=False,
        sigma=0.4, power=1.8, 
        centre_drift=True, square = False, vacuum=5.0, fast=False):
        
        if atoms:
            self.atom_positions = atoms.positions[:,:2]
            self.atom_numbers = atoms.numbers
        else:
            if not positions:
                raise AttributeError(
            "You must supply either an ase Atoms object or a list of positions and atomic numbers"
            )
            self.atom_positions = positions[:,:2]
            self.atom_numbers = numbers

        if fast: # For each unique XY position, only keep one atom. Much faster, but will miss atoms.
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

        self.drift_speed = drift_speed
        self.drift_vector = drift_vector
        self.centre_drift = centre_drift
        self.scan_rotation = scan_rotation
        self.square = square
        self.margin = vacuum
        
        self.create_probe_positions()
        self.create_parameters()
        
    def init_sympy(self):
        xy = sp.symbols('x y')
        parameters = sp.symbols('A xc yc sigma', cls=sp.IndexedBase)
        i,n = sp.symbols("i n", integer=True)
        self.symbols = xy + parameters + (n,)
        A, xc, yc, sigma = parameters
        
        Gauss = SympyGaussian2D(xy[0], xy[1], A[i], xc[i], yc[i], sigma[i])
        model = sp.Sum(Gauss, (i,0,n-1))
        self.model = model

    def create_probe_positions(self):
        xlow, ylow = self.atom_positions.min(0) - self.margin
        xhigh, yhigh = self.atom_positions.max(0) + self.margin
        scale = (xhigh - xlow)/100

        if self.square:
            xlow = ylow = min(xlow, ylow)
            xhigh = yhigh = max(xhigh, yhigh)
        xrange = cp.arange(xlow, xhigh+scale, self.pixel_size)
        yrange = cp.arange(ylow, yhigh+scale, self.pixel_size)
        self.probe_positions = cp.stack(cp.meshgrid(xrange, yrange))
        XYshape = self.probe_positions.shape
        
        if self.jitter_strength:
            self.probe_positions += add_line_jitter(
                XYshape = XYshape, 
                strength=self.jitter_strength, 
                horizontal=self.jitter_horizontal, 
                vertical=self.jitter_vertical)
            
        if self.scan_rotation:
            mean = self.probe_positions.mean(axis=(-1,-2))[:, None]
            self.probe_positions = (    
                cp.asarray(rotation_matrix(self.scan_rotation)) @ (
                    self.probe_positions.reshape((2, -1)) - mean) + mean
            ).reshape((2, *XYshape[1:]))

        if self.drift_speed:
            #speed = self.drift_speed / np.prod(XYshape[1:])
            drift = add_drift(XYshape[1:], self.drift_vector, self.drift_speed)
            self.probe_positions += drift
            
            if self.centre_drift:
                driftx, drifty = drift
                offsetx = driftx.max() if driftx.max() > -driftx.min() else driftx.min()
                offsety = drifty.max() if drifty.max() > -drifty.min() else drifty.min()
                self.probe_positions -= cp.array([offsetx, offsety])[:, None, None] / 2
        
    def create_parameters(self):
        xc, yc = self.atom_positions.T
        A = self.atom_numbers ** self.power
        sigma = np.ones(self.number_of_atoms) * self.sigma
        self.parameters = cp.asarray(np.array([A, xc, yc, sigma]))
        
    def generate_lambdify(self):
        self.init_sympy()
        func = sp.lambdify(self.symbols, self.model, modules = 'numpy')
        self.func = func
        return func(*self.probe_positions, *self.parameters, self.number_of_atoms)
    
    def generate_lambdify_cupy(self):
        self.init_sympy()
        func = sp.lambdify(self.symbols, self.model, modules = 'cupy')
        self.func_cupy = func
        return func(*self.probe_positions, *self.parameters, self.number_of_atoms)
    
    def generate(self):
        X, Y = self.probe_positions
        img = cp.zeros(X.shape)
        for parameters in self.parameters.T:
            img += Gaussian2D(X, Y, *parameters)
        return img
 
    def generate_cupy_ram(self):
        X, Y = self.probe_positions
        img = cp.sum(Gaussian2D(X[..., None], Y[..., None], *self.parameters), -1)
        return img
