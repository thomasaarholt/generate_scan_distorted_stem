import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from tqdm.auto import tqdm
from matplotlib.transforms import Affine2D
from matplotlib.patches import Arrow


from .utils import cp, asnumpy

def get_atoms():
    "Simple interesting-looking structure for testing"
    from ase.spacegroup import crystal
    from ase.build import make_supercell

    a = 9.04
    skutterudite = crystal(('Co', 'Sb'),
                        basis=[(0.25, 0.25, 0.25), (0.0, 0.335, 0.158)],
                        spacegroup=204,
                        cellpar=[a, a, a, 90, 90, 90])

    atoms = make_supercell(skutterudite, np.diag([4,4,1]))
    atoms.numbers[330] = 80
    atoms.numbers[331] = 80
    atoms.numbers[346] = 80
    return atoms

def get_rotation_series(atoms = None, vacuum=10., pixel_size = 0.1, nImages=4, minScanAngle=0, maxScanAngle=360, drift_speed=5, drift_angle=None, jitter_strength=0., centre_drift=True, random_offset=False, square=True, **kwargs):
    '''Quickly generate images of a atoms object taken at various scan angles.
    TODO: Add option for random translation on each image, so that we don't accidentally have a perfectly centered image stack
    
    Parameters
    ----------
    atoms: :class:`ase.Atoms`
        ASE Atoms object descirbing the crystal structure along the viewing direction.
    vacuum: float
        Vacuum padding in Å. Add whitespace around image.
    pixel_size: float
        Pixel dimensions in Å. Affects image resolution.
    
    nImages: int
        Number of images in the rotation series which will have uniform scan rotations between minScanAngle and maxScanAngle.
    minScanAngle: float, int
        Minimum scan rotation angle of the fast-scan direction in degrees.
    maxScanAngle: float, int
        Maximum scan rotation angle of the fast-scan direction in degrees.
    
    drift_speeed: int.
        Total drift speed in units of drift_pixels/total_image_pixels.
    drift_angle: float, int
        Angle of drift in degrees.  This provides the angle of the unit vector provided to ImageModel.
    jitter_strength: float
        Shifts each scanline by a random factor.
    centre_drift: float
        Centres the drift so the image doesn't drift out of frame
    **kwargs
        Additional parameters accepted by ImageModel.
    
    Returns
    -------
    np.array(dtype=float)
        Rotation image series as a numpy array.
    '''
    if atoms is None:
        atoms = get_example_atoms()
    
    if drift_angle is None:
        random_angle = np.random.random() * 2*np.pi
    else:
        random_angle = drift_angle
    drift_vector = [np.cos(random_angle),np.sin(random_angle)]
    
    images = []

    scanangles = np.linspace(minScanAngle, maxScanAngle, nImages, endpoint=False)
    models = []
    for scanangle in tqdm(scanangles):
        model = ImageModel(atoms, scan_rotation=scanangle,
                    pixel_size=pixel_size, vacuum=vacuum,
                        drift_speed=drift_speed, 
                        drift_vector=drift_vector,
                        jitter_strength=jitter_strength, 
                        centre_drift=centre_drift,
                        jitter_vertical=True,
                        fast=False, random_offset=random_offset, 
                        **kwargs
                    )
        models.append(model)
        img = model.generate()
        if square:
            side = np.minimum(*img.shape)
            images.append(img[:side, :side])
        else:
            images.append(img)
    images = cp.stack(images)
    print(f"Size: {images.nbytes / 1e9} GB")
    print(f"Shape: {images.shape}")
    return images, scanangles, random_angle, models
    
def drift_points_readable(shape=(10,10), drift_speed=0, drift_angle=0):
    '''Calculate pixel coordinates for an image with uniform drift.
    
    Parameters
    ----------
    shape: tuple of int
        Shape of image array.
    drift_speeed: int.
        Total drift speed in units of drift_pixels/total_image_pixels.
    drift_angle: float, int
        Angle of drift in degrees.  This provides the angle of the unit vector provided to ImageModel.
        
    Returns
    -------
    ndarray of new pixel coordinates after drift with shape (*shape, 2)
    '''
    lenX, lenY = shape
    drift_vector = (rotation_matrix(drift_angle) @ [1,0]) * drift_speed
    drift = np.zeros(2)
    arr = np.zeros((lenX, lenY, 2))
    for yi in range(lenY):
        for xi in range(lenX):
            drift += drift_vector
            arr[xi, yi] = (xi, yi) - drift
    return arr

def drift_points_readable2(shape=(10,10), drift_speed=0, drift_angle = 0):
    '''Calculate pixel coordinates for an image with uniform drift.
    
    Parameters
    ----------
    shape: tuple of int
        Shape of image array.
    drift_speeed: int.
        Total drift speed in units of drift_pixels/total_image_pixels.
    drift_angle: float, int
        Angle of drift in degrees.  This provides the angle of the unit vector provided to ImageModel.
        
    Returns
    -------
    ndarray of new pixel coordinates after drift with shape (*shape, 2)
    '''
    lenX, lenY = shape
    drift_vector = (rotation_matrix(drift_angle) @ [1,0]) * drift_speed
    drift = np.zeros(2)
    arr = np.zeros((lenX, lenY, 2))
    for yi in range(lenY):
        for xi in range(lenX):
            drift += drift_vector
            arr[xi, yi] = (xi, yi) - drift
    return arr

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
    '''Turn a 2D points array (N, 2) into (N, 3) by padding with ones.
    Used to calculate transform matrix between two sets of points.
    '''
    return np.hstack([arr_of_2d, np.ones((len(arr_of_2d),1))])
    
def calculate_transform_matrix(points, prime):
    '''Calculate the affine transform matrix needed to turn one set of points into another.

    Parameters
    ----------
    points: array of shape (N, 2)
    prime: array of shape (N, 2)

    Returns
    -------
    transform matrix: array of shape (3,3)
    '''
    points = extend_3D_ones(points)
    prime = extend_3D_ones(prime)
    T, *_ = np.linalg.lstsq(points, prime, rcond=None)
    return T

def transform_points(points: "(2, N)", transform: "(3,3)"):
    '''Transform a 2D points array (2, N) by a 3x3 transform

    Turns the list of points into a (N, 3)  array, transforms
    and then removes the third coordinate again.
    Supports cp as well!
    '''
    shape = points.shape
    points = points.reshape((2, -1))
    points = extend_3D_ones(points.T).T
    prime = transform @ points
    return prime[:2].reshape(shape)

def get_and_plot_peaks(data, average_distance_between_peaks=80, threshold = 1):
    import scipy.ndimage as ndimage
    import scipy.ndimage.filters as filters
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
    '''Creates a symetric 2d-Gaussian.
    
    Parameters
    ----------
    x,y: ndarray
        Spatial coordiantes.
    A: float, int
        Amplitude
    xc, yc: float, int
        Gaussian center position.
    sigma: float, int
        Standard deviation.
    
    Returns
    -------
    ndarray of gaussian intensity
    '''
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
    
def add_drift(XYshape, drift_vector = [1,0], drift_speed=1e-4):
    '''Provide the pixel cordinate shift as a result of drift.
    
    Parameters
    ----------
    XYshape: tuple of int
        Shape of the probe positions. (2, X, Y)
    drift_vector: length-2 vector
        Direction of drift.
    drift_speed: float
        Drift speed in number of pixels.
        Automatically divided by total number of images pixels within the function.
        Should be 0-10.
    
    Returns
    -------
    ndarray of pixel shift after drift.
    '''
    drift_speed /= np.prod(XYshape) #total number of pixels
    drift_vector = -cp.array(drift_vector)
    probe_indices = cp.arange(cp.prod(cp.array(XYshape))).reshape(XYshape)
    return (drift_speed * drift_vector * probe_indices.T[..., None]).T
    

def add_line_jitter(XYshape, strength = 0.3, horizontal=True, vertical=False, ):
    '''Shift pixel rows and columns to simulate jittering in a STEM image.
    
    Parameters
    ----------
    XYshape: tuple of int
        Shape of probe positions. (2, X, Y)
    strength: float, int
        Strength of jitter.
        Applied to each row or column as :math:`strength*(2*random_number-1)` such that the pixel shift is in the interval [strength, strength).
    horizontal: bool
        Add jitter along the horizontal direction.
    vertical: bool
        Add jitter along the vertical direction.
        
    Returns
    -------
    ndarray of Å shift after jitter.
    '''
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
    '''Create a STEM-HAADF like image from a list of positions and atomic numbers, or from
    an ASE atoms object. 
    Images are generated by placing a 2D gaussian on each atom XY position.
    If a list of positions and numbers, positions should have shape (N, 2) and numbers shape (N,)
    
    Parameters
    ----------
    scan_rotation: float, int
        Angle of the fast-scan direction in degrees.
    drift_speed: float
        Automatically divided by image shape - should be 0-10.
    drift_vector: length-2 vector
        Direction of drift.
    pixel_size: float
        Pixel dimensions in Å. Affects image resolution.

    jitter_strength: float
        Shifts each scanline by a random factor.
    jitter_horizontal: bool
        Shift scanline leftright by above.
    jitter_vertical: bool
        Shift scnaline updown by above.
    sigma: float
        Standard deviation of 2D gaussian representing atomic columns.
    power: float
        HAADF n-factor - ~1.4-2.0

    centre_drift: bool
        Shift image borders so drifted image is centered.
    square: bool
        Make image square.
    vacuum: float
        Vacuum padding in Å. Add whitespace around image.
    fast: bool
        Only compute one layer of unique atoms
    random_offset: bool, float
        Add a random offset to the initial probe position to avoid
        the generated images automatically being centered wrt eachother
    '''

    def __init__(
        self, 
        atoms=None, positions=None, numbers=None,
        scan_rotation = 0, drift_speed = 0, drift_vector=[1,0], 
        pixel_size=0.1, jitter_strength=0,
        jitter_horizontal=True, jitter_vertical=False,
        sigma=0.4, power=1.8, 
        centre_drift=True, square = False, vacuum=5.0, fast=False,
        random_offset=False):

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

        self.random_offset = random_offset

        self.create_probe_positions()
        self.create_parameters()
        
    def init_sympy(self):
        xy = sp.symbols('x y')
        parameters = sp.symbols('A xc yc sigma', cls=sp.IndexedBase)
        i,n = sp.symbols("i n", integer=True)
        self.symbols = xy + parameters + (n,)
        A, xc, yc, sigma = parameters
        
        Gauss = SympyGaussian2D(xy[0], xy[1], A[i], xc[i], yc[i], sigma[i])
        model = sp.Sum(Gauss, (i, 0, n-1))
        self.model = model

    def create_probe_positions(self):
        '''Create probe positions with experimental artifacts.'''
        xlow, ylow = self.atom_positions.min(0) - self.margin
        xhigh, yhigh = self.atom_positions.max(0) + self.margin
        scale = (xhigh - xlow)/100
        if self.random_offset:
            if self.random_offset is True:
                self.random_offset = 5
            dX = (2*np.random.random()-1) * self.random_offset
            dY = (2*np.random.random()-1) * self.random_offset
            xlow += dX
            xhigh += dX
            ylow += dY
            yhigh += dY

        if self.square:
            xlow = ylow = min(xlow, ylow)
            xhigh = yhigh = max(xhigh, yhigh)
        xrange = cp.arange(xlow, xhigh+scale, self.pixel_size)
        yrange = cp.arange(ylow, yhigh+scale, self.pixel_size)
        self.probe_positions = cp.stack(cp.meshgrid(xrange, yrange))
        XYshape = self.probe_positions.shape

        if self.jitter_strength:
            self.jitter = add_line_jitter(
                XYshape = XYshape, 
                strength=self.jitter_strength, 
                horizontal=self.jitter_horizontal, 
                vertical=self.jitter_vertical) * self.pixel_size
            self.probe_positions += self.jitter 

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
        'Create the parameters that will describe the 2D-Gaussian distribution asigned to atoms.'
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
