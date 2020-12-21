import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

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

def rotation_matrix(deg):
    c = np.cos(np.deg2rad(deg))
    s = np.sin(np.deg2rad(deg))
    return np.array([[c, -s],[s, c]])

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
    
def add_drift(Xshape, drift_vector = [1,0], strength=1e-4):
    vector = -cp.array(drift_vector)
    probe_indices = cp.arange(cp.prod(cp.array(Xshape))).reshape(Xshape)
    return (strength * vector * probe_indices.T[..., None]).T

def add_line_jitter(XYshape, strength = 0.3):
    jitter = cp.zeros(XYshape)
    jitter[0] += 0.3*(2*cp.random.random((XYshape[1])) - 1)[:, None]
    return jitter

class ImageModel:

    def __init__(
        self, 
        atoms=None, positions=None, numbers=None,
        scan_rotation = 0, jitter_strength=0,
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
            self.probe_positions_cupy += add_line_jitter(XYshape, strength=self.jitter_strength)
            
        if self.scan_rotation:
            mean = self.probe_positions_cupy.mean(axis=(-1,-2))[:, None]
            self.probe_positions_cupy = (
                rotation_matrix_cupy(self.scan_rotation) @ (
                    self.probe_positions_cupy.reshape((2, -1)) - mean) + mean
            ).reshape((2, *XYshape[1:]))

        if self.drift_strength:
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
    