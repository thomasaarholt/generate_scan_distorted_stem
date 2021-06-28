import numpy as np

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
    return T.T

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
    
def add_shifts_and_rotation_to_transform(transform, image_shape, scan_rotation_deg, post_shifts=[0,0]):
    shift1, shift2 = (np.array(image_shape) - 1) / 2
    post_shift1, post_shift2 = post_shifts[::-1]

    T = (
        Affine2D().translate(shift1, shift2)
        + Affine2D(transform)
        .rotate_deg(scan_rotation_deg)
        .translate(-shift1, -shift2)
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
    T2 = swap_transform_standard(T.get_matrix()).copy()
    T3 = cp.array(T2)
    T4 = cp.linalg.inv(T3)
    return affine_transform(
        img, 
        T4,
        order=1,
        cval=cval)
        
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