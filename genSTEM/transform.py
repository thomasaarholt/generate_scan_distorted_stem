import numpy as np

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
    
