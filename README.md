# genSTEM
This packages generates images of atomic structures as a function of scan rotation angle. It pretends to be a Scanning Transmission Electron Microscope (STEM). 
The images are not proper STEM-simulations, just a HAADF approximation using two-dimensional gaussians.

![Example fo generated images](images/Generated.png?raw=true "Generate_images")

As input it can take an ASE atoms object or a list of 2D/3D positions and atomic numbers, and creates images as a function of:
- scan rotation angle
- drift in the XY plane at various strengths
- scanning distortion that either 
    - shifts lines horizontally or vertically
    - shifts individual probe positions
    
To get started, we recommend trying the [Affine Correction Example Notebook](examples/AffineCorrectionExample.ipynb) in the  [examples](examples/) directory.
- Numpy
- Matplotlib 
- scipy
- ase
- sympy
- tqdm
- cupy (optional, speeds things up, but needs CUDA 11.1 (not newer - yet)
- ipympl (to show inline interactive figures in the notebook)

The easiest way to have everything is to create an anaconda environment with the required packages, and
install genSTEM directly from the github. For the best, "widescreen" viewing experience, we recommend jupyter lab.

```bash
conda create --name stem numpy matplotlib scipy ase sympy tqdm notebook ipympl # jupyterlab
conda activate stem
pip install https://github.com/thomasaarholt/generate_scan_distorted_stem/archive/main.zip
```

The final aim of the correction package is to fully correct scan distortion in STEM images. So far, a global affine correction for constant drift has been implemented:
![Example of drift-corrected images](images/Corrected.png?raw=true "Corrected Images")
