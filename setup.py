from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'PyCorrCPI - corelation analysis for charged-particle imaging experiments'

setup(
        name="PyCorrCPI", 
        version=VERSION,
        author="Felix Allum",
        author_email="fallum@stanford.edu",
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'matplotlib', 'scipy', 'pandas', 'numba'],
        url='https://github.com/f-allum/PyCCorrCPI/',
        download_url='https://github.com/f-allum/PyCESim/archive/refs/tags/v0.0.1.tar.gz',
        keywords=['Coincidence', 'Covariance', 'Cumulant', 'Velocity-map Imaging']
)