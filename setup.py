from setuptools import setup, find_packages
import os
import sys

# Get version from package
def get_version():
    """Get version from chirpkit package"""
    version_file = os.path.join('src', 'chirpkit', '_version.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                # Extract version and strip whitespace, quotes, and newlines
                version = line.split('=')[1].strip()
                return version.strip('"\'')
    return '0.1.2'  # Fallback

setup(
    name='chirpkit',
    version=get_version(),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    # Models are downloaded on first use, not included in package
    include_package_data=False,
    install_requires=[
        # Core dependencies - compatible with modern ecosystem
        'numpy>=1.21.0',  # Remove upper bound for compatibility with modern projects
        'scikit-learn>=1.0.0',
        'pandas>=1.3.0',
        'librosa>=0.9.0',
        'soundfile>=0.10.0',
        'joblib>=1.0.0',
        'requests>=2.32.4',  # Security: CVE fixes in 2.32.4
        'PyYAML>=5.4.0',  # Configuration file support
        # BirdNET dependency — pin to v2.4.0, the latest release that still exposes
        # the embedding API this package calls: birdnet_analyzer.model.load_model(
        # class_output=False) and .embeddings(). The default branch (and the PyPI
        # build) removed/renamed these, so an unpinned install breaks embedding
        # extraction (1024-dim) at inference time.
        'birdnet-analyzer @ git+https://github.com/birdnet-team/BirdNET-Analyzer.git@v2.4.0',
        # Web interface (optional by default)
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
    ],
    extras_require={
        # TensorFlow variants for different platforms
        'tensorflow': [
            'tensorflow>=2.12.0,<3.0.0',
        ],
        'tensorflow-macos': [
            # tensorflow-macos maxes out at 2.16.2 as of Jan 2025
            'tensorflow-macos>=2.12.0,<=2.16.2',
            'tensorflow-metal>=1.0.0',
        ],
        'tensorflow-gpu': [
            'tensorflow[and-cuda]>=2.12.0,<3.0.0',
        ],
        
        # PyTorch variants - modern versions compatible with ecosystem
        'torch': [
            'torch>=2.0.0',  # Modern PyTorch for compatibility
            'torchvision>=0.15.0',
            'torchaudio>=2.0.0',
        ],
        'torch-cpu': [
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            'torchaudio>=2.0.0',
        ],
        
        # Visualization and experiment tracking
        'viz': [
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'wandb>=0.12.0',
        ],
        
        # UI components
        'ui': [
            'gradio>=5.0.0',
            'python-multipart>=0.0.20',  # Required by gradio for file uploads
        ],
        
        # Dataset utilities
        'datasets': [
            'kagglehub>=0.1.0',
        ],
        
        # Audio enhancement
        'audio-enhanced': [
            'essentia>=2.1',
            'resampy>=0.4.0',
        ],
        
        # Inference-only (minimal production deployment)
        'inference': [
            'torch>=2.0.0',  # Core ML backend
        ],
        
        # Development dependencies
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
        ],
        
        # Complete installation with recommended backends
        'full': [
            'tensorflow-macos>=2.12.0,<=2.16.2; sys_platform == "darwin"',
            'tensorflow-metal>=1.0.0; sys_platform == "darwin"',
            'tensorflow>=2.12.0,<3.0.0; sys_platform != "darwin"',
            'torch>=2.0.0',  # Modern PyTorch
            'torchvision>=0.15.0',
            'torchaudio>=2.0.0',
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'gradio>=5.0.0',
            'python-multipart>=0.0.20',
            'wandb>=0.12.0',
            'kagglehub>=0.1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'chirpkit=chirpkit.cli:main',
            'chirpkit-doctor=chirpkit.cli:doctor',
            'chirpkit-fix=chirpkit.cli:fix',
        ],
    },
    python_requires='>=3.8',
    author='Patrick Metzger',
    description='A robust toolkit for insect sound classification and analysis',
    long_description=open('README.md').read() if 'README.md' in locals() else '',
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
    ],
)
