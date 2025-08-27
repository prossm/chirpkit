from setuptools import setup, find_packages

setup(
    name='chirpkit',
    version='0.1.2',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('models/trained', [
            'models/trained/insect_classifier_471species.pth',
            'models/trained/insect_classifier_471species_label_encoder.joblib',
            'models/trained/insect_classifier_471species_info.json'
        ]),
    ],
    include_package_data=True,
    install_requires=[
        # Core dependencies with broader compatibility ranges
        'numpy>=1.21.0,<2.0.0',
        'scikit-learn>=1.0.0',
        'pandas>=1.3.0',
        'librosa>=0.9.0',
        'soundfile>=0.10.0',
        'joblib>=1.0.0',
        'requests>=2.25.0',
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
            'tensorflow-macos>=2.12.0,<3.0.0',
        ],
        'tensorflow-gpu': [
            'tensorflow[and-cuda]>=2.12.0,<3.0.0',
        ],
        
        # PyTorch variants
        'torch': [
            'torch>=1.9.0',
            'torchvision>=0.10.0',
            'torchaudio>=0.9.0',
        ],
        'torch-cpu': [
            'torch>=1.9.0',
            'torchvision>=0.10.0',
            'torchaudio>=0.9.0',
        ],
        
        # Visualization and experiment tracking
        'viz': [
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'wandb>=0.12.0',
        ],
        
        # UI components
        'ui': [
            'gradio>=3.0.0',
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
        
        # Development dependencies
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
        ],
        
        # Complete installation with recommended backends
        'full': [
            'tensorflow-macos>=2.12.0,<3.0.0; sys_platform == "darwin"',
            'tensorflow>=2.12.0,<3.0.0; sys_platform != "darwin"',
            'torch>=1.9.0',
            'torchvision>=0.10.0',
            'torchaudio>=0.9.0',
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'gradio>=3.0.0',
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
