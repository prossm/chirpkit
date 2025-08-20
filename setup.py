from setuptools import setup, find_packages

setup(
    name='insect-sound-classifier',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch', 'torchvision', 'torchaudio', 'tensorflow', 'librosa', 'soundfile',
        'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'wandb', 'fastapi', 'uvicorn', 'pytest'
    ],
    entry_points={
        'console_scripts': [
            'download_data=scripts.download_data:main',
            'preprocess_data=scripts.preprocess_data:main',
            'train_model=scripts.train_model:main',
            'deploy_model=scripts.deploy_model:main',
        ],
    },
)
