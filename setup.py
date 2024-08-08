from setuptools import setup,find_packages


with open('requirements.txt', 'rt') as f:
    required_packages = [l.strip() for l in f.readlines()]

setup(name='WHITENet',
	version='1.0.0',
	description='White matter HyperIntensities Tissue Extraction using deep-learning Network',
	author='Camille Cathala',
    author_email='camille.cathala@epfl.ch',
    url='https://github.com/cathalacamille/WHITE-Net',
	install_requires=required_packages,
    entry_points={
        'console_scripts': [
            'apply_whitenet=WHITENet.WHITENet:main',  # This creates a command-line tool
        ],
    },
    package_data={
        'WHITENet': ['white_net_FLAIR.pt','white_net_FLAIR_T1w.pt'],  # Include the file in the package
    },
	packages=find_packages(),
	include_package_data=True)


