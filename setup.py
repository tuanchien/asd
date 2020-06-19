from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='ava_asd',
    description='Scripts and models for the AVA Active Speaker Detection dataset.',
    license='Apache License Version 2.0',
    author='Tuan Chien, James Diprose',
    url='https://github.com/tuanchien/asd',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'ava-download = download:main',
            'ava-extract = extract:main',
            'ava-train = train:main',
            'ava-evaluate = evaluate:main'
        ]
    },
)
