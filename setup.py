# Some inspiration from:
#   https://github.com/ArjanCodes/2023-package/blob/main/setup.py

from setuptools import find_packages, setup

with open('lib/README.md', 'r') as f:
    long_description = f.read()

setup(
    name='simple_portfolio_ledger',
    version='0.0.0',
    description='A simple ledger for investment tracking.',
    package_dir={'': 'lib'},
    packages=find_packages(where='lib'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url='...',
    author='Felipe Caballero',
    # author_email='mail@mail.mail',
    # license='MIT',
    # classifiers=[
    #     'License :: OSI Approved :: MIT License',
    #     'Programming Language :: Python :: 3.10',
    #     'Operating System :: OS Independent',
    # ],
    install_requires=['pandas>=2'], # Might need review
    # extras_require={
    #     'dev': ['pytest>=7.0', 'twine>=4.0.2'],
    # },
    python_requires='>=3',
)