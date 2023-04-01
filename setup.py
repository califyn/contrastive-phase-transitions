from setuptools import setup, find_packages

setup(
    name='ldcl',
    version='0.0.1',
    install_requires=[
        'importlib-metadata; python_version == "3.8"',
    ],
    packages=['ldcl.optimizers', 'ldcl.tools', 'ldcl.losses', 'ldcl.data', 'ldcl.models', 'ldcl.plot'],
)
