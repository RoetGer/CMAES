from distutils.core import setup

setup(name='cmaes',
            version='0.1',
            description='Implements the Covariance Matrix Adaption method',
            #url='http://github.com/storborg/funniest',
            author='Gernot Roetzer',
            license='MIT',
            packages=['cmaes', 'tests'],
            classifiers=['Development Status :: 3 - Alpha',
                         ('Topic :: Scientific/Engineering :: Artificial',
                         'Intelligence'),
                         'License :: OSI Approved :: MIT License'],
            setup_requires=['pytest-runner'],
            tests_require=['pytest'])
