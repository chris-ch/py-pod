from setuptools import setup, find_packages
import sys, os

version = '1.0'

setup(name='pypod',
      version=version,
      description='Proper Orthogonal Decomposition',
      long_description="""\
Replicating an input signal using a library of reference signals""",
      classifiers=['Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Information Analysis'        
      ], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='Vector Space Projection Proper Orthogonal Decomposition POD PCA',
      author='Christophe Alexandre',
      author_email='ch.alexandre at bluewin dot ch',
      url='http://code.google.com/p/py-pod',
      license='LGPL',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=True,
      install_requires=[
          # -*- Extra requirements: -*-
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,     
      test_suite='nose.collector',
      tests_require='nose',
      )
