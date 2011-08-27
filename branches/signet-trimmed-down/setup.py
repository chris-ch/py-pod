from setuptools import setup, find_packages
import sys, os

version = '1.0-sig'

setup(name='pod',
      version=version,
      description='Proper Orthogonal Decomposition',
      long_description="""\
The pod package is an implementation of a Proper Orthogonal Decomposition \
(POD) method. The idea is to find a lower dimensional description of a set \
of measure. POD, Principal Component Analysis  (PCA) and Singular Value \
Decomposition are closely related concepts. This package contains processing \
algorithms for decomposing an input using a set of predefined signals""",
      classifiers=['Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Information Analysis'        
      ], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='Vector Space Projection Proper Orthogonal Decomposition Singular Value Decomposition POD PCA SVD',
      author='Christophe Alexandre',
      author_email='calexandre@signetmanagement.com',
      url='',
      license='Signet Group',
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
