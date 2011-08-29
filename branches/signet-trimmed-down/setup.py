from setuptools import setup, find_packages
import sys, os

version = '1.0'

setup(name='signet-pca',
      version=version,
      description='Principal Component Analysis',
      long_description="""\
This package implements an algorithm that linearly decomposes the track record \
of an asset into the performance of several reference factors that best \
explain its evolution.\
""",
      classifiers=['Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Information Analysis'        
      ], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='Vector Space Projection Principal Components Analysis',
      author='Christophe Alexandre',
      author_email='calexandre@signetmanagement.com',
      url='',
      license='Signet Group',
      packages=find_packages(exclude=['ez_setup', 'tests']),
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
