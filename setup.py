# from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from setuptools import setup, find_packages


cmdclass = {}
cmdclass.update({'build_ext': build_ext})
ext_modules = [Extension("cython_nms", ["yolact_edge/utils/cython_nms.pyx"])]
for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(name='yolact_edge',
      version='0.0.1',
      package_dir={'yolact_edge': 'yolact_edge'},
      packages=find_packages(exclude=('data','calib_images','results')) + ['yolact_edge'],
      include_package_data=True,
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      )