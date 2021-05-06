from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

cmdclass = {}
cmdclass.update({'build_ext': build_ext})
ext_modules = [Extension("cython_nms", ["utils/cython_nms.pyx"])]
for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(name='yolact_edge',
      version='0.0.1',
      package_dir={'yolact_edge': ''},
      packages=['yolact_edge', 'yolact_edge.data', 'yolact_edge.layers',
                'yolact_edge.layers.functions', 'yolact_edge.layers.modules',
                'yolact_edge.utils'],
      include_package_data=True,
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      )
