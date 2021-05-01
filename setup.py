from setuptools import setup, find_packages

setup(name='microgrid',
      version='0.0.1',
      description='Multi-Agent Microgrid Environment',
      url='https://github.com/hepengli/multiagent-microgrid-envs',
      author='Hepeng Li',
      author_email='hepengli@uri.edu',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl', 'pandapower']
)
