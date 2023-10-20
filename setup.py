from setuptools import setup, find_packages

setup(
  name = 'zorro-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.1.1',
  license='MIT',
  description = 'Zorro - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/zorro-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'multimodal fusion'
  ],
  install_requires=[
    'beartype',
    'einops>=0.4',
    'torch>=1.6',
    'torchaudio'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
