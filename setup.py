from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='PLM',
    version='0.0.0',
    packages=['PLM', 'BELM', 'BELM.net'],
    url='',
    license='BSD (3-clause)',
    author='Charlie Dondapati',
    author_email='cdondapa@lakeheadu.ca',
    description='Bidirectional ELM neural network implementation for regression problem',
    install_requires=required,
    include_package_data=True,
    exclude_package_data={"": ["worksheet*"]},
    keywords='BELM PLM  regression '
)
