from setuptools import find_packages, setup

setup(
    name='disease-prediction',
    version='0.1.0',
    description='A production-ready multiple disease prediction system',
    author='jitu prajapati',
    author_email='jitu.prajapat0604@email@example.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'mlflow',
        'dagshub',
        'pyyaml',
        'kaggle',
        'dvc',
        'pyrebase4',
        'urllib3<2.0.0'
    ],
)
