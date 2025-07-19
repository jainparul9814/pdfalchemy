from setuptools import setup, find_packages

setup(
    name='pdfalchemy',
    version='0.1.0',
    description='A Python library for advanced PDF manipulation and processing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jainparul9814/pdfalchemy',
    author='Parul Jain',
    author_email='jainparul9814@gmail.com',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pydantic>=2.0.0',
        'pdf2image>=1.16.0',
        'Pillow>=9.0.0',
        'numpy>=1.21.0',
        'opencv-python>=4.8.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'pre-commit>=3.0.0',
        ],
        'docs': [
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'myst-parser>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'pdfalchemy=pdfalchemy.cli:main',
        ],
    },
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Markup',
    ],
    keywords=['pdf', 'document', 'processing', 'extraction', 'manipulation'],
    project_urls={
        'Homepage': 'https://github.com/jainparul9814/pdfalchemy',
        'Documentation': 'https://pdfalchemy.readthedocs.io/',
        'Repository': 'https://github.com/jainparul9814/pdfalchemy',
        'Bug Tracker': 'https://github.com/jainparul9814/pdfalchemy/issues',
    },
    include_package_data=True,
    package_data={
        'pdfalchemy': ['py.typed'],
    },
) 