from setuptools import setup, find_packages

setup(
    name="visprobe",
    version="0.2.0",
    description="Find robustness failures in your vision models in 5 minutes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Bilge Demirkaya",
    author_email="bilgedemirkaya07@gmail.com",
    url="https://github.com/bilgedemirkaya/VisProbe",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
        "Topic :: Security",
    ],
    keywords="adversarial-robustness computer-vision deep-learning neural-networks pytorch testing fuzzing security",

    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,

    python_requires=">=3.9",

    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy<2.0.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "pandas>=2.0.0",
        "pillow>=8.0.0",
        "scipy>=1.9.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],

    extras_require={
        "viz": ["altair>=4.2.0"],
        "adversarial": ["adversarial-robustness-toolbox>=1.18.0"],
        "bayesian": ["scipy>=1.9.0", "scikit-learn>=1.0.0"],
        "all": [
            "altair>=4.2.0",
            "adversarial-robustness-toolbox>=1.18.0",
            "scipy>=1.9.0",
            "scikit-learn>=1.0.0",
        ],
    },

    entry_points={
        'console_scripts': [
            'visprobe=visprobe.cli.cli:main',
        ],
    },
)