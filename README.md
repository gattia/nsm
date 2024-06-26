![Build Status](https://github.com/gattia/NSM/actions/workflows/build-test.yml/badge.svg?branch=main)<br>
|[Documentation](http://anthonygattiphd.com/NSM/)|



# Introduction

This pacakge is meant to develop generative deep learning models for creating human anatomy. The initial focus is on musculoskeletal tissues, particular of the knee. 

Steps to update this package for new repository: 
4. update `requirements.txt` and `dependencies` in `pyproject.toml`
     - To do - can dependencies read/update from requirements.txt?


# Install Instructions

```bash
conda create -n NSM python=3.8
conda activate NSM
```


``` bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
make requirements
```

```bash
# install in development mode... 
pip install -e .
```













# Install - only tried for SDF stuff
<!-- mamba install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
mamba install -c fvcore -c iopath -c conda-forge fvcore iopath -y
mamba install -c bottler nvidiacub -y
mamba install pytorch3d -c pytorch3d -y

make requirements -->


mamba install pytorch torchvision pytorch-cuda fvcore iopath nvidiacub pytorch3d -c pytorch -c nvidia -c fvcore -c iopath -c conda-forge -c bottler -c pytorch3d -y
make requirements

git clone https://github.com/gattia/pymskt
cd pymskt
make requirements 
make install


# Installation

You should be able to install this by cloning, navigating to this root directory, and installing with pip:

Install pymskt: https://github.com/gattia/pymskt




```
git clone https://github.com/gattia/NSM
cd NSM

conda create -n NSM python=3.8

pip install . 

# OR
make install
```

### Install for development
This method of install will install in editable mode. This means that the code wont be packaged and saved
in your python's `site-packages`, instead `site-packages` will point to this directory. So, if the code 
changes in here, so will the version of this package used on your local build. 
```
git clone https://github.com/gattia/NSM
cd NSM

conda create -n NSM python=3.8

make dev
make install-dev
```

# Examples

Navigate to the examples directory and run the scripts: 
```bash
cd examples
python examples/example_1.py
```

# Development / Contributing

## Tests
The test can be run by: 

```bash
pytest
```

or 
```bash
make test
```

Inidividual tests can be run by running 

```
python -m pytests path_to_test
```

## Coverage
- Coverage results/info requires `coverage` (`conda install coverage` or `pip install coverage`).
- These should be installed automatically with one of the  `make dev` commands.
- You can get coverage statistics by running: 
    - `coverage run -m pytest`
    or if using make: 
    - `make coverage`
        - This will save an html of the coverage results. 

### note about coverage:
    - Coverage runs by seeing how much of the code-base is covered when you run the command after coverage. 
    In this case, it is looking to see how much of the code-base is covered when we run the tests. 

## Contributing
If you want to contribute, please read over the documentaiton in `CONTRIBUTING.md`

## Docs
To build the docs, run `make docs`. If you want the docs published on gihutb, you need to activate github page.
Go to the `Settings` tab on your github repo, under `Pages` on the left, turn GitHub Pages on, and select the
home dir for the docs to be `/docs` on the `main` branch. Example here:  

![Setup Docs on Github Pages](media/setting_up_docs_automatically.png)


# License
MIT License