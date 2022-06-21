# Classification of Plasma dynamics by Self-injection Radiation
Laser-plasma accelerators (LPA) are the most promising technology to shrink the size of conventional, large-scale particle accelerators thus making them less costly, as well as increasing availability and access in science, industry, and medicine. In order to understand the complex plasma dynamics induced by the laser-matter interaction, ab-inito particle-in-cell simulations are the method of choice. They are capable of predicting the nanometer-femtosecond-scale dy- namics of the laser-plasma interaction. One method to access the plasma dynamics of LPAs is the emitted radiation of the electrons. This spectrally- and directionally-resolved self-emission radiation contains a lot of information on the plasma dynamics, such as self-injection radiation, which is emitted when electrons are injected into the plasma cavity and thus marks the start of the acceleration process. Radiation is relatively easy to measure in experiments, however the reconstruction of plasma dynamics based on the acquired radia- tion can be seen as an ambiguous inverse problems. That problem is currently solved by a computationally demanding parameter optimization of particle-in- cell codes.

This codes applies invertible neural networks tailored for solving inverse problems of HZDR's [NF4IP](https://github.com/Photon-AI-Research/NF4IP) library to the classification of self-injection radiation. 

## Example usage of NF4IP framework
This repository demonstrates the basic usage of HZDR's [NF4IP](https://github.com/Photon-AI-Research/NF4IP) framework for solving inverse problems by normalising flows. The general structure of the repository was created by
```
$ NF4IP generate project <NAME>
```
The workflow of the program can then be controlled by enabling needed plugins as well as defining hooks that inject knowledge into the training loop. The configuration file can be found in folder `config` and is called [lwfa.yml](lwfa/config/lwfa.yml). Additional workflows of our project are contributed in file [lwfa.py](lwfa/controllers/lwfa.py) of folder `controllers`. Our dataset is defined in the same folder, [dataset_rad_energy.py](lwfa/controllers/dataset_rad_energy.py). 

The training of the invertible neural network is derived from [NF4IP](https://github.com/Photon-AI-Research/NF4IP), i.e. the following operation `lwfa model train`. However, this project also required a tailored VAE implementation for compression of the radiation spectra which is not contained in NF4IP. We therefore had to implement an own controller [lwfa.py](lwfa/controllers/lwfa.py) that contributed the training of the VAE model `lwfa model train-vae`.

## Installation

```
$ pip install -r requirements.txt
$ pip install horovod[torch]
$ pip install git+https://github.com/VLL-HD/FrEIA.git
$ pip install -e NF4IP
$ pip install -e ClassifyInjectionRadiation_by_NF4IP
```

## Running the Classifier for Self-injection Radiation

This following command provides an overview about all commands.
$ lwfa --help

The invertible neural network can be trained by running

$ lwfa lwfa train

while the variational autoencoder for compression of the radiation spectra is trained by

$ lwfa lwfa train-vae


## Deployments

### Docker

Included is a basic `Dockerfile` for building and distributing `LWFA`,
and can be built with the included `make` helper:

```
$ make docker

$ docker run -it lwfa --help
```