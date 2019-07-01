# Real-world instantaneous NOx emissions prediction

Development of an artificial neural network architecture to predict vehicular NOx emissions in real-time.

### Prerequisites

To run on CPUs only:

```
Python 3.6.3
Numpy 1.13.3
Tensorflow 1.9.0
Pandas 0.21.0
Matplotlib 2.10
```

To run on GPUs, CUDA and CuDNN must be installed in addition:

```
CUDA 9.0
CuDNN 7
```

## Getting Started

Download the folder. Put your training file in the folder Data/Training and your testing file in the folder Data/Testing.
You can then launch testModel.py. Please note that the parameters of the models are currently set to some default values and will need to be tuned.

## Test the code

The freely available data from the UK Department for Transport can be used to test the code [DfTData](https://www.gov.uk/government/publications/vehicle-emissions-testing-programme-conclusions). Please note that this code available on this repository used the euro-6-renault-megane-1461.xlsx file available in the Euro 6 Vehicle Data as a training and testing file to ensure that the code was running properly. The code might need to be adapted to work with other file formats.

The code needs to be slightly adapted if data from several files is used.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Clémence Le Cornec**

## Citation

Please cite as:

Please cite as:
```bibtex
@article{LeCornec2019,
title = {Modelling of instantaneous emissions from diesel vehicles with a special focus on NOx: Insights from machine learning techniques, [in prep]},
author = {Clémence M. A. Le Cornec and Nick Molden and Marc E. J. Stettler},
year = {2019}
```

## License

This project is licensed under the GNU GPLv3 - see the [GNUGPLv3](https://www.gnu.org/licenses/gpl-3.0.html) for details


