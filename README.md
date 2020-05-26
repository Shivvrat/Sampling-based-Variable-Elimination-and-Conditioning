# Sampling-based-Variable-Elimination-and-Conditioning



## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project
In this project we will implement the Sampling-based Variable Elimination and Conditioning algorithm for inference on a given probabilistic graphical model.
The algorithm is as follows :-
![Sampling-based Variable Elimination and Conditioning algorithm](https://github.com/Shivvrat/Sampling-based-Variable-Elimination-and-Conditioning/blob/master/algo.jpg)



### Built With

* [Python 3.7](https://www.python.org/downloads/release/python-370/)


## Getting Started

Lets see how to run this program on a local machine.

### Prerequisites

You will need the following modules 
1 from numpy import product , array , unique , zeros , divide , nan_to_num
2 import sys 
3 from copy import deepcopy
4 from itertools import product as iter_product 
5 from more_itertools import locate
6 from numpy import prod , product , random , put , sum , divide , array , log10 
7 import argparse 
8 import sys
9 import warnings 
10 from copy import deepcopy
11 from random import choice , randint , sample 
12 from numpy import array , setdiff1d , concatenate , reshape , append , put , divide , nan_to_num , product , zeros
13 from numpy.random import rand

### Installation

1. Clone the repo
```sh
git clone https://github.com/Shivvrat/Learning-Algorithm-for-Bayesian-Networks.git
```
Use the main.py to run all three algorithm.


<!-- USAGE EXAMPLES -->
## Usage
Please enter the following command line argument :-
```
python main.py --uai_file <uai_file_directory_and_name > --task_id <task ID> --training_data <training_file_directory_and_name > --test_data <test_file_directory_and_name >
```
Please use the following command line parameters for the main.py file :-
* ```<uai_file_directory_and_name>``` 
Please provide the file that will be used as the .uai file.
* ```<training_file_directory_and_name>``` 
Please provide the file that will be used as the training data.
* ```<training_file_directory_and_name>```
Please provide the file that will be used as the test data.
* ```<task_id>``` 
Please provide the algorithm you want to use :
    1. FOD learn
    2. POD EM Learn     
    3. Mixture Random Bayes


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - Shivvrat Arya[@ShivvratA](https://twitter.com/ShivvratA) - shivvratvarya@gmail.com

Project Link: [https://github.com/Shivvrat/Learning-Algorithm-for-Bayesian-Networks.git](https://github.com/Shivvrat/Learning-Algorithm-for-Bayesian-Networks.git)
