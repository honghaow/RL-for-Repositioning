Author's implementation of the paper: A Reinforcement Learning and Prediction-Based Lookahead Policy for Vehicle Repositioning in Online Ride-Hailing Systems.

## Usage

The paper results can be reproduced by running such as:

```
bash ./run_demo_cars.sh
bash ./run_demo_cars_neighbor.sh
```

## Detailed Explanation of some of the arguments

```
("--online", default="true") #whether using online prediction or not
("--log_dir", default="temp") #log folder
("--obj", default="rate") #objective: completion rate or reward or discounted reward
("--obj_penalty", default=0) #reposition penalty (fuel cost) coefficients added to the objective function
("--neighbor", default="true") #whether using neighbor information in prediction
("--method", default="lstm_cnn") #prediction method: lasso, ridge, cnn, pcr_with_ridge, pcr_with_lasso, lstm_cnn
("--noiselevel", default=0, type=float) #noise level added to arrival rate
("--number_driver", default=300, type=int) #number of total drivers in the system
("--tlength", default=20, type=int) #t length of the T-lookahead policy
("--value_weight", default=0.2, type=float) #coefficients for the value function
("--start_hour", default=13, type=int) #start hour of the simulation
("--stop_hour", default=20, type=int) #stop hour of the simulatonr
("--num_grids", default=20, type=int) #number of grids of the map
```

## Extensions

The simulator is ready to generate to a more complicated version with advanced features. The details of the simulation environment can be found in `envDispatch_v2.py`. Feel free to make changes.

## Bibtex

```
author={Wei, Honghao and Yang, Zixian and Liu, Xin and Qin, Zhiwei (Tony) and Tang, Xiaocheng and Ying, Lei},
journal={IEEE Transactions on Intelligent Transportation Systems},
title={A Reinforcement Learning and Prediction-Based Lookahead Policy for Vehicle Repositioning in Online Ride-Hailing Systems},
year={2023},
volume={},
number={},
pages={1-11},
doi={10.1109/TITS.2023.3312048}}
```
