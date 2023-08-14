#!/bin/bash

python make.py --mode base --run train --num_experiment 4 --round 8
python make.py --mode base --run test --num_experiment 4 --round 8

python make.py --mode fl --run train --num_experiment 4 --round 16 --split_round 2
python make.py --mode fl --run test --num_experiment 4 --round 16 --split_round 2