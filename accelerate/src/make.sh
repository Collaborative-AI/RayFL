#!/bin/bash

python make.py --mode full --task_name s2s --run train --num_experiments 1 --resume_mode 1 --init_seed 0 --round 12 --split_round 1 --num_gpu 4
python make.py --mode full --task_name s2s --run test --num_experiments 1 --resume_mode 1 --init_seed 0 --round 12 --split_round 1 --num_gpu 4

python make.py --mode full --task_name clm --run train --num_experiments 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4
python make.py --mode full --task_name clm --run test --num_experiments 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4

python make.py --mode full --task_name sc --run train --num_experiments 1 --resume_mode 1 --init_seed 0 --round 16 --split_round 1 --num_gpu 4
python make.py --mode full --task_name sc --run test --num_experiments 1 --resume_mode 1 --init_seed 0 --round 16 --split_round 1 --num_gpu 4

python make.py --mode full --task_name ic --run train --num_experiments 1 --resume_mode 1 --init_seed 0 --round 16 --split_round 1 --num_gpu 4
python make.py --mode full --task_name ic --run test --num_experiments 1 --resume_mode 1 --init_seed 0 --round 16 --split_round 1 --num_gpu 4

python make.py --mode peft --task_name s2s --run train --num_experiments 1 --resume_mode 1 --init_seed 0 --round 12 --split_round 1 --num_gpu 4
python make.py --mode peft --task_name s2s --run test --num_experiments 1 --resume_mode 1 --init_seed 0 --round 12 --split_round 1 --num_gpu 4

python make.py --mode peft --task_name clm --run train --num_experiments 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 3 --num_gpu 4
python make.py --mode peft --task_name clm --run test --num_experiments 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 3 --num_gpu 4

python make.py --mode peft --task_name sc --run train --num_experiments 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4
python make.py --mode peft --task_name sc --run test --num_experiments 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4

python make.py --mode peft --task_name ic --run train --num_experiments 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4
python make.py --mode peft --task_name ic --run test --num_experiments 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4

python make.py --mode full_dreambooth --task_name t2i --run train --num_experiments 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4
python make.py --mode full_dreambooth --task_name t2i --run generate --num_experiments 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4

python make.py --mode peft_dreambooth --task_name t2i --run train --num_experiments 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4
python make.py --mode peft_dreambooth --task_name t2i --run generate --num_experiments 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4

