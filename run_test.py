import argparse
import os
import numpy as np
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--script_name", default='submit.sh')
parser.add_argument("--logs_folder", default='./logs')
parser.add_argument("--job_name", default='')
parser.add_argument("--env", nargs='+', default=["InvertedPendulum-v2"])

args = parser.parse_args()

# If submit script does not exist, create it
if not os.path.isfile(args.script_name):
    with open(args.script_name, 'w') as file:
        file.write(f'''#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=10000

./staskfarm ${{1}}\n''')


for env in args.env:
    for model_type in ['model', 'friction']:#, "model_friction", "friction_noise", "model_noise"]:

        path = f'{env}/{model_type}'
        if not os.path.isdir(f'{args.logs_folder}/{path}'):
            os.makedirs(f'{args.logs_folder}/{path}')
        print(path)
        command = f'python3.6 test.py --env {env} --eval_type {model_type} --optimizer adam --action_noise 0'
        experiment_path = f'{args.logs_folder}/{path}/command.txt'

        with open(experiment_path, 'w') as file:
            file.write(f'{command}\n')

        print(command)

        if not args.job_name:
            job_name = path
        else:
            job_name = args.job_name

        os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')
