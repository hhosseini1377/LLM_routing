#!/bin/bash

source env/bin/activate

python main.py --model_name distilbert --batch_size 32 --context_window 512 --strategy mean
python main.py --model_name distilbert --batch_size 32 --context_window 512 --strategy max
python main.py --model_name distilbert --batch_size 32 --context_window 512 --strategy last