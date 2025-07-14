#!/bin/bash

source env/bin/activate

python main.py --model_name distilbert --batch_size 32 --context_window 512 --data_size 10000 --strategy mean
python main.py --model_name distilbert --batch_size 32 --context_window 512 --data_size 10000 --strategy cls
python main.py --model_name distilbert --batch_size 32 --context_window 512 --data_size 10000 --strategy attention




python main.py --model_name deberta --batch_size 32 --context_window 512 --data_size None --strategy attention
python main.py --model_name deberta --batch_size 32 --context_window 512 --data_size None --strategy mean
python main.py --model_name deberta --batch_size 32 --context_window 512 --data_size None --strategy last