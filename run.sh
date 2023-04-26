#!/bin/bash

current_time=$(date "+%Y-%m-%d-%H-%M-%S")

nohup python main.py --gpuID 0,1,2 > "logDIR/"$current_time".log" &