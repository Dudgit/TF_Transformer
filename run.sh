#!/bin/bash

current_time=$(date "+%Y-%m-%d-%H-%M-%S")

nohup python main.py --gpuID 1 > "logDIR/"$current_time".log" &