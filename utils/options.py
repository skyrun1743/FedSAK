#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="human_activity")
    parser.add_argument("--model", type=str, default="dnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--L_k", type=float, default=0.2, help="Regularization term lambda")
    parser.add_argument("--num_global_iters", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default = 5)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="FedSAK",choices=["FedSAK", "FedAvg"]) 
    parser.add_argument("--subusers", type = float, default = 1.0, help="Sampling probability per client")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments")
    parser.add_argument("--times", type=int, default=5, help="Running times")

    args = parser.parse_args()

    return args
