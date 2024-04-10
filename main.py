#!/usr/bin/env python
import numpy as np
import random
import torch

from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverFedSAK import FedSAK
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *

from utils.model_utils import read_data
from utils.options import args_parser

def init_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

# Create an experiment with your api key:
def main(dataset, algorithm, model, batch_size, learning_rate, L_k, num_glob_iters,
         local_epochs, optimizer, numusers, gpu, times):
    
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    data = read_data(dataset) , dataset

    for i in range(times):
        print("---------------Running time:------------", i)
        # Generate model
        if(model == "mclr"):
            if(dataset == "human_activity"):
                model = Mclr_Logistic(561,6).to(device), model
            elif(dataset == "vehicle_sensor"):
                model = Mclr_Logistic(100,2).to(device), model
            else:
                model = Mclr_Logistic().to(device), model
        elif(model == "dnn"):
            if(dataset == "human_activity"):
                model = DNN(561,100,12).to(device), model
            elif(dataset == "vehicle_sensor"):
                model = DNN(100,20,2).to(device), model
            else:
                model = DNN().to(device), model
        elif(model == "cnn"):
            if(dataset == "cifar10"):
                model = FedAvgCNN(in_features=3, num_classes=10, dim=1600).to(device), model
            elif(dataset == "cifar100"):
                model = FedAvgCNN(in_features=3, num_classes=100, dim=1600).to(device), model
            elif(dataset == "pacs"):
                model = FedAvgCNN(in_features=3, num_classes=7, dim=1600).to(device), model
            else:
                model = FedAvgCNN().to(device), model
                
        # select algorithm        
        if(algorithm == "FedSAK"):
            server = FedSAK(device, data, algorithm, model, batch_size, learning_rate, L_k, num_glob_iters, local_epochs, optimizer, numusers, i)
        else:
            server = FedAvg(device, data, algorithm, model, batch_size, learning_rate, L_k, num_glob_iters, local_epochs, optimizer, numusers, i)

        server.train()
        server.test()

if __name__ == "__main__":
    init_seed(42)

    args = args_parser()
    
    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Lambda       : {}".format(args.L_k))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    main(
        dataset = args.dataset,
        algorithm = args.algorithm,
        model = args.model,
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        L_k = args.L_k,
        num_glob_iters = args.num_global_iters,
        local_epochs = args.local_epochs,
        optimizer = args.optimizer,
        numusers = args.subusers,
        gpu = args.gpu,
        times = args.times
        )


