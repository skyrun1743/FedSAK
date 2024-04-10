import torch

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_user_data

class FedAvg(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, L_k, num_glob_iters, local_epochs, optimizer, num_users, times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, L_k, num_glob_iters,local_epochs, optimizer, num_users, times)
        
        total_users = len(dataset[0][0])
  
        for i in range(total_users):
            id, train , test = read_user_data(i, dataset[0], dataset[1])
            user = UserAVG(device, id, train, test, model, batch_size, learning_rate, L_k, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:", num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        acc_list=list()
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")

            self.send_parameters()

            # Evaluate model each interation
            glob_acc = self.evaluate()
            acc_list.append(round(glob_acc*100, 2))

            self.selected_users = self.select_users(glob_iter,self.num_users)
            
            for user in self.selected_users:
                user.train(self.local_epochs)

            self.aggregate_parameters()
            
        print("Accurancy list:", acc_list)    

        self.save_results()