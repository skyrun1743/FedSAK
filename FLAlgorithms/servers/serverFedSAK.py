import torch

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_user_data

# Implementation for FedAvg Server
class NuclearNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 计算奇异值分解
        U, S, V = torch.svd(x)
        # 存储用于反向传播的变量
        ctx.save_for_backward(U, V)
        # 计算核范数（奇异值之和）
        return S.sum()

    @staticmethod
    def backward(ctx, grad_output):
        # 从前向传播中获取存储的变量
        U, V = ctx.saved_tensors
        # 计算核范数的梯度
        grad_input = grad_output * torch.matmul(U, V.t())
        return grad_input
    
class FedSAK(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, L_k, num_glob_iters, local_epochs, optimizer, num_users, times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, L_k, num_glob_iters,local_epochs, optimizer, num_users, times)
        
        total_users = len(dataset[0][0]) # 取客户端个数

        for i in range(total_users):
            id, train , test = read_user_data(i, dataset[0], dataset[1])
            user = UserAVG(device, id, train, test, model, batch_size, learning_rate, L_k, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        self.users_model = torch.nn.ModuleList([user.model for user in self.users])
        self.optimizer = torch.optim.SGD(
                self.users_model.parameters(),
                lr=self.learning_rate)
            
        print("Number of users / total users:", num_users, " / " ,total_users)
        print("Finished creating FedSAK server.")

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
        max_acc = 0
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")

            # Evaluate model each interation
            glob_acc = self.evaluate()
            acc_list.append(round(glob_acc*100, 2))
            if glob_acc > max_acc:
                max_acc = glob_acc

            self.selected_users = self.select_users(glob_iter, self.num_users)

            for user in self.selected_users:
                user.train(self.local_epochs)

            self.aggregate_parameters()

        self.save_results()

        print("Max Global Accurancy: ", max_acc)
        print("Accurancy list:", acc_list)

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)  

        self.optimizer.zero_grad()
        W = []
        loss = 0
        
        w1 = torch.cat([user.model.fc1.weight.unsqueeze(2) for user in self.selected_users], dim=2)
        w2 = torch.cat([user.model.fc2.weight.unsqueeze(2) for user in self.selected_users], dim=2)
        W.append(w1)
        W.append(w2)

        for i in range(len(W)):
            Trace_norm_input_to_hidden = self.TensorTraceNorm(W[i])
            loss_w = torch.sum(Trace_norm_input_to_hidden)
            loss += self.L_k * loss_w

        loss.backward()
        self.optimizer.step()
        
    def nuclear_norm(self,x):
        return NuclearNormFunction.apply(x)

    def tensor_unfold(self, A, k):
        tmp_arr = list(range(A.dim()))
        tmp_arr.pop(k)
        A = A.permute(k, *tmp_arr)
        shapeA = A.size()
        A = A.reshape(shapeA[0], -1)
        return A
    
    def TensorTraceNorm(self, X):
        shapeX = X.size()
        dimX = len(shapeX)
        re = []
        for j in range(dimX):
            re.append(self.nuclear_norm(self.tensor_unfold(X, j)))
        return torch.stack(re)

            
