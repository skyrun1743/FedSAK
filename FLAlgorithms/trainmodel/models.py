import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Mclr_CrossEntropy(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(Mclr_CrossEntropy, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

class DNN(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias']
                            ]
    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

#################################
##### Neural Network model #####
#################################

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, dim_out)
        self.softmax = nn.Softmax(dim=1)
        
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_hidden3.weight', 'layer_hidden3.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_hidden1(x)
        x = self.relu(x)

        x = self.layer_hidden2(x)
        x = self.relu(x)

        x = self.layer_hidden3(x)
        x = self.relu(x)

        x = self.layer_out(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, num_classes)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]
                            
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super(FedAvgCNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.act = nn.ReLU()

        self.FC = nn.Sequential(
            self.fc1,
            self.act,
            self.fc2
            )

    def forward(self, x): 
        out = self.conv1(x) 
        out = self.conv2(out) 
        out = torch.flatten(out, 1)
        out = self.FC(out)
        
        return F.log_softmax(out, dim=1)

class CNN_encoder(nn.Module):

    def __init__(self, dim_out=512):
        super(CNN_encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 7)
        self.act = nn.ReLU()
        self.fc = nn.Linear(128*23*23, dim_out)
        
        self.encoder = nn.Sequential(
            self.conv1,
            self.act,
            self.conv2,
            self.act,
            self.conv3,
            self.act,
            self.conv4
        )
    
    def forward(self, x):
        x = self.encoder(x)
        y = self.fc(x.view(-1, 128*23*23))
        return y


class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.weight_keys = [['fc1.weight', 'fc1.bias']]

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class heteropacs_sharelayer(nn.Module):
    def __init__(self, cin: int, hidden_size: int = 512):
        super(heteropacs_sharelayer, self).__init__()

        self.share_layer = nn.Sequential(
            nn.Linear(cin, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.share_layer(x)
    

class heteropacs(nn.Module):
    def __init__(self, num_classes: int, group: str, hidden_size: int = 512):
        super(heteropacs, self).__init__()

        if (group == "art"):
            self.hetero_feature = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=0, stride=1, bias=True),  
                nn.ReLU(inplace=True), 
                nn.MaxPool2d(kernel_size=(2, 2)), 
                nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
        elif (group == "cartoon"):
            self.hetero_feature = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=0, stride=1, bias=True),  
                nn.ReLU(inplace=True), 
                nn.MaxPool2d(kernel_size=(2, 2)), 
                nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
        elif (group == "photo"):
            self.hetero_feature = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=0, stride=1, bias=True),  
                nn.ReLU(inplace=True), 
                nn.MaxPool2d(kernel_size=(2, 2)), 
                nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
        elif (group == "sketch"):
            self.hetero_feature = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=0, stride=1, bias=True), 
                nn.ReLU(inplace=True), 
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
        
        self.share_layer = heteropacs_sharelayer(1600, hidden_size)
        self.cls = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, dataset): 
        if(dataset == "art"):
            fea = self.hetero_feature(x)
        elif(dataset == "cartoon"):
            fea = self.hetero_feature(x)
        elif(dataset == "photo"):
            fea = self.hetero_feature(x)
        elif(dataset == "sketch"):
            fea = self.hetero_feature(x)

        fea_p = torch.flatten(fea, 1)
        fea = self.share_layer(fea_p)
        out = self.cls(fea)
        return F.log_softmax(out, dim=1)
    
class heteroface_sharelayer(nn.Module):
    def __init__(self, cin: int, hidden_size: int = 512):
        super(heteroface_sharelayer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(cin,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1600, 512), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)
    
class heteroface(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 512):
        super(heteroface, self).__init__()

        self.share_layer = heteroface_sharelayer(3, hidden_size)
        self.cls = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.share_layer(x)
        logits = self.cls(x)
        return F.log_softmax(logits, dim=1), x
