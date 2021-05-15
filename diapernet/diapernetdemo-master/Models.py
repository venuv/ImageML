import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from collections import namedtuple
import gc
from sklearn.svm import SVR
import sklearn.linear_model as lm

from Preprocessing import DataPreparation

class Trany:
    def __init__(self,feature_input,feature_hidden,n_class,n_layer,prob,**kwargs):
        super().__init__(**kwargs)
        self.f_input = feature_input
        self.f_hidden = feature_hidden
        self.n_class = n_class
        self.n_layer = n_layer
        self.prob = prob
        self.net = None

    def Features(self):
        pass

    def Classifier(self,regression=False):
        layers = [None]*self.n_layer*3
        dim1 = self.f_input
        dim2 = self.f_hidden
        for i in range(0,self.n_layer*3,3):
            layers[i] = nn.Linear(dim1,dim2)
            layers[i+1] = nn.Dropout(0.3)
            layers[i+2] = nn.ReLU()
            dim1 = int(dim2)
            dim2 = int(dim2/2)
        layers.append(nn.Linear(dim1,self.n_class))
        if not regression:
            if self.prob:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Softmax())
        return nn.Sequential(*layers)

class VGG(Trany,nn.Module):
    def __init__(self,vgg_type,last_layer,last_trained,**kwargs):
        super().__init__(**kwargs)
        
        exec("vggNet = models."+vgg_type+"(pretrained='imagenet')",globals())
        
        self.feature = nn.Sequential(*list(vggNet.features.children())[:last_layer])
        index = 0
        for i,p in self.feature.named_parameters():
            if index<=last_trained:
                p.requires_grad=False
            index = index+1
        #self.collector = nn.AvgPool2d(kernel_size=5) # 7 for Kylberg
        self.classifier = self.Classifier()
    
    def forward(self,x):
        x = self.feature(x)
        #x = self.collector(x)
        x = nn.AvgPool2d(kernel_size=(x.size(2),x.size(3)))(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
    def __str__(self):
        out = ""
        for node in self.children():
            for layer in node.children():
                out+= str(layer)
                out+= "\nTrainable\n"
                for i,p in layer.named_parameters():
                    out= out + str(i) +"-->"+ str(p.requires_grad)+"\n"
        return out

class DiaperNet(Trany,nn.Module):
    def __init__(self,texture_model,**kwargs):
        super().__init__(**kwargs)
        texNet = torch.load(texture_model)
        featureList = list(texNet.feature)
        del texNet
        gc.collect()
        index = 0
        last_trained=15
        featureList.append(nn.Conv2d(512,1024,5,stride=2,padding=2))
        featureList.append(nn.ReLU(inplace=True))
        featureList.append(nn.AvgPool2d(2))
        #featureList.append(nn.Conv2d(1024,1024,5,stride=1,padding=2))
        #featureList.append(nn.ReLU(inplace=True))
        #featureList.append(nn.MaxPool2d(2))
        self.feature = nn.Sequential(*featureList)

        for i,p in self.feature.named_parameters():
            if index<=last_trained:
                p.requires_grad=False
            index = index+1

        self.classifier = self.Classifier(regression=True)

    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

class AmirNet(nn.Module):
    def __init__(self,fc_layers,cnn_patches,cnn_layers,n_class,in_size,pools):
        super(AmirNet,self).__init__()
        cnn = []
        
        for i in range(1,len(cnn_layers)):
            cnn.append(nn.Conv2d(cnn_layers[i-1],cnn_layers[i],cnn_patches[i-1],stride=1,padding=int(cnn_patches[i-1]/2)))
            cnn.append(nn.ReLU())
            cnn.append(nn.MaxPool2d(pools[i-1]))
            in_size = int(in_size/pools[i-1])
        self.features = nn.Sequential(*cnn)

        num_features = in_size*in_size*cnn_layers[-1]
        fc_layers.insert(0,num_features)
        fc = []
        for j in range(1,len(fc_layers)):
            fc.append(nn.Linear(fc_layers[j-1],fc_layers[j]))
            fc.append(nn.Dropout(0.3))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(fc_layers[-1],n_class))
        fc.append(nn.Softmax())
        self.classifier = nn.Sequential(*fc)

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x



		

class Build:
    def __init__(self,net,status,regression=False):
        self.device = torch.device('cuda:'+str(status.cuda) if torch.cuda.is_available() else 'cpu')
        self._net=net.to(self.device)
        self.optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, net.parameters())),lr=status.lr,weight_decay=status.weight_decay,betas=(status.momentum,status.momentum+0.09))
        if not regression:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

    def StepTrain(self,tempData,tempLabel):
        self.optimizer.zero_grad()
        outputs=self._net(tempData.to(self.device))

        loss = self.criterion(outputs,tempLabel.to(self.device))
        loss.backward()
        self.optimizer.step()
        del tempData,tempLabel,outputs
        gc.collect()
        return loss.to(torch.device('cpu'))
    def Validate(self,validData,validLabel=None,nClass=28,portion=100):
        n = validData.size(0)
        outs = []
        ress = []
        accs = []
        for por in range(int(n/portion)):
            vData = validData[por*portion:por*portion+portion].to(self.device)
            vLabel = validLabel[por*portion:por*portion+portion].to(self.device)
            #print(vData.shape,vLabel.shape,validData.shape,validLabel.shape)
            outputs = self._net(vData)
            acc = 0
            if (type(validLabel)==type(None)):
                return torch.max(outputs,1).item()

            if nClass==1:
                acc = torch.sqrt(torch.sum(torch.pow(outputs-vLabel,2))).item()
            else:
                acc = (torch.max(outputs,1)[1]==vLabel).sum().item()/float(vLabel.size(0))*100.0
            del vData,vLabel
            outs.append(outputs.to(torch.device('cpu')).detach().numpy())
            ress.append(torch.max(outputs,1)[1].to(torch.device('cpu')).numpy())
            accs.append(acc)
            del acc,outputs
            gc.collect()
        return np.concatenate(outs),np.concatenate(ress),sum(accs)/len(accs)

class ShallowNet(nn.Module):
    def __init__(self,hiddens):
        self._hiddens = hiddens
        super(ShallowNet,self).__init__()
        n_layer = len(hiddens)
        fc = []
        for i in range(n_layer-2):
            fc.append(nn.Linear(self._hiddens[i],self._hiddens[i+1]))
            fc.append(nn.Sigmoid())
        fc.append(nn.Linear(self._hiddens[-2],self._hiddens[-1]))
        self._net = nn.Sequential(*fc)

    def forward(self,x):
        x = self._net(x)
        return x

class Stat:
    def __init__(self,data,label,train_scale = 0.75):
        self._data = data
        self._label = label
        self._dim = data.shape[1]
        self._n = data.shape[0]
        self._model = None
        self.train_size = int(self._n*train_scale)
    
    def StatTrainEval(self):
        self._model.fit(self._data[:self.train_size],self._label[:self.train_size])
        out = self._model.predict(self._data[self.train_size:,:])
        R2 = self._model.score(self._data[self.train_size:],self._label[self.train_size:])
        return out,R2

    def SVM(self,kernel='poly'):
        self._model = SVR(kernel=kernel)
        return self.StatTrainEval()


    def LogReg(self):
        self._model = lm.LogisticRegression()
        return self.StatTrainEval()

    def NN(self,layer_neuron,epoch=50000,lr=0.001,wd=0.000001,mom=0.6,lrd=0.00001):
        layers = [self._dim]+layer_neuron
        self._model = ShallowNet(layers)
        dp = DataPreparation(self._data,self._label)
        optimizer = torch.optim.Adam(self._model.parameters(),lr=lr,weight_decay=wd,betas=(mom,mom+0.09))
        for i in range(epoch):
            batch = dp.Batch(1,self.train_size)
            for j in range(int(self.train_size/1)):
                tempData,tempLabel = next(batch)
                criterion = torch.nn.MSELoss()
                optimizer.zero_grad()
                output = self._model(torch.from_numpy(tempData).float())
                loss = criterion(output,torch.from_numpy(tempLabel).float())
                loss.backward()
                optimizer.step()
            dp.drawProgressBar(j+1,self.train_size/1,'Loss: '+str(loss.item()))
            out = self._model(torch.from_numpy(self._data[self.train_size:]).float())
            acc = torch.sqrt(torch.sum(torch.pow(out-torch.from_numpy(self._label[self.train_size:]).float(),2))).item()
            #print('\n Test Loss: {}'.format(acc))
            dp.drawProgressBar(i+1,epoch,'Loss: '+str(acc))
        return out,acc









