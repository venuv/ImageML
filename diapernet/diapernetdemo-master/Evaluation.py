import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')

import matplotlib.image as img
import torch
import torchvision
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import math
import gc
import os
import torch.nn as nn
import seaborn as sb
import pandas as pd
import Preprocessing as pp
import skimage.transform as im_tr
#import scipy.misc as im_tr

class Test:
    def __init__(self,net_name,gpu_cpu='cpu',cuda=3):
        self._device = torch.device('cuda:'+str(cuda) if gpu_cpu=='gpu' else 'cpu')
        self._net = torch.load(net_name).to(self._device)
        self._net = self._net.eval()
    def Test(self,image_set): #recieves a set of images with shape n*h*w*channels
        n,a,b,c = image_set.shape
        image_set2 = np.zeros([n,c,a,b])
        for ch in range(c):
            image_set2[:,ch,:,:] = image_set[:,:,:,ch]
        tens_image=torch.Tensor(image_set2)
        tens_image = tens_image.to(self._device)
        outputs = self._net(tens_image).to(torch.device('cpu'))
        del tens_image
        gc.collect()
        return outputs.detach().numpy(),torch.max(outputs,1)[1]

    def DetailOut(self,tens):
        outs = []
        tens2 = tens.clone()
        for l in self._net.feature:
            tens2 = l(tens2)
            outs.append(tens2.clone().detach().numpy())
        del tens2
        gc.collect()
        return outs

class Visualization(Test):
    def __init__(self,net_name,gpu_cpu='cpu',cuda=1):
        Test.__init__(self,net_name,gpu_cpu,cuda)    
        self._net = self._net.to(torch.device('cpu'))
        print(self._net)
    def Filter(self,res='test.jpg'):
        gabors = np.array(self._net.feature[0].weight) #64*channels*3*3
        s=gabors.shape
        fig, axs = plt.subplots(int(math.sqrt(s[0])),int(math.sqrt(s[0])))
        cnt=0
        for i in range(int(math.sqrt(s[0]))):
            for j in range(int(math.sqrt(s[0]))):
                axs[i,j].imshow(gabors[cnt].T)
                axs[i,j].axis('off')
                cnt+=1
        fig.savefig(res)
        plt.close()
    def FeatureMaps(self,py_image,layers,folder='TextureResults'):
        a,b,c = py_image.shape
        py_tens = torch.Tensor(py_image)
        tens = torch.Tensor(1,c,a,b)
        plt.imsave(folder+'/image.jpg',py_image)
        for i in range(c):
            tens[0,c-i-1,:,:] = py_tens[:,:,i]
        maps = self.DetailOut(tens)
        for l in layers:
            map_tmp = maps[l].squeeze()
            fig, axs = plt.subplots(4,4)
            ind=0
            for x1 in range(4):
                for x2 in range(4):
                    axs[x1,x2].imshow(map_tmp[ind])
                    ind+=1
                    axs[x1,x2].axis('off')
            fig.savefig(folder+'/Map_'+str(l)+'.jpg')
            plt.close()
    def Descriptor(self,py_image,inhibition=False):
        a,b,c = py_image.shape
        py_tens = torch.Tensor(py_image)
        tens = torch.Tensor(1,c,a,b)
        for i in range(c):
            tens[0,c-i-1,:,:] = py_tens[:,:,i]
        res = self._net(tens)
        maps = self.DetailOut(tens)
        dims = maps[-1].shape
        descript_net = NegNet(self._net,inhibition=inhibition)
        return descript_net(torch.Tensor([1.0]),maps[1],dims) #res can be replaced by 1


class Result:
    def LearningTrack(self,res_file,n_class=11):
        acc0 = 100.0/n_class
        if n_class==1:
            acc0=0.25
        #with open(res_file,'rb') as f:
        #    res = pickle.load(f)
        res = torch.load(res_file)
        acc = res['acc']
        loss = res['loss']
        model = str(res['args'])
        #if 'model' in args.keys():
        #    model = args.model
        del res
        acc.insert(0,acc0)
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(acc)
        plt.title('Accuracy: '+str(model))
        plt.xticks(np.arange(0, len(acc), step=1))
        plt.subplot(2,1,2)
        plt.plot(loss,color='red')
        plt.title('Loss: '+str(model))
        plt.tight_layout()
        plt.savefig(res_file.split('.')[0]+'_Track.jpg')
        plt.close()
    def Compare(self,out,res):
        n = out.shape[0]
        X = np.linspace(1,n,n)
        plt.figure(figsize=(15,10))
        plt.plot(X,out,'b',label='Output', linewidth=2)
        plt.plot(X,res,'r',label='Target', linewidth=2)
        plt.plot(X,(out-res),'--k',label='Error')
        plt.legend()
        plt.savefig('test.jpg')
        plt.show()
        #plt.close()

        plt.figure(figsize=(15,10))
        ind_sort = res.argsort()
        plt.plot(X,out[ind_sort],'bs',label='Output')
        plt.plot(X,res[ind_sort],'ro',label='Target')
        plt.legend()
        plt.savefig('test2.jpg')
        
        plt.figure()
        scat = sb.regplot(out,res)
        fig = scat.get_figure()
        fig.savefig('test3.jpg')

        plt.figure()
        df = pd.DataFrame()
        df['ConsumerRate']=res
        df['Prediction']=out
        df = df.groupby('ConsumerRate').mean()
        df.reset_index(inplace=True)
        scat2 = sb.regplot(np.array(df['Prediction']),np.array(df['ConsumerRate']))
        fig2 = scat2.get_figure()
        fig2.savefig('test4.jpg')

class Inhibition(torch.nn.Module):
    def __init__(self):
        super(Inhibition,self).__init__()
    def forward(self,x):
        x=x*(-1.0)
        return torch.nn.ReLU()(x)

class NegNet(torch.nn.Module):
    def __init__(self,for_net,last_scale=2,inhibition=False):
        super(NegNet,self).__init__()
        #for_net = net #torch.load(net)
        n_feature = len(for_net.feature)
        n_classifier = len(for_net.classifier)
        my_classifier = []
        self.n_class = 1
        flag=True
        for lc in range(n_classifier-1,-1,-1):
            layer = for_net.classifier[lc]
            if layer.__module__=='torch.nn.modules.linear':
                if flag:
                    self.n_class = layer.out_features
                    flag=False
                temp = torch.nn.Linear(in_features = layer.out_features,out_features = layer.in_features,bias=True)
                temp.weight = torch.nn.Parameter(layer.weight.transpose(0,1))
                #if lc>0:
                #    temp.bias = torch.nn.Parameter(for_net.classifier[lc-1].bias)
                my_classifier.append(temp)
                del temp
            elif layer.__module__=='torch.nn.modules.activation' and inhibition:
                my_classifier.append(Inhibition())
            elif layer.__module__=='torch.nn.modules.activation':
                my_classifier.append(layer)
            
        self.classifier = torch.nn.Sequential(*my_classifier)
        self.expander = torch.nn.UpsamplingNearest2d(scale_factor=last_scale)
        my_feature=[]
        for lf in range(n_feature-1,-1,-1):
            layer = for_net.feature[lf]
            if layer.__module__=='torch.nn.modules.pooling':
                #print(layer.stride,type(layer.stride))
                my_feature.append(torch.nn.UpsamplingNearest2d(scale_factor=layer.stride))
            elif layer.__module__=='torch.nn.modules.activation':
                if inhibition:
                    my_feature.append(Inhibition())
                else:
                    my_feature.append(layer)
            else:
                if layer.stride[0]>1:
                    #print(layer.padding)
                    my_feature.append(torch.nn.UpsamplingNearest2d(scale_factor=layer.stride[0]))
                #print(layer.padding)
                temp = torch.nn.Conv2d(layer.out_channels,layer.in_channels,kernel_size=layer.kernel_size,padding=layer.padding)
                temp.weight = torch.nn.Parameter(layer.weight.transpose(0,1))
                my_feature.append(temp)
                del temp
        self.feature = nn.Sequential(*my_feature[:-1])
        last_active = torch.nn.ReLU()
        if inhibition:
            last_active = Inhibition()
        self.map = torch.nn.Sequential(*[my_feature[-1],last_active])

    def forward(self,x,net_map,dims):
        #y=torch.zeros(1,self.n_class).float()
        y=x.clone()
        y = self.classifier(y)
        y = y.squeeze()
        
        y = y.reshape(dims) #1*512*7*7 for example
        f_map = self.feature(y)
        
        net_map_con = np.zeros([net_map.shape[0],net_map.shape[1],f_map.shape[2],f_map.shape[3]])
        for c1 in range(net_map.shape[0]):
            for c2 in range(net_map.shape[1]):
                net_map_con[c1,c2] = im_tr.resize(np.float64(net_map[c1,c2,:,:]),(f_map.shape[2],f_map.shape[3]))
        
        # This is for NOW. Will be changed to use CV2 and return to Tensor
        #net_map = net_map[:,:,0:f_map.shape[2],0:f_map.shape[3]]

        raw_map = self.map(f_map)
        #print(f_map.shape,net_map.shape)
        cov_map = f_map*torch.from_numpy(net_map_con).float()
        heat_map = (self.map(cov_map))
        return pp.Torch2Numpy([f_map,raw_map,cov_map,heat_map])









                







        





        

