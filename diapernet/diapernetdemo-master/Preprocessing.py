import numpy as np
from scipy import ndimage as img
from numba import jit
from PIL import Image,ImageEnhance
import sys
import cv2
from six.moves import cPickle as pickle
import torch
import os

class Imaging:
    def __init__(self,section=[350,350]):
        self.index_table = {}
        self.section_size = section
        self._orig_size = [2048,2720]
        self._HF_tblr = [570,628,350,370]
        self._HE_tblr = [0,1198,350,370]
        self._H_Pad_tblr = [250,250,100,80]
        self._V1_tblr = [14,34,1055,815]
        self._V2_tblr = [0,48,1055,815]
        self._V_Pad_tblr = [80,100,250,250]
        self._Dome_tblr = self._HF_tblr #[424,424,320,200]
        #print('Hi')
    
    
    def Segmentation(self,im,imName):
        x,y,_ = im.shape
        scale = float(x)/float(self._orig_size[0])
        crop_tblr = [0,0,0,0]
        pad_tblr = [0,0,0,0]
        if ('_Dome_' in imName):
            crop_tblr = np.array(self._Dome_tblr)*scale
            pad_tblr = crop_tblr+np.array(self._H_Pad_tblr)*scale
        elif ('_HF_' in imName or '_HC_' in imName):
            crop_tblr = np.array(self._HF_tblr)*scale
            pad_tblr = crop_tblr+np.array(self._H_Pad_tblr)*scale
        elif ('_HE_' in imName):
            crop_tblr = np.array(self._HE_tblr)*scale
            pad_tblr = crop_tblr+np.array(self._H_Pad_tblr)*scale
        elif ('_V1_' in imName):
            crop_tblr = np.array(self._V1_tblr)*scale
            pad_tblr = crop_tblr+np.array(self._V_Pad_tblr)*scale
        elif ('_V2_' in imName):
            crop_tblr = np.array(self._V2_tblr)*scale
            pad_tblr = crop_tblr+np.array(self._V_Pad_tblr)*scale
        #print(crop_tblr,pad_tblr)
        # do segmentation
        im_crop = im[int(crop_tblr[0]):int(x-crop_tblr[1]),int(crop_tblr[2]):int(y-crop_tblr[3]),:]
        im_pad = im[int(pad_tblr[0]):int(x-pad_tblr[1]),int(pad_tblr[2]):int(y-pad_tblr[3]),:]
        return im_crop,im_pad

    @jit
    def Grid(self,sheet,overlap=5,top=0,left=0):
        x,y,c = sheet.shape
        sections = []
        for i in range(left,y-overlap,self.section_size[1]-overlap):
            if (i+self.section_size[1])<=y:
                sections.append(sheet[top:,i:i+self.section_size[1],:])
                #means = sections[-1].mean(axis=0).mean(axis=0)
                #sections[-1] = sections[-1]-means
            
        return np.stack(sections)

    #@jit
    def Augmentation(self,im,land=None,rotation=None,flipping=None,light=None,scale=None,resize=None):
        a,b,c = im.shape
        
        augmented = []
        if land!=None:
            return img.rotate(im,land,reshape=True)
        if rotation!=None:
            for r in rotation:
                im_rotate1 = img.rotate(im,r,reshape=True)
                x,y,_ = im_rotate1.shape
                crop_x = int(x//2-a//2)
                crop_y = int(y//2-b//2)
                augmented.append(im_rotate1[crop_x:crop_x+a,crop_y:crop_y+b,:])
                del im_rotate1
        if flipping!=None:
            augmented.append(np.flipud(im))
            augmented.append(np.fliplr(im))
        if light!=None:
            for l in light:
                contrast = im.copy()
                for ch in range(c):
                    contrast[:,:,ch] = np.power(contrast[:,:,ch],l)              
                augmented.append(contrast)
                del contrast
        if scale!=None:
            x,y,_ = im.shape
            for s in scale:
                im2 = cv2.resize(im,(int(s*y), int(s*x)))
                crop_x = int(s*x//2-x//2)
                crop_y = int(s*y//2-y//2)
                augmented.append(np.array(im2)[crop_x:crop_x+x,crop_y:crop_y+y,:])
                del im2
        if resize!=None:
            x,y = im.shape[0:2]
            if resize<=1:
                im2 = cv2.resize(im,(int(resize*y), int(resize*x)))
            else:
                im2 = cv2.resize(im,(int(resize), int(resize)))
            augmented.append(np.array(im2))
        return augmented
        
    
    def LabelGenerator(self,csv_file):
        #generates label from the CSV file and image path
        self.index_table = {'data':0,'labels':[1,2,3,4]}

class DataPreparation:
    def __init__(self,data,label=None,classes=None):
        self._data = data
        self._label = label
        self._class=classes
    
    @property
    def data(self):
        return self._data
    
    @property
    def label(self):
        return self._label

    @property
    def category(self):
        return self._class

    def SaveData(self,fileName,mean=0,stdv=1,k=5):
        self.Shuffle()
        samples = self._label.shape[0]
        section = int(samples/k)
        for s in range(k):
            with open(fileName+"Sec"+str(s+1)+'.pickle','wb') as f:    
                save={'data': self._data[s*section:s*section+section],
                        'label': self._label[s*section:s*section+section],
                        'class': self._class,
                        'norm': [mean,stdv]}
                pickle.dump(save,f)
    
    def LoadData(self,fileName,k=5):
        root = fileName.split('.')[0]
        full_data=[]
        full_label=[]
        for i in range(k):
            with open(root+'Sec'+str(i+1)+'.pickle','rb') as f:
                dataset = pickle.load(f)
                full_data.append(dataset['data'].copy())
                full_label.append(dataset['label'].copy())
                del dataset
        self._data = np.concatenate(full_data,axis=0)
        self._label = np.concatenate(full_label,axis=0)
        del full_data,full_label
    
    #@jit
    def Augment(self,rotation=None,flipping=None,light=None,scale=None):
        im_proc = Imaging()
        n = self._data.shape[0]
        temp_ims = []
        temp_lab = []
        for i in range(n):
            j=len(temp_ims)
            temp_ims = temp_ims + im_proc.Augmentation(self._data[i],rotation=rotation,flipping=flipping,light=light,scale=scale)
            k=len(temp_ims)
            temp_lab = temp_lab + [self._label[i]]*(k-j)
            self.drawProgressBar(i+1,n,desc=' ')
        self._data = np.concatenate([self._data,np.stack(temp_ims)])
        self._label = np.concatenate([self._label,np.stack(temp_lab)])
        self.Shuffle()
    
    @jit
    def Normalization(self,mean=0,stdv=1):
        channels = 0
        if len(self._data[0].shape)>3:
            channels = self._data[0].shape[3] 
        if mean==0:
            if channels==0:
                mean = np.mean(self._data)
                stdv=np.std(self._data)
                self._data = (self._data-mean)/stdv
            else:
                mean=np.zeros([channels])
                stdv = np.zeros([channels])
                for ch in range(channels):
                    mean[ch] = np.mean(self._data[:,:,:,ch])
                    stdv[ch] = np.std(self._data[:,:,:,ch])
                    self._data[:,:,:,ch] = (self._data[:,:,:,ch]-mean[ch])/stdv[ch]
        else:
            if channels==0:
                self._data = (self._data-mean)/stdv
            else:
                for ch in range(channels):
                    self._data[:,:,:,ch] = (self._data[:,:,:,ch]-mean)/stdv

        return mean,stdv

    
    def Shuffle(self):
        p = np.random.permutation(self._label.shape[0])
 #       print(p)
        self._data = self._data[p]
        self._label = self._label[p]
        #samples = self._label.shape[0]
        #for i in range(samples):
        #    rand = np.random.randint(samples)
        #    #print(rand)
        #    self._data[i],self._data[rand] = self._data[rand],self._data[i]
        #    #print(rand)
        #    self._label[i],self._label[rand] = self._label[rand],self._label[i]

    def NumpytoTorch(self,regression=True):
        n,a,b,c = self._data.shape
        temp_data = np.zeros([n,c,a,b])
        for channel in range(c):
            temp_data[:,channel,:,:] = self._data[:,:,:,channel]

        self._data = torch.from_numpy(temp_data).float()
        del temp_data
        if regression:
            self._label = torch.from_numpy(self._label).float()
        else:
            self._label = torch.from_numpy(self._label).long()

    def Batch(self,b_size,n):
        for i in range(0,n,b_size):
            yield self._data[i:i+b_size],self._label[i:i+b_size]

    def drawProgressBar(self,prog,barLen,desc='None'):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[%-50s] %.2f%% \033[92m (%s) \033[93m %d of %d \033[0m   " % ('='*int(prog/barLen*50), 100.0*prog/barLen,desc,prog,barLen))
        sys.stdout.flush()

    def LabelNormalize(self,infimum=0,suprimum=1):
        if len(self._label.shape)==1:
            self._label = self._label.reshape([self._label.shape[0],1])
        d=self._label.shape[1]
        for j in range(d):
            mini = np.min(self._label[:,j])
            maxi = np.max(self._label[:,j])
            self._label[:,j] = np.array(list(map(lambda z: (z-mini)/(maxi-mini)*(suprimum-infimum)+infimum,self._label[:,j])))
 #       self._label=self._label.squeeze()

def Torch2Numpy(args):
    n=len(args)
    new_args = []
    for i in range(n):
        n,c,a,b = args[i].shape
        temp = np.zeros([n,a,b,c])
        for cc in range(c):
            temp[:,:,:,cc] = args[i][:,cc,:,:].detach().numpy()
        new_args.append(temp)
        del temp
    return new_args



























