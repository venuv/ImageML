#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:41:33 2019

@author: amir
"""

from tkinter import *
import tkinter.messagebox as ms
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as filedialog
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
#import torch
from Evaluation import *
#import cv2
from Preprocessing import Imaging
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread

class DiaperNetGUI:
    
    def __init__(self):
        self.__net='DiaperModels/DiaperNet_Net.net'
        self.scale=0.6
        self.root= Tk()
        self.photo_tk = None
        #self.im = None
        self.root.title('Diaper Net Demo')
        # ******* Menu ******** #
        self.menu = Menu(self.root,bg='blue')
        self.root.config(menu=self.menu)
        self.file_menu = Menu(self.menu)
        self.menu.add_cascade(label='File',menu=self.file_menu)
        self.file_menu.add_command(label='New',command=self.new_project)
        self.file_menu.add_command(label='Save',command=self.save_results)
        self.file_menu.add_command(label='Exit',command=self.exit1)
        self.proc_menu = Menu(self.menu)
        self.menu.add_cascade(label='Process',menu=self.proc_menu)
        self.proc_menu.add_command(label='Run',command=self.run_net)
        self.help_menu = Menu(self.menu)
        self.menu.add_cascade(label='Help',menu=self.help_menu)
        self.help_menu.add_command(label='How to Use',command=self.helpme)
        self.help_menu.add_command(label='About Us',command=self.about)
        
        photo_new = Image.open("new.png").resize((40,40), Image.ANTIALIAS)
        photo_save = Image.open("save.png").resize((40,40), Image.ANTIALIAS)
        photo_run = Image.open("run.png").resize((40,40), Image.ANTIALIAS)
        photo_help = Image.open("help.png").resize((40,40), Image.ANTIALIAS)
        
        photo_new = ImageTk.PhotoImage(photo_new)
        photo_save = ImageTk.PhotoImage(photo_save)
        photo_run = ImageTk.PhotoImage(photo_run)
        photo_help = ImageTk.PhotoImage(photo_help)
        
        toolbar = Frame(self.root,bg='blue')
        
        self.new_but = Button(toolbar,width=40,height=40,image=photo_new,command=self.new_project)
        self.save_but = Button(toolbar,text="",width=40,height=40,image=photo_save,command=self.save_results)
        self.run_but = Button(toolbar,text="",width=40,height=40,image=photo_run,command=self.start_run_net)
        self.help_but = Button(toolbar,text="",width=40,height=40,image=photo_help,command=self.helpme)
        self.result = Label(toolbar,text="",bg='blue',fg='red',font=20)
        
        self.new_but.pack(side=LEFT,padx=2,pady=2)
        self.save_but.pack(side=LEFT,padx=2,pady=2)
        self.run_but.pack(side=LEFT,padx=2,pady=2)
        self.help_but.pack(side=LEFT,padx=2,pady=2)
        self.result.pack(side=LEFT,padx=10,pady=2)
        toolbar.pack(side=TOP,fill=X)
        
        brows = Frame(self.root)
        self.file_name = "  "
        self.sheet_var = IntVar()
        rb1 = Radiobutton(brows, text="Whole Sheet", variable=self.sheet_var, value=1,command=self.sel)
        rb1.pack(side=LEFT)
        rb2 = Radiobutton(brows, text="Pad Sheet", variable=self.sheet_var, value=2,command=self.sel)
        rb2.pack(side=LEFT)
        brows_but = Button(brows,fg="red",text="Select a Diaper Top Sheet",command=self.file_dialog)
        brows_but.pack(side=LEFT)
        brows.pack(side=TOP,anchor=W)
        
        self.labelphoto = Label(self.root)
        self.labelphoto.pack()
        
        
        fr1 = Frame(self.root)
        self.file_label = Label(fr1,text=self.file_name)
        self.file_label.pack()
        fr1.pack(fill=X)
        
        self.frame = Frame(self.root,width=450,height=5,bg='blue')
        self.frame.pack(fill=X)
        
        self.selection = []
        
        self.labstat = Label(self.root,text='Info: tavanaei.a@pg.com', bd=1,relief=SUNKEN,anchor=W)
        self.labstat.pack(side=BOTTOM, fill=X)
        
        self.canvas=None
        
        self.root.mainloop()
        
    def file_dialog(self):
        self.file_name = filedialog.askopenfilename()
        head, tail = os.path.split(self.file_name)
        self.file_label.config(text='Image Name: '+tail)
        resize_dim = (400,180) if self.scale==0.6 else (400,90)
        self.photo_tk = ImageTk.PhotoImage(Image.open(self.file_name).resize(resize_dim,Image.ANTIALIAS))
    ##self.photo_tk = PhotoImage(file=self.file_name)
        self.labelphoto.config(image=self.photo_tk)
    
    def sel(self):
        if self.sheet_var.get()==1:
            self.__net='DiaperModels/DiaperNet_Net.net'
            self.scale=0.6
            self.selection = [35,43,54,56]
        else:
            self.__net='DiaperModels/PadNet_Net.net'
            self.scale=0.8
            self.selection = [35,43,54,62]
    
    def new_project(self):
        self.labelphoto.config(image='')
        self.file_label.config(text='')
        self.result.config(text='')
        self.f.clear()
        self.canvas.get_tk_widget().pack_forget()
        self.canvas._tkcanvas.pack_forget()
    def save_results(self):
        save_name = filedialog.asksaveasfilename()
        self.f.suptitle(self.result.cget("text"))
        self.f.savefig(save_name)
        ms.showinfo('Saved','Results Are Stored')
    def exit1(self):
        answer = ms.askquestion('Exit?','Are You Sure to Exit from the DiaperNet App? \nYou are breaking my heart :(')
        if answer=='yes':
            self.root.quit()
        else:
            return
    def start_run_net(self):
        Thread(target=self.run_net, daemon=True).start()
    def run_net(self):
        self.labstat.config(text='Processing ...')
        vis = Visualization(self.__net)
        im_proc = Imaging()
        im = im_proc.Augmentation(img.imread(self.file_name),resize=self.scale)[0]/255.0 - 0.5
        test = Test(self.__net)
        out,_ = test.Test(np.expand_dims(im,axis=0))
        out = round(out.item()*100,2)
        self.result.config(text='Premiumness Score: '+str(out)+'%')
        f_map,raw_map,cov_map,heat_map = vis.Descriptor(im,inhibition=False)
        
        self.f = plt.figure()
        a1 = self.f.add_subplot(321)
        a1.imshow(f_map.squeeze().mean(axis=2))
        a1.axis('off')
        a1.set_title('Driving Regions')
        a2 = self.f.add_subplot(322)
        a2.imshow(cov_map.squeeze().mean(axis=2))
        a2.axis('off')
        a2.set_title('Driving Regions-Textures')
        a3 = self.f.add_subplot(323)
        a3.imshow(cov_map.squeeze()[:,:,self.selection[0]],cmap='Greys_r')
        a3.axis('off')
        a3.set_title('Driving Sub Patterns')
        a4 = self.f.add_subplot(324)
        a4.imshow(cov_map.squeeze()[:,:,self.selection[1]],cmap='Greys_r')
        a4.axis('off')
        a4.set_title('Driving Sub Patterns')
        a5 = self.f.add_subplot(325)
        a5.imshow(cov_map.squeeze()[:,:,self.selection[2]],cmap='Greys_r')
        a5.axis('off')
        a5.set_title('Driving Sub Patterns')
        a6 = self.f.add_subplot(326)
        a6.imshow(cov_map.squeeze()[:,:,self.selection[3]],cmap='Greys_r')
        a6.axis('off')
        a6.set_title('Driving Sub Patterns')
        
        self.canvas = FigureCanvasTkAgg(self.f, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack()
        self.canvas._tkcanvas.pack()
        
        self.labstat.config(text='Info: tavanaei.a@pg.com')
        
    def helpme(self):
        ms.showinfo('HELP','1- Select either "whole sheet" or "pad"\n'+
                    '2- Select a diaper image\n'+
                    '3- Run\n'+
                    '4- Save the results')
    def about(self):
        ms.showinfo('About Us','Designed by Amir Tavanaei, DS/AI-P&G')

diper = DiaperNetGUI()
