#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:03:13 2018

@author: justintian
a """

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal

class MOG():
    def __init__(self,numOfGauss=3,BG_thresh=0.6, lr=0.01, height=240, width=320):
        self.numOfGauss=numOfGauss
        self.BG_thresh=BG_thresh
        self.lr=lr
        self.height=height
        self.width=width
        self.mus=np.zeros((self.height,self.width, self.numOfGauss,3)) ## assuming using color frames
        self.sigmaSQs=np.zeros((self.height,self.width,self.numOfGauss)) ## all color channels share the same sigma and covariance matrices are diagnalized
        self.omegas=np.zeros((self.height,self.width,self.numOfGauss))
        for i in range(self.height):
            for j in range(self.width):
                self.mus[i,j]=np.array([[122, 122, 122]]*self.numOfGauss) ##assuming a [0,255] color channel
                self.sigmaSQs[i,j]=[36.0]*self.numOfGauss
                self.omegas[i,j]=[1.0/self.numOfGauss]*self.numOfGauss
                
    def reorder(self):
        BG_pivot=np.zeros((self.height,self.width),dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                BG_pivot[i,j]=-1
                ratios=[]
                for k in range(self.numOfGauss):
                    ratios.append(self.omegas[i,j,k]/np.sqrt(self.sigmaSQs[i,j,k]))
                indices=np.array(np.argsort(ratios)[::-1])
                self.mus[i,j]=self.mus[i,j][indices]
                self.sigmaSQs[i,j]=self.sigmaSQs[i,j][indices]
                self.omegas[i,j]=self.omegas[i,j][indices]
                cummProb=0
                for l in range(self.numOfGauss):
                    cummProb+=self.omegas[i,j,l]
                    if cummProb>=self.BG_thresh and l<self.numOfGauss-1:
                        BG_pivot[i,j]=l
                        break
                ##if no background pivot is made the last one is foreground
                if BG_pivot[i,j]==-1:
                    BG_pivot[i,j]=self.numOfGauss-2
        return BG_pivot
    
    def updateParam(self, curFrame, BG_pivot):
        labels=np.zeros((self.height,self.width))
        for i in range(self.height):
            for j in range(self.width):
                X=curFrame[i,j]
                match=-1
                for k in range(self.numOfGauss):
                    CoVarInv=np.linalg.inv(self.sigmaSQs[i,j,k]*np.eye(3))
                    X_mu=X-self.mus[i,j,k]
                    dist=np.dot(X_mu.T, np.dot(CoVarInv, X_mu))
                    if dist<6.25*self.sigmaSQs[i,j,k]:
                        match=k
                        break
                if match!=-1:  ## a match found
                    ##update parameters
                    self.omegas[i,j]=(1.0-self.lr)*self.omegas[i,j]
                    self.omegas[i,j,match]+=self.lr
                    rho=self.lr * multivariate_normal.pdf(X,self.mus[i,j,match],np.linalg.inv(CoVarInv))
                    self.sigmaSQs[match]=(1.0-rho)*self.sigmaSQs[i,j,match]+rho*np.dot((X-self.mus[i,j,match]).T, (X-self.mus[i,j,match]))
                    self.mus[i,j,match]=(1.0-rho)*self.mus[i,j,match]+rho*X
                    ##label the pixel
                    if match>BG_pivot[i,j]:
                        labels[i,j]=250
                else:
                    self.mus[i,j,-1]=X
                    labels[i,j]=250
        return labels
            
    def streamer(self):
        ## initialize pixel gaussians
        cap=cv2.VideoCapture(0)
        cap.set(3,self.width)
        cap.set(4,self.height)
        frameCnt=0
        
        
        while(True):
            ret,frame=cap.read()
            frameCnt+=1
            if frameCnt%10==0:
                print("number of frames: ", frameCnt)
                BG_pivots=self.reorder()
                labels=self.updateParam(frame,BG_pivots)
                #labels=cv2.cvtColor(labels,cv2.COLOR_GRAY2BGR)
                #stacked = np.concatenate((image, labels), axis=1)
                cv2.imshow('FGBG',labels)
                cv2.imwrite( "res{}.png".format(frameCnt), labels );
            #cv2.imshow('frame',frame)
            #print(frame[0,0])
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        
        plt.show()
        cap.release()
        cv2.destroyAllWindows()
        
        
        
def main():
    """
    cap=cv2.VideoCapture(0)
    cap.set(3,320)
    cap.set(4,240)
    while(True):
        ret,frame=cap.read()
        cv2.imshow('frame',frame)
        print(frame[0,0])
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    """
    subtractor=MOG()
    subtractor.streamer()
    
    
if __name__=='__main__':
    main()