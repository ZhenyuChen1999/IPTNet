from locale import normalize
import numpy as np
import cv2 as cv
import os
import scipy.optimize as op
from sklearn.linear_model import LinearRegression
import copy
import numba
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
eps=1e-5

# Dir of (reformatted) dataset.
ROOT="./dataset/"

# Dir of single channel masks.
MASK_ROOT="./mask/"
    

@numba.njit
def singlePixelRegression(points):
    Sum_X=0.0+eps
    Sum_Y=0.0+eps
    Sum_Z=0.0+eps
    Sum_XZ=0.0+eps
    Sum_YZ=0.0+eps
    Sum_Z2=0.0+eps
 
    for i in range(0,points.shape[0]):
        xi=points[i,0]
        yi=points[i,1]
        zi=points[i,2]
 
        Sum_X = Sum_X + xi
        Sum_Y = Sum_Y + yi
        Sum_Z = Sum_Z + zi
        Sum_XZ = Sum_XZ + xi*zi
        Sum_YZ = Sum_YZ + yi*zi
        Sum_Z2 = Sum_Z2 + zi**2
 
    n = points.shape[0]
    den = n*Sum_Z2 - Sum_Z * Sum_Z
    k1 = (n*Sum_XZ - Sum_X * Sum_Z)/ den
    b1 = (Sum_X - k1 * Sum_Z)/n
    k2 = (n*Sum_YZ - Sum_Y * Sum_Z)/ den
    b2 = (Sum_Y - k2 * Sum_Z)/n
    
    # ignore b.
    
    temp=np.empty((3))
    temp[0]=k1
    temp[1]=k2
    temp[2]=1.0
    
    return temp/np.linalg.norm(temp)

@numba.njit
def bias(points:np.ndarray,albedo:np.ndarray):
    new=np.copy(points)
    
    n=points.shape[0]
    sum=np.empty((3),dtype=np.float32)
    sum[0]=0
    sum[1]=0
    sum[2]=0
    for i in range(n):
        temp=np.linalg.norm(new[i,:])
        if temp!=0.0:
            new[i,:]/=temp
        sum+=((new[i,:]-albedo)**2)
    return sum/n
    
def generateVarianceMapKernel(block:np.ndarray,albedoMap:np.ndarray,biasMap:np.ndarray):
    shape=block.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            # print(i,j)
            albedoMap[i,j]=singlePixelRegression(block[i,j,:,:])
            biasMap[i,j]=bias(block[i,j,:,:],albedoMap[i,j])
            
    return albedoMap,biasMap

def generateVarinceMap(imgSeries:list(),mask:np.ndarray):
    shape=imgSeries[0].shape
    stackB=imgSeries[0][:,:,0]
    stackG=imgSeries[0][:,:,1]
    stackR=imgSeries[0][:,:,2]
    
    stackB=stackB[:,:,np.newaxis]
    stackG=stackG[:,:,np.newaxis]
    stackR=stackR[:,:,np.newaxis]
    
    for i in range(1,len(imgSeries)):
        stackB=np.concatenate((stackB,imgSeries[i][:,:,0][:,:,np.newaxis]),axis=-1)
        stackG=np.concatenate((stackG,imgSeries[i][:,:,1][:,:,np.newaxis]),axis=-1)
        stackR=np.concatenate((stackR,imgSeries[i][:,:,2][:,:,np.newaxis]),axis=-1)
        
    stackB=stackB[:,:,:,np.newaxis]
    stackG=stackG[:,:,:,np.newaxis]
    stackR=stackR[:,:,:,np.newaxis]
    stack=np.concatenate((stackB,stackG,stackR),axis=-1)
    
    albedo=np.zeros([shape[0],shape[1],3],dtype=stack.dtype)
    variance=np.zeros([shape[0],shape[1],3],dtype=stack.dtype)
    albedo,variance=generateVarianceMapKernel(stack,albedo,variance)
        
    return albedo*mask,variance*mask

if __name__=="__main__":
    
    for i in range(0,50):
        
        imgList=[]
        
        paramFile=open(ROOT+str(i)+"/params.txt")
        paramLine=paramFile.readline()
        paramFile.close()
        # paramWord=paramLine.split(" ")[1].split(".")[0]
        paramWord=paramLine.split(" ")[1].split("/")[2].split(".")[0]
        print(paramWord)
        
        mask=cv.imread(MASK_ROOT+paramWord+".png")/255.0
        mask=mask.astype(np.float32)
        
        for j in range(10):
            img=cv.imread(ROOT+str(i)+"/"+str(i)+"_"+str(j)+".exr",-1)
            if not img is None:
                # img=img**(1/0.45)
                img=img.astype(np.float32)
                imgList.append(img)
        img,biasImage=generateVarinceMap(imgList,mask)
        
        cv.imwrite(ROOT+str(i)+"/albedo.exr",img.astype(np.float32))
        cv.imwrite(ROOT+str(i)+"/bias.exr",biasImage.astype(np.float32))