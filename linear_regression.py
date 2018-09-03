import numpy as np
import matplotlib.pyplot as plt
import re
import math
from pandas import read_csv
from sklearn.model_selection import train_test_split

def getdata(data):
    features=[]
    targets=[]
    file=open(data)
    for line in file:
        lines=[float(j) for j in re.findall(r'[+\d.\d]+',line)]
        #lines.pop(0)
        #lines.insert(0,1)
        targets.append(lines.pop())
        features.append(lines)
    features=np.array(features)
    targets=np.array(targets)
    X_train,X_test,Y_train,Y_test=train_test_split(features,targets,test_size=0.1,random_state=1)
    print(X_train.shape[1])
    return X_train,X_test,Y_train,Y_test
    
        
        
        
            
        
        
    
    
            
            


def Gradient_descent(trainset,parameters,target):
        alp=0.00000001 
        l=0.00001
        parameters[0]=parameters[0]-(alp*(np.sum((trainset.dot(parameters)-target))))/(2*len(trainset))
        
        for i in range(1,len(parameters)):
            parameters[i]=(parameters[i]*(1-((alp*l)/len(trainset))))-(alp*(np.sum((trainset.T).dot((trainset.dot(parameters)-target)))))/(2*len(trainset))
        #parameters[1:]=(parameters[1:]
        print(parameters)
        return parameters
    
    
    
def normal(trainset,p,target):
    l=1000
    a=len(trainset[1])
    z=np.zeros((a,a),int)
    np.fill_diagonal(z,1)
    z[0,0]=0
    k=z*l
    
    t1=((np.linalg.inv((trainset.T).dot(trainset)-k.T)).dot(trainset.T)).dot(target)
    
    
    return t1
     
     
    
           
    
def cost_function(parameters,trainset,target):
    l=0.00001
    cost=((np.sum((trainset.dot(parameters)-target))**2)+(np.sum(l*parameters[1:]**2)))/(2*len(trainset))
    return cost
            
def plotdata(cost,k):
    x=range(k)
    y=cost
    plt.plot(x,y)
    plt.show()

def plot(t,tar,p):
    x=[]
    t=t.tolist()
    y=tar.tolist()
    for i in t:
        x.append(i[1])
    
    plt.scatter(x,y)
    plt.plot(t,p)
    plt.show()
    
  
    
def predict(new_values,parameters,fuck):
    hx=new_values.dot(parameters)
    return hx


def cal_error(b,trainset,target):
      hx=abs(trainset.dot(b)-target)
      y=np.mean(hx)
      print(y)
      return  y
      #hx=hx.tolist()
      #print(hx) '''
      

    

def main():
    
    trainset,testset,target,pre_target=getdata('C:/Users/venu/Desktop/ml/datasets/work.txt.txt')
    b=[]
    
    for i in range(len(trainset[1])):
        i=[0]
        b.append(i)
    b=np.array(b)
    
    a=normal(trainset,b,target)
    cost=cost_function(a,trainset,target)
    
    h=predict(trainset,a,target)
        
    
    
    
   
    plot(trainset,target,h)
    k=[]
    x=1
    
    for j in range(x):
                b=Gradient_descent(trainset,b,target)
                cost=cost_function(b,trainset,target) 
                k.append(cost)

    h=predict(trainset,b,target)
    #plotdata(k,x)
    
    cal_error(b,trainset,target)
    
    
    
    
    values=int(input('how many values you want  to predict:\n'))
    s=[]
    print("enter the values to predict:")
    for m in range(values):
          m=[]
          area=float(input())
          m.append(1)
          m.append(area)
          s.append(m)
    s=np.array(s)      
    print('hey you got it...!')
    print(predict(s,b))
    
    predicted=predict(testset,b,pre_target)
    s=abs(predicted-pre_target)
    y=np.mean(s)
    print(y)
    
    for i in range(len(testset)):
        print(predicted[i],pre_target[i]
   
           

   
    
    
    
    
main()