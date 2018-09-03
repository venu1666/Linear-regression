import numpy as np
import matplotlib as plt
import re
import sklearn as s



def load_data(file):
    features=[]
    targets=[]
    x=open(file)
    for line in x:
      lines=[float(j) for j in re.findall(r'[+\d.\d]+',line)]
      
      targets.append(lines.pop())
      lines.insert(0,1)
      features.append(lines)
    X=np.array(features)
    Y=np.array(targets)
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=1)
    
    from sklearn import linear_model
    reg=linear_model.LinearRegression()

    reg.fit(X_train,Y_train)
    
    #print(reg.predict(X_test))
    
    regg=Ridge(alpha=0.001,normalize=True)
    regg.fit(X_train,Y_train)
    y=regg.predict(X_test)
    from sklearn.metrics import mean_squared_error
    a=mean_squared_error(Y_train,regg.predict(X_train))
    b=mean_squared_error(Y_test,regg.predict(X_test))
    print(abs(a-b))
    print(a)
    print(b)
    Ridge()
    
    
    
    
def main():    
    
    load_data('C:/Users/venu/Desktop/ml/datasets/work.txt.txt')   



main()

