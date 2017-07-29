import numpy as np

#returns svm
def perceptron(train,label,m):   
    size = len(train)
    w = np.zeros(len(train.transpose()))
    t = 0
    error = 0
    
    #batch algo m-times
    while count < m:
        while t < size:
            u = train[t] if np.linalg.norm(train[t]) == 0 else train[t]/np.linalg.norm(train[t]) 
            z = 1 if np.dot(w,u) >= 0 else -1
            if z != label[t]:
                w = w - u*z
            t += 1    
        count += 1
        t = 0
    
    #error testing
    while t < size:
        z = 1 if np.dot(w,train[t]) >= 0 else -1
        if z != label[t]:
            error += 1
        t += 1
    print ("number of batches: ", m)
    print ("training error: ", error, "out of", size)
    
    return w

    
def predict_perceptron(train,label,m,test):
    w = perceptron(train,label,m)
    size = len(test)
    t = 0
    predict = []
    
    while t < size:
        z = 1 if np.dot(w,test[t]) >= 0 else -1
        predict.append(z)
        t += 1
    
    print (predict)
        

train = np.loadtxt("train35.digits")
label = np.loadtxt("train35.labels")
test = np.loadtxt("test35.digits")  
      
predict_perceptron(train,label,10,test)



        
