# import numpy as np
# from sklearn.datasets import fetch_california_housing


# def loss(w, X, Y):
#     m = X.shape[0]
#     errors = np.dot(X, w) - Y
#     return np.dot(errors, errors.T) / (2 * m)


# def decay_alpha(alpha, epoch, decay_rate=0.1, lower_bound=1e-4):
#     return alpha * np.exp(-decay_rate * epoch) if alpha > lower_bound else lower_bound


# def linear_regression(alpha, w, X, Y, epochs=1000, min_value=1e-6):
#     ''' 
#         Describe about what the function is doing, as well as description about all the input variables.
#         Also add the example on how to run it properly.
#     '''
#     step = 0
#     losses = []
#     L = loss(w, X, Y)
#     m = X.shape[0]

#     for epoch in range(epochs):
#         step = epoch + 1
#         errors = np.dot(X, w) - Y

#         # measure the gradient of the learning parameters.
#         gd_w = np.dot(X.T, errors) / m

#         # recalculate the learning rate alpha using decay function.
#         alpha = decay_alpha(alpha, epoch, decay_rate=0.2, lower_bound=1e-3)

#         # update new weights' values.
#         w = w - alpha * gd_w

#         # everything about the losses :  remeasure the loss and add to loss list. ... 
#         L = loss(w, X, Y)
#         losses.append(L)

#         if epoch % 100 == 0:
#             print(f"epoch: {epoch}, alpha={alpha}, Loss is {L}")

#         if L < min_value:
#             break

#     return w, step, L


# def predict_house_price(features, w):
#     return features @ w


# def preprocess_output(Y):
#     return Y * 1.0 / 10.0


# def reverse_preprocess_output(Y):
#     return Y * 10.0


# if __name__ == "__main__":
#     cali = fetch_california_housing()
#     X = cali.data
#     Y = cali.target
#     Y = preprocess_output(Y)

#     n_data, n_features = X.shape

#     mean = X.mean(axis=0)
#     std = X.std(axis=0)
#     std[std == 0] = 1.0
#     X = (X - mean) / std

#     bias = np.ones((n_data, 1))
#     X = np.concatenate((bias, X), axis=1)     # we add one more column where all the values is 1 because it is for the bias term.

#     print(X.shape)

#     # we add 1 for the features because we have the bias term that we appended to X.
#     w = np.zeros(n_features + 1)
#     alpha = 0.05
#     learned_w, step, loss_value = linear_regression(alpha, w, X, Y)
#     print(learned_w)
#     print(f"step={step}, loss={loss_value}")

#     print("Now predicting a house price using the below random features")
#     features = X[0].copy() 
#     print("features:", features)
#     # apply all the normalization that we did for X as before.
#     # feautres = scaler.fit_transform(features)
#     # features = (features - mean) / std
#     predicted_normalized_price = predict_house_price(features, learned_w) * 1e6
#     print(f"predicted normalized price = {predicted_normalized_price}")
#     print(f"final price: {reverse_preprocess_output(predicted_normalized_price)}")


import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def L1_regularization(w,X,Y,beta) :
    m=X.shape[0]
    errors=np.dot(X,w) - Y
    J=np.dot(errors,errors.T)/(2*m)
    absolute_value=np.sum(np.abs(w))
    L1=J + beta*absolute_value
    return L1

def decay_alpha(alpha,epoch,decay_rate=0.001,lower_bound=1e-4) :
    if alpha <= lower_bound :
        return alpha
    else :
        alpha=alpha*np.exp(-decay_rate*epoch)
        return alpha

def gradien(w,X,Y,beta,alpha):
    m=X.shape[0]
    errors=np.dot(X,w) - Y
    gd_J=np.dot(X.T,errors)/m
    gd_absolute_value=beta*np.sign(w)
    w=w - alpha*(gd_J+gd_absolute_value)
    return w
    
def linear_regression(w,X,Y,alpha,beta,max_epoch=1000,min_loss=1e-7) :
    steps=0
    m=X.shape[0]
    Loss=L1_regularization(w,X,Y,beta)
    for i in range(max_epoch+1) :
        steps=i+1
        alpha=decay_alpha(alpha,i)
        w=gradien(w,X,Y,beta,alpha)
        Loss=L1_regularization(w,X,Y,beta)
        if Loss<min_loss :
            break
    return steps,w,Loss

def preprocess(Y) :
    return Y *1.0/10.0

def reverse_process(Y) : 
    return Y * 10.0

def evaluate(w,X,Y):
    y_pred=np.dot(X,w)
    errors=y_pred-Y
    Mse=np.mean(errors**2)
    return Mse

if __name__ == "__main__" :
    cali=fetch_california_housing()
    X=cali.data
    Y=cali.target
    y=preprocess(Y)
    X_train,X_temp,Y_train,Y_temp=train_test_split(X,Y,test_size=0.3,random_state=42,shuffle=True)
    X_validation,X_test,Y_validation,Y_test=train_test_split(X_temp,Y_temp,test_size=(2/3),random_state=42,shuffle=True)
    mean=np.mean(X_train,axis=0)
    std=np.std(X_train,axis=0)
    std[std==0]=1.0
    X_train=(X_train-mean)/std
    X_validation=(X_validation-mean)/std
    X_test=(X_test-mean)/std
    bias=np.ones((X_train.shape[0],1))
    X_train=np.concatenate((X_train,bias),axis=1)
    bias=np.ones((X_validation.shape[0],1))
    X_validation=np.concatenate((X_validation,bias),axis=1)
    bias=np.ones((X_test.shape[0],1))
    X_test=np.concatenate((X_test,bias),axis=1)
    n_features=X.shape[1]
    w=np.random.random(n_features+1)
    choose_beta=[0.0,0.0001,0.001,0.01,0.1,1.0]
    min_loss=1e9
    W_use=w
    for beta in choose_beta :
        steps,W,L=linear_regression(w.copy(),X_train,Y_train,0.2,beta)
        MSe=evaluate(W,X_validation,Y_validation)
        if MSe < min_loss:
            min_loss=MSe
            W_use=W
    print(evaluate(W_use,X_test,Y_test))
    


