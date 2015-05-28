


"""
An example script that fits the data with linear regression with a different orders of polynomial.

"""



# imports 
import scipy as sp
import numpy as np
import scipy.io as sio
import sklearn.linear_model as linear_model
import matplotlib.pyplot as plt

# import data
ex5_data = sio.loadmat('ex5data1.mat') # Loads the matlab/octave file as a dict

# Define variables
X = ex5_data['X']
y = ex5_data["y"]
Xtest = ex5_data['Xtest']
ytest = ex5_data['ytest']
Xval = ex5_data['Xval']
yval = ex5_data['yval']



# Define higer order features up to polynomial 10
N = 10
X10 = np.array([X.squeeze()**n for n in range(1,N+1)]).transpose()
Xtest10 = np.array([Xtest.squeeze()**n for n in range(1,N+1)]).transpose()

# Define a lr model and fit for each order polynomial
lr_models = [linear_model.LinearRegression(normalize=True) for n in range(N)]
[lr_model.fit(X10[:,:n+1], y) for n, lr_model in zip(range(N), lr_models)]
lr_models_ridgeCV = [linear_model.RidgeCV([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], normalize=True) for n in range(N)]
[lr_model_ridgeCV.fit(X10[:,:n+1], y) for n, lr_model_ridgeCV in zip(range(N), lr_models_ridgeCV)]

# Compute the training and test errors
for i, models in zip([0,1], [lr_models, lr_models_ridgeCV]):
    yfit_train = np.array([lr_model.predict(X10[:,:n+1]) for n, lr_model in zip(range(N), models)])
    yfit_test = np.array([lr_model.predict(Xtest10[:,:n+1]) for n, lr_model in zip(range(N), models)])

    # Cost functions for 
    Npoly = sp.arange(1,11)
    J_train = 1 / (2.0 * yfit_train.shape[1]) * ((y - yfit_train)**2).sum(1)
    J_test = 1 / (2.0 * yfit_test.shape[1]) * ((ytest - yfit_test)**2).sum(1)

    # Make a plot
    if i == 0 :
        f0 = plt.figure(0, (5,5), facecolor='white')
        f0.clf()
        a0 = f0.add_axes([.1, .1, .85, .85])
        
        a0.plot(Npoly, J_train, 'b', linewidth=2, label="err-train")
        a0.plot(Npoly, J_test, 'g', linewidth=2, label="err-test")
        a0.set_title("Error as a function of polynomial order")
    else:
        a0.plot(Npoly, J_train, '--b', linewidth=2, label="err-train-RidgeCV")
        a0.plot(Npoly, J_test, '--g', linewidth=2, label="err-test-RidgeCV")

a0.set_ybound(.001, 40)
a0.set_xbound(.5, 9.5)
a0.legend()
f0.show()
f0.savefig("wgm-ex5-learning-curve.png")
