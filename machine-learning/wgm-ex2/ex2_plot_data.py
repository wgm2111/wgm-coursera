
# Load the data as pandas data frames

from __future__ import print_function, division


import scipy as sp
import matplotlib.pyplot as plt

from ex2data import data1, data2, X1, X2, y1, y2
from ex2data1_regression import logistic_model_data1, logistic_poly_model_data1
from ex2data2_regression import logistic_model_data2, logistic_poly_model_data2

# output folder
FOUT1 = "figures/data1_scatter.png"
FOUT2 = "figures/data2_scatter.png"


# make a figure
def make_figure(FOUT, data, X, y, sklearn_model, 
                savefig=True, use_figure_axis=[None, None]):

    # Make a figure for the first data set
    if use_figure_axis[0]==None:
        fig = plt.figure(0, (6,6), facecolor='white')
        fig.clf()
        ax = fig.add_axes([.08, .08, .85, .85])

        # plot the data as a scatter plot
        xplot, yplot = sp.array(data[data.ix[:,2]==1].ix[:, :2]).T
        ax.plot(xplot, yplot, c='blue', linewidth=0, marker='o',label='y=1') # plot the y=1 features
        xplot, yplot = sp.array(data[data.ix[:,2]==0].ix[:, :2]).T
        ax.plot(xplot, yplot, c='red', linewidth=0, marker='x',label='y=0') # plot the y=0 features
        
    else:
        fig, ax = use_figure_axis

    # plot decision boundary as a contour plot
    _x = sp.linspace(X.min(), X.max(), 250)
    xx, yy = sp.meshgrid(_x, _x)
    Xplot = sp.array([xx.reshape(-1), yy.reshape(-1)]).T
    plotvals = (
        sklearn_model.predict_proba(Xplot).transpose().reshape(2, xx.shape[0], xx.shape[1]))
    plotvals = plotvals[1,:]
    ax.contour(xx, yy, plotvals, levels=[.5], label='linear fit')
    
    # Output to file
    if savefig==True:
        ax.legend()
        ax.set_title("Logistic regression example")
        fig.savefig(FOUT)

    return fig, ax



# make a figure for the first dataset
fig1, ax1 = make_figure(FOUT1, data1, X1, y1, logistic_model_data1, savefig=False)
fig1, ax1 = make_figure(FOUT1, data1, X1, y1, logistic_poly_model_data1,
                        use_figure_axis=[fig1, ax1])


# make a figure for the second dataset
fig2, ax2 = make_figure(FOUT2, data2, X2, y2, logistic_model_data2, savefig=False)
fig2, ax2 = make_figure(FOUT2, data2, X2, y2, logistic_poly_model_data2,
                        use_figure_axis=[fig2, ax2])


# # Make a figure for the first data set
# fig_data1 = plt.figure(0, (6,6), facecolor='white')
# fig_data1.clf()
# ax1 = fig_data1.add_axes([.08, .08, .85, .85])

# # plot the data as a scatter plot
# xplot, yplot = sp.array(data1[data1.ix[:,2]==1].ix[:, :2]).T
# ax1.plot(xplot, yplot, c='blue', linewidth=0, marker='o',label='y=1') # plot the y=1 features
# xplot, yplot = sp.array(data1[data1.ix[:,2]==0].ix[:, :2]).T
# ax1.plot(xplot, yplot, c='red', linewidth=0, marker='x',label='y=0') # plot the y=0 features

# # plot the probability function as a contour plot
# # --
# _x = sp.linspace(X1.min(), X1.max(), 250)
# xx, yy = sp.meshgrid(_x, _x)
# Xplot = sp.array([xx.reshape(-1), yy.reshape(-1)]).T

# # plot with only linear features
# out = (logistic_model_data1.predict_proba(Xplot)
#        ).transpose().reshape(2, xx.shape[0], xx.shape[1])
# vals = out[1,:]
# ax1.contour(xx, yy, vals, levels=[.5], label='linear fit')

# # plot a fit with quadratic featuers
# out = (logistic_poly_model_data1.predict_proba(Xplot)
#        ).transpose().reshape(2, xx.shape[0], xx.shape[1])
# vals = out[1,:]
# ax1.contour(xx, yy, vals, c='green', levels=[.5], label='quadradic fit')


# ax1.legend()
# ax1.set_title("Data set 1 for logistic regression")

# fig_data1.savefig(FOUT1)
