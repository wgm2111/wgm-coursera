




from __future__ import print_function, division
from sklearn import linear_model


from ex2data import X1, y1



# make a Logistic Regression model
logistic_model_data1 = linear_model.LogisticRegression(penalty='l2', intercept_scaling=X1.mean())
logistic_model_data1.fit(X1, y1)


