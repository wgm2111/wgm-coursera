




from __future__ import print_function, division
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from ex2data import X2, y2



# make a Logistic Regression model
logistic_model_data2 = linear_model.LogisticRegression(penalty='l2', intercept_scaling=X2.mean())
logistic_model_data2.fit(X2, y2)

# make a polynomial logistic regression model
model = Pipeline([('poly', PolynomialFeatures(degree=2)), 
                  ('linear', linear_model.LogisticRegression(fit_intercept=False))])
logistic_poly_model_data2 = model.fit(X2, y2)


