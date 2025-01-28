




import pandas as pd

creditdata= pd.read_csv('creditcard.csv')
creditprep= creditdata.copy()
creditprep.isnull().sum(axis=0)

X=creditprep.drop('Class', axis=1)
Y=creditprep['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, Y_train)


Y_pred = lr.predict(X_test)

from sklearn.metrics import classification_report
cr=classification_report(Y_test, Y_pred)