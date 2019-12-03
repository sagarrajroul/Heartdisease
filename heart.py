import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from fancyimpute import KNN 
#import random

dataset= pd.read_csv('heart_disease_data.csv')

Y=dataset['target']
X = dataset.drop(['target'],axis=1)
#dataset.isnull().sum()
#conti = ['age','testbps','chol','thalach','oldpeak']

#Adding missing value to the dataset
a=X[['age','trestbps','chol','thalach','oldpeak']]
X=X.drop(a,axis=1)
a=a.mask(np.random.random(a.shape)<0.2)
a.isnull().sum()

#merging two dataset
X=pd.concat([X, a.reindex(X.index)], axis=1)

#missing value analysis
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer=imputer.fit(X[['age','trestbps','chol','thalach','oldpeak']])
X[['age','trestbps','chol','thalach','oldpeak']]=imputer.transform(X[['age','trestbps','chol','thalach','oldpeak']])

#X.info()
#adding dummy variable 
from sklearn.preprocessing import StandardScaler
X = pd.get_dummies(X, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X[columns_to_scale] = standardScaler.fit_transform(X[columns_to_scale])

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

##Decission Tree
from sklearn.tree import DecisionTreeClassifier
dtscores = []
for i in range(1, len(X.columns) + 1):
    dtclassifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dtclassifier.fit(X_train, y_train)
    dtscores.append(dtclassifier.score(X_test, y_test))
plt.plot([i for i in range(1, len(X.columns) + 1)], dtscores, color = 'red')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dtscores[i-1], (i, dtscores[i-1]))
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of features')

#calculating accuracy
from sklearn.metrics import accuracy_score
dt_pred=dtclassifier.predict(X_test)
print (round(accuracy_score(y_test, dt_pred )*100,2))

#showing confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, dt_pred )
print(cm)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
model_1 = LogisticRegression(penalty='l1',solver='liblinear',random_state=0,multi_class='ovr',C=0.1)
model_1.fit(X_train,y_train)

#predicting the test variable
y_pred_1 =  model_1.predict(X_test)

print(round(accuracy_score(y_test,y_pred_1)*100))
cm1=confusion_matrix(y_test,y_pred_1)
print(cm1)




