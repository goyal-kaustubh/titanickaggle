#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#Taking The dataset
dataset = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")
train_age1 = dataset[dataset["Age"].notna()]
test_age1 = dataset[dataset["Age"].isnull()]
#dividing dataset in test and train
train_age1 = train_age1.iloc[:,[5,6,7]]
test_age1 = test_age1.iloc[:,[5,6,7]]

#Dividing into Classes
train_age_x1 = train_age1.iloc[:,1:3]
train_age_y1 = train_age1.iloc[:,0:1]
test_age1 = test_age1.drop("Age",1)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(train_age_x1, train_age_y1)

y_pred = regressor.predict(test_age1)



'''labelencoder_x1 = LabelEncoder()
train_age_x["SibSp"] = labelencoder_x1.fit_transform(train_age_x["SibSp"])
labelencoder_x2 = LabelEncoder()
train_age_x["Parch"] = labelencoder_x2.fit_transform(train_age_x["Parch"])
'''


train1 = dataset[dataset["Age"].notna()]
train2 = dataset[dataset["Age"].isnull()]
train2["Age"] = y_pred
train2["Age"] = train2["Age"].round()
frames = [train1,train2]
#Ready DataSet
dataset = pd.concat(frames)
dataset = dataset[dataset["Embarked"].notna()]
x = dataset.iloc[:,[2,4,5,11]]
y = dataset.iloc[:,1:2]

x.dtypes


#Categorical Variables

labelencoder_x1 = LabelEncoder()
x["Pclass"] = labelencoder_x1.fit_transform(x["Pclass"])
labelencoder_x2 = LabelEncoder()
x["Sex"] = labelencoder_x2.fit_transform(x["Sex"])
labelencoder_x4 = LabelEncoder()
x["Embarked"] = labelencoder_x4.fit_transform(x["Embarked"])
'''x['Age'] = pd.cut(x['Age'],bins = 4)
labelencoder_x3 = LabelEncoder()
x["Age"] = labelencoder_x3.fit_transform(x["Age"])'''
onehotencoder1 = OneHotEncoder(categorical_features = [0])
x = onehotencoder1.fit_transform(x).toarray()
x = x[:,1:6]
onehotencoder2 = OneHotEncoder(categorical_features = [2])
x = onehotencoder2.fit_transform(x).toarray()
x = x[:,1:6]
onehotencoder3 = OneHotEncoder(categorical_features = [4])
x = onehotencoder3.fit_transform(x).toarray()
x = x[:,1:7]


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x, y)







#For Test Set
train_age2 = test_set[test_set["Age"].notna()]
test_age2 = test_set[test_set["Age"].isnull()]
#dividing dataset in test and train
train_age2 = train_age2.iloc[:,[4,5,6]]
test_age2 = test_age2.iloc[:,[4,5,6]]

#Dividing into Classes
train_age_x2 = train_age2.iloc[:,1:3]
train_age_y2 = train_age2.iloc[:,0:1]
test_age2 = test_age2.drop("Age",1)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(train_age_x2, train_age_y2)

y_pred = regressor.predict(test_age2)

train3 = test_set[test_set["Age"].notna()]
train4 = test_set[test_set["Age"].isnull()]
train4["Age"] = y_pred
train4["Age"] = train4["Age"].round()
frames2 = [train3,train4]
#Ready DataSet
test_set = pd.concat(frames2)




x2 = test_set.iloc[:,[1,3,4,10]]

labelencoder_x1 = LabelEncoder()
x2["Pclass"] = labelencoder_x1.fit_transform(x2["Pclass"])
labelencoder_x2 = LabelEncoder()
x2["Sex"] = labelencoder_x2.fit_transform(x2["Sex"])
labelencoder_x4 = LabelEncoder()
x2["Embarked"] = labelencoder_x4.fit_transform(x2["Embarked"])
'''x['Age'] = pd.cut(x['Age'],bins = 4)
labelencoder_x3 = LabelEncoder()
x["Age"] = labelencoder_x3.fit_transform(x["Age"])'''
onehotencoder1 = OneHotEncoder(categorical_features = [0])
x2 = onehotencoder1.fit_transform(x2).toarray()
x2 = x2[:,1:6]
onehotencoder2 = OneHotEncoder(categorical_features = [2])
x2 = onehotencoder2.fit_transform(x2).toarray()
x2 = x2[:,1:6]
onehotencoder3 = OneHotEncoder(categorical_features = [4])
x2 = onehotencoder3.fit_transform(x2).toarray()
x2 = x2[:,1:7]


t_pred = classifier.predict(x2)
ans = pd.DataFrame()
ans["PassengerId"] = test_set["PassengerId"]
ans["Survived"] = t_pred.astype(int)
ans.to_csv("Titanic_Soln.csv",index = False)