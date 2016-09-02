import time

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

print "# Reading data from files..."
df_train = pd.read_csv("../data/train.csv", dtype={"Age": np.float64}, )
df_test = pd.read_csv("../data/test.csv", dtype={"Age": np.float64}, )

print "# Pre-processing"

print "- Dropping unnecessary columns..."
# drop unnecessary columns, these columns won't be useful in analysis and prediction
df_train = df_train.drop(['PassengerId','Name','Ticket'], axis=1)
df_test = df_test.drop(['Name','Ticket'], axis=1)

# Embarked
df_train.drop(['Embarked'], axis=1,inplace=True)
df_test.drop(['Embarked'], axis=1,inplace=True)

print "- Processing variable: Fare"
# only for df_test, since there is a missing "Fare" values
df_test["Fare"].fillna(df_test["Fare"].median(), inplace=True)
# convert from float to int
df_train['Fare'] = df_train['Fare'].astype(int)
df_test['Fare'] = df_test['Fare'].astype(int)

# # Fare

# # get fare for survived & didn't survive passengers
# fare_not_survived = df_train["Fare"][df_train["Survived"] == 0]
# fare_survived     = df_train["Fare"][df_train["Survived"] == 1]

# # get average and std for fare of survived/not survived passengers
# avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
# std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

# avgerage_fare.index.names = std_fare.index.names = ["Survived"]

print "- Processing variable: Age"

# get average, std, and number of NaN values in df_train
avg_age_train  = df_train["Age"].mean()
std_age_train = df_train["Age"].std()
nanCount_age_train = df_train["Age"].isnull().sum()

# get average, std, and number of NaN values in df_test
average_age_test = df_test["Age"].mean()
std_age_test = df_test["Age"].std()
nanCount_age_test = df_test["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(avg_age_train - std_age_train, avg_age_train + std_age_train, size = nanCount_age_train)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = nanCount_age_test)

# fill NaN values in Age column with random values generated
df_train["Age"][np.isnan(df_train["Age"])] = rand_1
df_test["Age"][np.isnan(df_test["Age"])] = rand_2

# convert from float to int
df_train['Age'] = df_train['Age'].astype(int)
df_test['Age']    = df_test['Age'].astype(int)

print "- Processing variable: Cabin"
# It has a lot of NaN values, so it won't cause a remarkable impact on prediction
df_train.drop("Cabin",axis=1,inplace=True)
df_test.drop("Cabin",axis=1,inplace=True)

print "- Processing features: Family"

# Instead of having two columns Parch & SibSp,
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
df_train['Family'] = df_train["Parch"] + df_train["SibSp"]
df_train['Family'].loc[df_train['Family'] > 0] = 1
df_train['Family'].loc[df_train['Family'] == 0] = 0

df_test['Family'] =  df_test["Parch"] + df_test["SibSp"]
df_test['Family'].loc[df_test['Family'] > 0] = 1
df_test['Family'].loc[df_test['Family'] == 0] = 0

# drop Parch & SibSp
df_train = df_train.drop(['SibSp','Parch'], axis=1)
df_test    = df_test.drop(['SibSp','Parch'], axis=1)

# average of survived for those who had/didn't have any family member
# family_perc = df_train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

print "- Processing variable: Sex"

# Sex

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex

df_train['Person'] = df_train[['Age','Sex']].apply(get_person,axis=1)
df_test['Person']    = df_test[['Age','Sex']].apply(get_person,axis=1)

# No need to use Sex column since we created Person column
df_train.drop(['Sex'],axis=1,inplace=True)
df_test.drop(['Sex'],axis=1,inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
personDummies_train = pd.get_dummies(df_train['Person'])
personDummies_train.columns = ['Child','Female','Male']
personDummies_train.drop(['Male'], axis=1, inplace=True)

personDummies_test = pd.get_dummies(df_test['Person'])
personDummies_test.columns = ['Child','Female','Male']
personDummies_test.drop(['Male'], axis=1, inplace=True)

df_train = df_train.join(personDummies_train)
df_test = df_test.join(personDummies_test)

# average of survived for each Person(male, female, or child)
# person_perc = df_train[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

df_train.drop(['Person'], axis=1, inplace=True)
df_test.drop(['Person'], axis=1, inplace=True)

print "- Processing variable: Pclass"

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclassDummies_train = pd.get_dummies(df_train['Pclass'])
pclassDummies_train.columns = ['Class_1','Class_2','Class_3']
pclassDummies_train.drop(['Class_3'], axis=1, inplace=True)

pclassDummies_test = pd.get_dummies(df_test['Pclass'])
pclassDummies_test.columns = ['Class_1','Class_2','Class_3']
pclassDummies_test.drop(['Class_3'], axis=1, inplace=True)

df_train.drop(['Pclass'], axis=1, inplace=True)
df_test.drop(['Pclass'], axis=1, inplace=True)

df_train = df_train.join(pclassDummies_train)
df_test = df_test.join(pclassDummies_test)

print "# Getting ready to learn..."

X_train = df_train.drop("Survived",axis=1)
y_train = df_train["Survived"]
X_test  = df_test.drop("PassengerId",axis=1).copy()

# Prepare for generating output files
def burn(predictions, accuracy, methodString):
  print "Writing", methodString, "predictions to file..."
  global df_test
  result = pd.DataFrame({"PassengerId": df_test["PassengerId"],
                         "Survived": predictions})
  fileName = "../results/journey_" +\
             methodString + "_" +\
             str(int(time.time())) + "-" +\
             str(accuracy)[2:7] + ".csv"
  result.to_csv(fileName, index=False)

print "# Training Logistic Regression classifier..."
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_train, y_train)
print "- Logistic Regression accuracy on training data:", score
y_pred = logreg.predict(X_test)
burn(y_pred, score, "LRC")

print "# Training Support Vector classifier..."
svc = SVC()
svc.fit(X_train, y_train)
score = svc.score(X_train, y_train)
print "- SVC accuracy on training data:", score
y_pred = svc.predict(X_test)
burn(y_pred, score, "SVC")

print "# Training Random Forest classifier..."
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
score = random_forest.score(X_train, y_train)
print "- RFC accuracy on training data:", score
y_pred = random_forest.predict(X_test)
burn(y_pred, score, "RFC")

print "# Training K-Nearest Neighbors classifier..."
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
score = knn.score(X_train, y_train)
print "- KNN accuracy on training data:", score
y_pred = knn.predict(X_test)
burn(y_pred, score, "KNN")

print "# Training Gaussian Naive Bayes classifier..."
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
score = gaussian.score(X_train, y_train)
print "- Gaussian Naive Bayes accuracy on training data:", score
y_pred = gaussian.predict(X_test)
burn(y_pred, score, "GNB")