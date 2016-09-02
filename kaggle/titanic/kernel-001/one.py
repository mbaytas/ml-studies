import time

import numpy as np
import pandas as pd

print "# Reading data from files..."
train = pd.read_csv("../data/train.csv", dtype={"Age": np.float64})
test = pd.read_csv("../data/test.csv", dtype={"Age": np.float64})


def harmonize_data(df):
    df['Age'] = df['Age'].fillna(df['Age']).median()
    
    df.loc[df['Sex'] == "male", 'Sex'] = 0
    df.loc[df['Sex'] == "female", 'Sex'] = 1
    
    df['Embarked'] = df['Embarked'].fillna("S")

    df.loc[df['Embarked'] == "S", 'Embarked'] = 0
    df.loc[df['Embarked'] == "C", 'Embarked'] = 1
    df.loc[df['Embarked'] == "Q", 'Embarked'] = 2

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    return df
    

print "# 'Harmonizing' data..."
df_train = harmonize_data(train)
df_test  = harmonize_data(test)

    
def create_submission(alg, train, test, predictors, algString):

    alg.fit(train[predictors], train['Survived'])
    trainAccuracy = alg.score(train[predictors], train['Survived'])

    predictions = alg.predict(test[predictors])

    result = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })
    
    fileName = "../results/001_" +\
             algString + "_" +\
             str(int(time.time())) + "-" +\
             str(trainAccuracy)[2:7] + ".csv"
             
    result.to_csv(fileName, index=False)
    
    
print "# Getting ready to train..."

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

print "# Training Logistic Regression classifier..."

alg = LogisticRegression(random_state=1)

scores = cross_validation.cross_val_score(alg,
                                          df_train[predictors],
                                          df_train['Survived'],
                                          cv=3)
                                          
print "- Logistic Regression score:", scores.mean()

print "- Saving to file..."
create_submission(alg, df_train, df_test, predictors, "LRC")
    
print "# Training Random Forest classifier..."

alg = RandomForestClassifier(random_state=1,
                             n_estimators=150,
                             min_samples_split=4,
                             min_samples_leaf=2)

scores = cross_validation.cross_val_score(alg,
                                          df_train[predictors],
                                          df_train['Survived'],
                                          cv=3)

print "- RFC score:", scores.mean()

print "- Saving to file..."
create_submission(alg, df_train, df_test, predictors, "RFC")
