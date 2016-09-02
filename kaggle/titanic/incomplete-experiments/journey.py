# Based on https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Options
sns.set_style('whitegrid')
np.set_printoptions(precision=4, threshold=10000, linewidth=160, edgeitems=999, suppress=True)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.width', 160)
pandas.set_option('expand_frame_repr', False)
pandas.set_option('precision', 4)

print "# Reading data from files"

df_train = pandas.read_csv("data/train.csv", dtype={"Age": np.float64})
df_test = pandas.read_csv("data/test.csv", dtype={"Age": np.float64})

print "# Head (Training Data):\n"
print df_train.head()

print "\n# Info for training data:"
print df_train.info()
print "\n# Info for test data:"
print df_test.info()

print "\n# Dropping some variables..."
df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1)
df_test = df_test.drop(['Name', 'Ticket'], axis=1)

print "\n# Processing variable: Fare"

df_test["Fare"].fillna(df_test["Fare"].median(), inplace=True)

df_train['Fare'] = df_train['Fare'].astype(int)
df_test['Fare'] = df_test['Fare'].astype(int)


print "\n# Processing variable: Age"

avg_age_train = df_train["Age"].mean()
std_age_train = df_train["Age"].std()
nanCount_age_train = df_train["Age"].isnull().sum()

avg_age_test = df_test["Age"].mean()
std_age_test = df_test["Age"].std()
nanCount_age_test = df_test["Age"].isnull().sum()

rand_1 = np.random.randint(avg_age_train - std_age_train,
                           avg_age_train + std_age_train,
                           size=nanCount_age_train)
rand_2 = np.random.randint(avg_age_test - std_age_test,
                           avg_age_test + std_age_test,
                           size=nanCount_age_test)

df_train["Age"][np.isnan(df_train["Age"])] = rand_1
df_test["Age"][np.isnan(df_test["Age"])] = rand_2

df_train['Age'] = df_train['Age'].astype(int)
df_test['Age'] = df_test['Age'].astype(int)
