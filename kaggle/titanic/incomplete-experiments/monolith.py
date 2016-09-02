# Based on https://github.com/UltravioletAnalytics/kaggle-titanic

import re

import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

### OPTIONS

np.set_printoptions(precision=4, threshold=10000, linewidth=160, edgeitems=999, suppress=True)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.width', 160)
pandas.set_option('expand_frame_repr', False)
pandas.set_option('precision', 4)

### LOAD & PREP

print "# Reading data from files..."

df_train = pandas.read_csv('data/train.csv', header=0)
df_test = pandas.read_csv('data/test.csv',  header=0)

print "# Merging training and test data for processing..."

df = pandas.concat([df_train, df_test])
df = df.reset_index()
df = df.drop('index', axis=1)
df = df.reindex_axis(df_train.columns, axis=1)  # so we can access the first column at 0 instead of 1

print "# Processing individual variables..."

print "- Processing variable: Name"

print "-- Creating feature: NumNames"

df['NumNames'] = df['Name'].map(lambda x: len(x.split(" ")))

print "-- Creating feature: Title"

df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

# group low-occuring, related titles together
lldf.loc[df['Title'].isin(["Ms", "Mlle"]), 'Title'] = "Miss"
df.loc[df['Title'] == "Mme", 'Title'] = "Mrs"
df.loc[df['Title'].isin(["Capt", "Don", "Major", "Col"]), 'Title'] = "Sir"
df.loc[df['Title'].isin(["Dona", "Lady", "the Countess"]), 'Title'] = "Lady"

df['Title_id'] = pandas.factorize(df['Title'])[0] + 1

print "- Processing variable: Fare"

print "-- Filling blanks with median value..."

df['Fare'].fillna(df['Fare'].median(), inplace=True)

print "-- Getting rid of zeros..."

df.loc[df['Fare'] == 0, 'Fare'] = df['Fare'][df['Fare'].nonzero()[0]].min() / 10

print "- Processing variables: SibSp, Parch"

df['SibSp'] += 1  # interaction variables require no zeros
df['Parch'] += 1  # interaction variables require no zeros

print "- Processing variable: Sex"

df['Sex'] = np.where(df['Sex'] == 'male', 1, 0)

print "- Processing variable: Pclass"

print "-- Filling blanks with mode..."

df['Pclass'].fillna(df['Pclass'].dropna().mode().values[0], inplace=True)

print "- Processing variable: Age"

print "-- Filling blanks using a random forest classifier"

age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Title_id', 'Pclass', 'NumNames']]
X = age_df.loc[(df.Age.notnull())].values[:, 1::]
y = age_df.loc[(df.Age.notnull())].values[:, 0]
rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
rtr.fit(X, y)
predictedAges = rtr.predict(age_df.loc[(df.Age.isnull())].values[:, 1::])
df.loc[(df.Age.isnull()), 'Age'] = predictedAges

print "# Dropping variables: Cabin, Ticket, Embarked, Name, Title"

df = df.drop(['Cabin', 'Ticket', 'Embarked', 'Name', 'Title'], axis=1)

print "# Moving columns around..."
# Move the survived column back to the first position
columns_list = list(df.columns.values)
columns_list.remove('Survived')
new_col_list = list(['Survived'])
new_col_list.extend(columns_list)
df = df.reindex(columns=new_col_list)

# print df.describe()
# print df.info()
# print df