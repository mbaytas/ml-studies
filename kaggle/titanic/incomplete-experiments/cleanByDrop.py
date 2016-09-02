import numpy as np
import pandas

### FEATURE ENGINEERING

def engineer(df, context):

    # Extract 'Title' feature out of 'Name'
    def extract_title(s):
        t = s.split(' ')[1].strip()
        if t[-1] == ".":
           return t
        else:
            return np.nan

    df['Title'] = df['Name'].map( lambda s: extract_title(s) )
    print df['Title'].unique()

    # Convert non-number values to numbers

    def make_num(f, col):
        l = list(context[col].unique())
        map(lambda s: str(s).strip(), l)
        print l
        f[col] = f[col].map( lambda x: l.index(str(x).strip()) if (not str(x) == "nan") else np.nan )
        return f

    df = make_num(df, 'Title')
    df = make_num(df, 'Sex')
    df = make_num(df, 'Embarked')

    # TO DO: Family stuff

    # Drop & burn

    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df = df.dropna()

    ids = df['PassengerId'].values

    try:
        y = df['Survived'].values
        X = df.drop(['PassengerId', 'Survived'], axis=1).values
        return ids, X, y
    except KeyError:
        X = df.drop(['PassengerId'], axis=1).values
        return ids, X

# Execute

df_train = pandas.read_csv('data/train.csv')
df_test = pandas.read_csv('data/test.csv')
context = pandas.read_csv('data/train.csv')

id_train, X_train, y_train = engineer(df_train, context)
id_test, X_test = engineer(df_test, context)


### TRAIN & PREDICT

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

model = model.fit(X_train, y_train)

predictions = model.predict(X_test)

### BURN

df = pandas.DataFrame({"Survived" : predictions, "PassengerId" : id_test})
df.to_csv("output.csv", index=False)