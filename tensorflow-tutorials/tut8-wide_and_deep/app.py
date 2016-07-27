import tempfile

import pandas as pd
import tensorflow as tf

# Categorical base features
gender = tf.contrib.layers.sparse_column_with_keys("gender",
                                                   keys=["female", "male"])
race = tf.contrib.layers.sparse_column_with_keys("race",
                                                 keys=["Amer-Indian-Eskimo",
                                                       "Asian-Pac-Islander",
                                                       "Black",
                                                       "Other",
                                                       "White"])
education = tf.contrib.layers.sparse_column_with_hash_bucket("education",
                                                            hash_bucket_size=1000)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status",
                                                                  hash_bucket_size=100)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship",
                                                                hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass",
                                                             hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation",
                                                              hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country",
                                                                  hash_bucket_size=1000)

# Continuous base features
age = tf.contrib.layers.real_valued_column("age")
age_buckets = tf.contrib.layers.bucketized_column(age,
                                                  boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

# Wide features
wide_columns = [gender,
                native_country,
                education,
                occupation,
                workclass,
                marital_status,
                relationship,
                age_buckets,
                tf.contrib.layers.crossed_column([education, occupation],
                                                 hash_bucket_size=int(1e4)),
                tf.contrib.layers.crossed_column([native_country, occupation],
                                                 hash_bucket_size=int(1e4)),
                tf.contrib.layers.crossed_column([age_buckets, race, occupation],
                                                 hash_bucket_size=int(1e6))]

# Deep features
deep_columns = [tf.contrib.layers.embedding_column(workclass, dimension=8),
                tf.contrib.layers.embedding_column(education, dimension=8),
                tf.contrib.layers.embedding_column(marital_status, dimension=8),
                tf.contrib.layers.embedding_column(gender, dimension=8),
                tf.contrib.layers.embedding_column(relationship, dimension=8),
                tf.contrib.layers.embedding_column(race, dimension=8),
                tf.contrib.layers.embedding_column(native_country, dimension=8),
                tf.contrib.layers.embedding_column(occupation, dimension=8),
                age,
                education_num,
                capital_gain,
                capital_loss,
                hours_per_week]

# Defining the model
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.DNNLinearCombinedClassifier(model_dir=model_dir,
                                                 linear_feature_columns=wide_columns,
                                                 dnn_feature_columns=deep_columns,
                                                 dnn_hidden_units=[100, 50])

# Read in the data

COLUMNS = ["age",
           "workclass",
           "fnlwgt",
           "education",
           "education_num",
           "marital_status",
           "occupation",
           "relationship",
           "race",
           "gender",
           "capital_gain",
           "capital_loss",
           "hours_per_week",
           "native_country",
           "income_bracket"]
LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = ["workclass",
                       "education",
                       "marital_status",
                       "occupation",
                       "relationship",
                       "race",
                       "gender",
                       "native_country"]
CONTINUOUS_COLUMNS = ["age",
                      "education_num",
                      "capital_gain",
                      "capital_loss",
                      "hours_per_week"]

df_train = pd.read_csv("adult.data",
                       names=COLUMNS,
                       skipinitialspace=True)
df_test = pd.read_csv("adult.test",
                      names=COLUMNS,
                      skipinitialspace=True,
                      skiprows=1)
df_train[LABEL_COLUMN] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)

def input_fn(df):
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  label = tf.constant(df[LABEL_COLUMN].values)
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)

# Train
m.fit(input_fn=train_input_fn, steps=200)

# Evaluate
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
  print "%s: %s" % (key, results[key])