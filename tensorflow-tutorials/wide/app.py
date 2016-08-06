import tempfile

import pandas as pd
import tensorflow as tf

# Read CSVs into Pandas dataframes

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

df_train = pd.read_csv("adult.data",
                       names=COLUMNS,
                       skipinitialspace=True)

df_test = pd.read_csv("adult.test",
                      names=COLUMNS,
                      skipinitialspace=True,
                      skiprows=1)

# Construct a label column
LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

# Column types

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

# Functions to convert data into tensors

def input_fn(df):
  # Create mappings from continuous feature columns to constant yensors
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Create mappings from  categorical feature columns sparse tensors
  categorical_cols = {k: tf.SparseTensor(
    indices=[[i, 0] for i in range(df[k].size)],
    values=df[k].values,
    shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merge two dictionaries into one
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Convert label column to constant tensor
  label = tf.constant(df[LABEL_COLUMN].values)
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)

# Base categorical features
gender = tf.contrib.layers.sparse_column_with_keys("gender",
                                                   keys=["female", "male"])
education = tf.contrib.layers.sparse_column_with_hash_bucket("education",
                                                             hash_bucket_size=1000)
race = tf.contrib.layers.sparse_column_with_keys("race",
                                                 keys=["Amer-Indian-Eskimo",
                                                       "Asian-Pac-Islander",
                                                       "Black",
                                                       "Other",
                                                       "White"])
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

# Base continuous features
age = tf.contrib.layers.real_valued_column("age")
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

# Bucketization
age_buckets = tf.contrib.layers.bucketized_column(age,
                                                  boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

# Crossed feature columns
education_x_occupation = tf.contrib.layers.crossed_column([education, occupation],
                                                          hash_bucket_size=int(1e4))
age_buckets_x_race_x_occupation = tf.contrib.layers.crossed_column([age_buckets, race, occupation],
                                                                   hash_bucket_size=int(1e6))

# Defining the model
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[gender,
                                                       native_country,
                                                       education,
                                                       occupation,
                                                       workclass,
                                                       marital_status,
                                                       race,
                                                       age_buckets,
                                                       education_x_occupation,
                                                       age_buckets_x_race_x_occupation],
                                      # optimizer=tf.train.FtrlOptimizer(
                                      #   learning_rate=0.1,
                                      #   l1_regularization_strength=1.0,
                                      #   l2_regularization_strength=1.0),
                                      model_dir=model_dir)

# Training
m.fit(input_fn=train_input_fn, steps=200)

# Evaluation
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
  print "%s: %s" % (key, results[key])