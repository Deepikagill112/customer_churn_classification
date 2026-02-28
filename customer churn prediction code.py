

**1. Importing the dependencies**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
import pickle

"""**2. Data Loading and understanding**

"""

#load csv data to a pandas dataframe
df = pd.read_csv("/content/sample_data/Telco-Customer-Churn dataset.csv")

df.shape

df.head()

pd.set_option("display.max_columns", None)

df.head(4)

df.info()

# Based on loaded_features, 'customerID' was part of the training data. So, we should not drop it.
# dropping customerID column as this is not required for modelling
# df = df.drop(columns=["customerID"])
# We'll keep customerID for encoding as per loaded_features.

df.head(4)

print(df["gender"].unique())

print(df["SeniorCitizen"].unique())

df.columns

# printing the unique values in all columns
for col in df.columns:
  print(col,df[col].unique())
numerical_features_list = ["tenure", "MonthlyCharges","TotalCharges"]

for col in df.columns:
  if col not in numerical_features_list:
    print(col,df[col].unique)
  print(col,df[col].unique)
  print("-" * 50)

print(df.isnull().sum())

df.isnull().sum()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df[df["TotalCharges"].isnull()]

len(df[df["TotalCharges"].isnull()])

df["TotalCharges"] =df["TotalCharges"].replace({" ":"0.0"})

df["TotalCharges"] =df["TotalCharges"].astype(float)

df.info()

# Fill NaN values in 'TotalCharges' with 0, as these correspond to customers with 0 tenure.
df['TotalCharges'].fillna(0, inplace=True)
#checking the distribution of target column
print(df["Churn"].value_counts())

"""**insights:**
1. Customer ID removed as it is not required for modelling
2. No missing values in the dataset
3. Missing values in the totalCharges column were replaced with 0
4. Class imbalance identified in the target

**3. Exploratory Dats Analysis(EDA)**
"""

df.shape

df.columns

df.head(4)

df.describe()

"""**Numerical Features - Analysis**

Understand the distribution of teh numerical features
"""

def plot_histogram(df,column_name):

    plt.figure(figsize=(6,3))
    sns.histplot(data=df,x=column_name,kde=True)
    plt.title(f"Distribution of {column_name}")
    plt.xlabel(column_name)
    #calculate the mean and median values for the columns
    col_mean = df[column_name].mean()
    col_median = df[column_name].median()
    # add vertical lines for mean and median
    plt.axvline(col_mean,color="red",linestyle="--",label="Mean")
    plt.axvline(col_median,color="green",linestyle="--",label="Median")

    plt.legend()
    plt.show()

plot_histogram(df, "tenure")

plot_histogram(df, "MonthlyCharges")

plot_histogram(df,"TotalCharges")

"""**Box plot for numerical features**"""

def plot_boxplot(df,column_name):
  plt.figure(figsize=(6,3))
  sns.boxplot(data=df,x=column_name)
  plt.title(f"Box Plot of {column_name}")
  plt.xlabel(column_name)
  plt.show

plot_boxplot(df,"tenure")

plot_boxplot(df,"MonthlyCharges")

plot_boxplot(df,"TotalCharges")

"""**correlation Heatmap for numerical columns**"""

## correlation matrix = heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df[["tenure","MonthlyCharges","TotalCharges"]].corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

"""**Categorial features -Analysis**"""

df.columns

df.info()

"""**Countplot for Categorical columns**"""

object_cols = df.select_dtypes(include="object").columns.to_list()
object_cols = ["SeniorCitizen"] + object_cols
for col in object_cols:
  plt.figure(figsize=(6,3))
  sns.countplot(data=df,x=col)
  plt.title(f"Count Plot of {col}")
  plt.show()

"""**4. Data Preprocessing**"""

df.head(4)

"""Label encoding of target column"""

df['Churn'] = df['Churn'].replace({'Yes':1, 'No':0})

df.head(4)

print(df["Churn"].value_counts())

"""Label encoding of categorial features"""

# identifying columns with object data type
object_columns = df.select_dtypes(include="object").columns.to_list()

print(object_columns)

# identifying columns with object data type
object_columns = df.select_dtypes(include="object").columns.to_list()

# Exclude 'TotalCharges' from object columns if present, as it should be numerical.
# TotalCharges was already handled as numerical, so it shouldn't be in object_columns now.
if 'TotalCharges' in object_columns:
    object_columns.remove('TotalCharges')

print(f"Object columns identified for encoding: {object_columns}")

# initialize a dictionary to save the encoders
encoders = {}
print(f"Encoders initialized: {encoders}")

# apply label encoding and store the encoders
for col_name in object_columns:
  label_encoder = LabelEncoder()
  df[col_name] = label_encoder.fit_transform(df[col_name])
  encoders[col_name] = label_encoder

print(f"Encoders after loop: {encoders}")

# save the encoders to a pickle file
with open("encoders.pkl","wb") as f:
  pickle.dump(encoders,f)

encoders

df.head()

"""**Training and test data split**"""

# splitting the features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

print(X)

# split training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

print(y_train.shape)

print(y_train.value_counts())

"""**Synthetic Minority Oversampling Technique(SMOTE)**"""

smote = SMOTE(random_state = 42)

from imblearn.over_sampling import SMOTE

# ensure numeric + no NaN
X_train = X_train.astype(float)
X_train = X_train.fillna(0)

smote = SMOTE(random_state=42, k_neighbors=2)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("SMOTE applied successfully ")

print(y_train_smote.shape)

print(y_train_smote.value_counts())

"""**5.MODEL TRAINING**

Training with default hyperparameters
"""

## dictionary of models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBRFClassifier()
}

# dictionary to store the cross validation results
CV_scores = {}
# perform 5-fold cross Validation for each model
for model_name, model_obj in models.items():
  print(f"Training {model_name} with default parameters ")
  scores = cross_val_score(model_obj,X_train_smote,y_train_smote,cv=5,scoring="accuracy")
  CV_scores[model_name] = scores
  print(f"{model_name} Cross-validation accuracy: {np.mean(scores):.2f}")
  print("-" * 70)

CV_scores

"""**Random Forest gives the higest accuracy compared to other model with default parameters**"""

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote,y_train_smote)

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_smote, y_train_smote)

print(y_test.value_counts())

"""**6.MODEL EVALUATION**"""

# suggested code may be subject to a licence|wongngaisun/ai-crypto-trading
# evaluate on test data
y_test_pred = rfc.predict(X_test)

print("Accuracy Score:\n", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

X.columns.tolist()

#save the trained model as a pickle file
model_data = {"model": rfc,"features_names": X.columns.tolist()}
with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

"""**7.LOAD THE SAVED MODEL AND BUILD A PREDICTIVE SYSTEM**"""

# load teh saved model and the encoders
with open("/content/model.pkl","rb") as f:
  model_data = pickle.load(f)
  loaded_model = model_data["model"]
  loaded_features = model_data["features_names"]

print(loaded_model)

print(loaded_features)

import pandas as pd
import io

# Sample data as a string. Correcting the incomplete 'TotalCharges' value.
sample_data_str = """
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
"""

# Read the sample data into a DataFrame
sample_df = pd.read_csv(io.StringIO(sample_data_str))

# Display the loaded sample data
print(sample_df)

# This sample_df will then need to be preprocessed (e.g., label encoded)
# in the same way as the training data before being passed to the loaded_model for prediction.

input_data = {
    'customerID' : '7590-VHVEG',
    'gender' : 'Female',
    'SeniorCitizen' : 0,
    'Partner' : 'Yes',
    'Dependents' : 'No',
    'tenure' : 1,
    'PhoneService' : 'No',
    'MultipleLines' : 'No phone service',
    'InternetService' : 'DSL',
    'OnlineSecurity' : 'No',
    'OnlineBackup' : 'Yes',
    'DeviceProtection' : 'No',
    'TechSupport' : 'No',
    'StreamingTV' : 'No',
    'StreamingMovies' : 'No',
    'Contract' : 'Month-to-month',
    'PaperlessBilling' : 'Yes',
    'PaymentMethod' : 'Electronic check',
    'MonthlyCharges' : 29.85,
    'TotalCharges' : 29.85,
}

input_data_df = pd.DataFrame([input_data])
with open("encoders.pkl","rb")as f:
  encoders = pickle.load(f)

  # encode categorical features using teh saved encoders
  for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

# make a prediction
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

print(prediction)

# results
print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediction Probability: {pred_prob}")

encoders
