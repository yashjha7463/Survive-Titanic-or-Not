import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the DataFrame#


train_df = pd.read_csv("train.csv")
print(train_df)


# Manipulate data


""" We need to perform the following operations on our data before our logistic regression model can use it:

 Assign a numerical value to the feature ‘Sex’.

 Use one-hot encoding on the feature ‘Pclass’.

 Fill in the missing values in the age column.

 Only select the required features.

 We’ll define a function to transform our data to make it usable for our logistic regression model. """

def manipulate_df(df):
    # Update sex column to numerical
    df['Sex'] = df['Sex'].map(lambda x: 0 if x == 'male' else 1)
    # Fill the nan values in the age column
    df['Age'].fillna(value = df['Age'].mean() , inplace = True)
    # Create a first class column
    df['FirstClass'] = df['Pclass'].map(lambda x: 1 if x == 1 else 0)
    # Create a second class column
    df['SecondClass'] = df['Pclass'].map(lambda x: 1 if x == 2 else 0)
    # Create a second class column
    df['ThirdClass'] = df['Pclass'].map(lambda x: 1 if x == 3 else 0)
    # Select the desired features
    df= df[['Sex' , 'Age' , 'FirstClass', 'SecondClass' ,'ThirdClass' , 'Survived']]
    return df

manipulated_df = manipulate_df(train_df)

print(manipulated_df)


# Train-test split
""" We’ll split the dataset into two parts, 70% of it for training and the remaining 30% for testing. """

features= train_df[['Sex' , 'Age' , 'FirstClass', 'SecondClass','ThirdClass']]
survival = train_df['Survived']
X_train , X_test , y_train , y_test = train_test_split(features , survival ,test_size = 0.3)


# Scale the feature data

""" We need to scale the data. To do that, we’ll set mean = 0 and standard deviation = 1 """
# Import the necessary library
from sklearn.preprocessing import StandardScaler

# Create an instance of the StandardScaler
scaler = StandardScaler()

# Transform the training features using the fit_transform method of the StandardScaler
train_features = scaler.fit_transform(X_train)

# Transform the test features using the transform method of the StandardScaler
test_features = scaler.transform(X_test)

# Print the transformed test features
print(test_features)

#  Build the model

""" We’ll use the LogisticRegression class to create a model. """

# Create and train the model
model = LogisticRegression()
model.fit(train_features , y_train)
train_score = model.score(train_features,y_train)
test_score = model.score(test_features,y_test)
y_predict = model.predict(test_features)
print("Training Score: ",train_score)
print("Testing Score: ",test_score)
print("Predicted values: ",y_predict)

