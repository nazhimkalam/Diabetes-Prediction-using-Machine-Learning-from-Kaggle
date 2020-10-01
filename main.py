### IMPORTING THE LIBRARIES
import pandas as pd                 # reading data set
import matplotlib.pyplot as plt     # data visualization
import numpy as np      # arrays
import seaborn as sns   # make use of the colourful maps eg:- Heat Map
from sklearn.model_selection import train_test_split        # train-test splitting
from sklearn.impute import SimpleImputer                    # Imputer function
from sklearn.ensemble import RandomForestClassifier         # Algorithm performed
from sklearn import metrics                                 # Calculating Accuracy


### READING THE CSV FILE
data = pd.read_csv("diabetes.csv")

### CHECKING THE DATA FROM THE DATASET/CSV FILE
# print(data.shape)    # gives the shape of the data present (number_of_rows, number_of_columns)
# print(data.head())   # displays the first "n" rows of the data from the dataset
# print(data.columns)  # displays the columns in the dataset.

### CHECKING IF I HAVE NULL VALUES IN MY DATASET
# print(data.isnull().values.any())  # returns False so that means there are no null values

### GETTING CORRELATION (describing the relationship) OF EACH FEATURE IN DATASET
corrMat = data.corr()  # makes the correlation matrix
top_corr_features = corrMat.index  # gets the columns feature names
graph = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")  # plotting the heatmap

plt.figure(figsize=(20, 20))
plt.show()

### GETTING THE CORRELATION VALUES
# print(data.corr())    # returns a table with the data used for the heatmap above

### GETTING THE COUNT OF DIABETES TRUE AND FALSE
print(data.head(3))
diabetes_true_count = len(data.loc[data['Outcome'] == True])        # 1 -> True
diabetes_false_count = len(data.loc[data['Outcome'] == False])      # 0 -> False
print(diabetes_true_count, diabetes_false_count)                    # Displays the count 268 500

### PERFORMING TRAIN-TEST SPLIT
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']  # collecting independent features
predicted_class = ['Outcome']    # collecting dependent features

X = data[feature_columns].values    # independent feature values
y = data[predicted_class].values    # dependent feature values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)     # splitting data into train and test

### CHECKING HOW MANY OTHER MISSING ZERO VALUES
print()
print("Total number of rows : {0}".format(len(data)))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['Glucose'] == 0])))
print("number of rows missing Pregnancies: {0}".format(len(data.loc[data['Pregnancies'] == 0])))
print("number of rows missing BloodPressure: {0}".format(len(data.loc[data['BloodPressure'] == 0])))
print("number of rows missing SkinThickness: {0}".format(len(data.loc[data['SkinThickness'] == 0])))
print("number of rows missing Insulin: {0}".format(len(data.loc[data['Insulin'] == 0])))
print("number of rows missing BMI: {0}".format(len(data.loc[data['BMI'] == 0])))
print("number of rows missing DiabetesPedigreeFunction: {0}".format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
print("number of rows missing Age: {0}".format(len(data.loc[data['Age'] == 0])))


### USING IMPUTER FUNCTION TO FILL THE MISSING VALUES
fill_values = SimpleImputer(missing_values=0, strategy="mean")  # replacing the missing values with the "mean" value for all the rows

X_train = fill_values.fit_transform(X_train)    # the 0 values in the X_train will get replaced with the mean value
X_test = fill_values.fit_transform(X_test)      # the 0 values in the X_train will get replaced with the mean value


### APPLYING ALGORITHM
random_forest_model = RandomForestClassifier(random_state=10)   # using the random forest algo model
random_forest_model.fit(X_train, y_train.ravel())               # training the model using the training data
predict_train_data = random_forest_model.predict(X_test)        # using the test data to predict the result
# print(predict_train_data)

### CALCULATING THE ACCURACY
print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))  # checking the accuracy between the y-test data and the predicted y-data


