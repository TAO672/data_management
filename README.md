# data_management

##  Data Cleaning
-- Step 1:1: Make sure the data has been loaded into Hive correct
SELECT * FROM wine LIMIT 10;

-- Step 2: Data Cleaning
-- 2.1 Check if each column has missing values
SELECT COUNT(*) - COUNT(Alcohol) AS missing_Alcohol,
       COUNT(*) - COUNT(Malicacid) AS missing_Malicacid,
       COUNT(*) - COUNT(Ash) AS missing_Ash
FROM wine;

-- 2.2 Delete duplicate data
CREATE TABLE wine_cleaned AS 
SELECT DISTINCT * FROM wine;

## Data Analysis and Modeling
-- Step 1: Target variable analysis
SELECT class, COUNT(*) 
FROM wine
GROUP BY class;

-- Step 2: Calculate statistics
SELECT AVG(Alcohol) AS avg_Alcohol, 
       MAX(Alcohol) AS max_Alcohol,
       MIN(Alcohol) AS min_Alcohol,
       AVG(Malicacid) AS avg_Malicacid,
       MAX(Malicacid) AS max_Malicacid,
       MIN(Malicacid) AS min_Malicacid,
FROM wine;

-- Continue with statistics of other features
## Data Visualization (using Python)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the file
data = pd.read_csv("C:\\Users\\PC 30\\Desktop\\000000_0.csv", header=None)

# Set column names
data.columns = ['class', 'Alcohol', 'Malicacid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',  'Total_phenols', 'Flavonoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue','OD280_OD315_of_diluted_wines',
'Proline']


# Visualize the distribution of the target variable class
plt.figure(figsize=(6,4))
sns.countplot(x='class', data=data)
plt.title('Distribution of Classes')
plt.show()
[!chart1](pic/1.png)

# Visualize the relationship between Alcohol and Malicacid
plt.figure(figsize=(8,6))
sns.scatterplot(x='Alcohol', y='Malicacid', hue='class', data=data, palette="Set1")
plt.title('Alcohol vs Malicacid')
plt.show()
[!chart1](pic/2.png)

# Calculate and display the correlation heatmap between features
plt.figure(figsize=(10,8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
[!chart1](pic/3.png)

sns.pairplot(data, hue='class', vars=['Alcohol', 'Malicacid', 'Ash', 'Magnesium'])
plt.suptitle('Pairplot of Features by Class', y=1.02)
plt.show()
[!chart1](pic/4.png)

plt.figure(figsize=(8,6))
sns.violinplot(x='class', y='Alcohol', data=data)
plt.title('Alcohol Distribution by Class')
plt.show()

plt.figure(figsize=(8,6))
sns.violinplot(x='class', y='Malicacid', data=data)
plt.title('Malicacid Distribution by Class')
plt.show()
[!chart1](pic/5.png)
[!chart1](pic/6.png)

## Model Building and Evaluation
#In this section, we will use Random Forest Classifier to build and evaluate the model. Random Forest is a powerful ensemble learning method that performs classification by voting on multiple decision trees.

#1. Data Preprocessing
#First, the dataset needs to be split into a training set and a test set. Usually, we use 70% of the data for training and 30% for testing.

#2. Build a Random Forest Model
#The main advantage of the Random Forest model is that it can automatically handle a large number of features and can effectively reduce overfitting. We will use RandomForestClassifier to build a classification model.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Data preparation
# Features and target variables
X = data.drop('class', axis=1)  # 选择特征（除去目标变量 'class'）
y = data['class']  # 目标变量 'class'

# Split the dataset (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluate model performance
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))


 ## Evaluate the performance of the model
Accuracy: Shows the accuracy of the model on the test set, a simple but effective evaluation indicator.

Precision: Indicates the proportion of samples predicted as positive by the model that are actually positive. High precision means fewer false positives.

Recall: Indicates the proportion of samples successfully predicted as positive by the model among all samples that are actually positive. High recall means that the model has fewer false negatives.

F1 score: The harmonic mean of precision and recall, used to measure the balanced performance of the model.


 ## Feature Importance
#The Random Forest model can view the importance of each feature through the feature_importances_ attribute. This helps understand which features have the greatest impact on the model's predictions.

# Check the importance of each feature
importances = model.feature_importances_
features = X.columns

# Create a data frame and sort by importance
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()
[!chart1](pic/7.png)
