# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: JANANI S
RegisterNumber:  212223230086
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/c0c96962-7909-4430-a55a-fea62a769e04)
```
data.info()
data.isnull().sum()
data["left"].value_counts()
```
![image](https://github.com/user-attachments/assets/135beedb-ebd9-4bb6-a066-f2202449223a)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
![image](https://github.com/user-attachments/assets/f1b7dc83-d71b-40b2-926d-c2d308acb7bc)
```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
![image](https://github.com/user-attachments/assets/3f7465ad-4788-4011-b7e3-af97a61ddb77)
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/499f958f-b169-424c-89cc-dea4b6e84826)
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/75cfbc9d-a330-4f80-8de7-79757a8c54fe)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
