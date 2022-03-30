

#importing the libraries
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib.style as stl
import seaborn as sns
import sklearn as skl
import pickle

data=pd.read_csv("dataset.csv")


#filling the missing values using median(Right skewed data)
filler = data["Data_Value"].median()
data["Data_Value"] = data["Data_Value"].fillna(filler)

filler = data["Confidence_Limit_High"].median()
data["Confidence_Limit_High"] = data["Confidence_Limit_High"].fillna(filler)

filler = data["Confidence_Limit_Low"].median()
data["Confidence_Limit_Low"] = data["Confidence_Limit_Low"].fillna(filler)

#Dropping Unwanted columns
data.drop("Data_Value_Footnote_Symbol", axis=1,inplace=True)
data.drop("Data_Value_Footnote", axis=1,inplace=True)
data.drop("GeoLocation", axis=1,inplace=True)



"""2. Outlier removal """



for i in ['Data_Value_Alt','Data_Value','Confidence_Limit_Low', 'Confidence_Limit_High']:
#finding quartile values    
  Q1=np.percentile(data[i],25,interpolation='midpoint')
  Q2=np.percentile(data[i],50,interpolation='midpoint')
  Q3=np.percentile(data[i],75,interpolation='midpoint')
#finding the IQR    
  IQR=Q3-Q1    
#finding upper and lowe limits    
  lower_limit=Q1-1.5*IQR
  upper_limit=Q3+1.5*IQR
#finding outliers
  data.loc[data[i]<lower_limit, i] = lower_limit
  data.loc[data[i]>upper_limit, i] = upper_limit


"""### 3. Encoding
##### 3.1 Label encoding
"""

#import label encoder
from sklearn.preprocessing import LabelEncoder
#creating an instance LabelEncoder
label_en =LabelEncoder()
a=["Topic","Category","Indicator","Break_out","LocationDesc"]
for i in np.arange(len(a)):
    data[a[i]]=label_en.fit_transform(data[a[i]])


"""##### 3.2 One hot encoding"""

#import one hot encoder
from sklearn.preprocessing import OneHotEncoder
data= pd.get_dummies(data, columns = ['Data_Value_Type','Break_Out_Category'])

"""### 4. Feature Reduction"""

#Dropping unwanted columns
b=["LocationAbbr",'CategoryID', 'TopicID','IndicatorID', 'Data_Value_TypeID', 'BreakoutCategoryID', 'BreakOutID','LocationID','Data_Value_Unit','Datasource']
for i in np.arange(len(b)):
    data.drop(b[i], axis=1,inplace=True)



"""### 5. Feature Enigineering"""

#Create dummies for PriorityArea1
p1=pd.get_dummies(data['PriorityArea1'])
#Create dummies for PriorityArea2
p2=pd.get_dummies(data['PriorityArea2'])
#Create dummies for PriorityArea3
p3=pd.get_dummies(data['PriorityArea3'])
#Create dummies for PriorityArea4
p4=pd.get_dummies(data['PriorityArea4'])
#Concatenating PriorityAreas
data=pd.concat([data,p1,p2,p3,p4],axis=1)

data["Million Hearts"]=data["Million Hearts"]*data["PriorityArea1"]
data["ABCS"]=data["ABCS"]*data["PriorityArea2"]
data["Healthy People 2020"]=data["Healthy People 2020"]*data["PriorityArea3"]

#create new column "PriorityArea"
data["PriorityArea"]=data["Million Hearts"]+data["ABCS"]+data["Healthy People 2020"]

#removing all columns named None
data=data.drop(['None'],axis=1)

#removing unwanted columns
data=data.drop(["Million Hearts","ABCS","Healthy People 2020","PriorityArea1","PriorityArea2","PriorityArea3","PriorityArea4"],axis=1)

#label encoding new PriorityArea
data["PriorityArea"]=label_en.fit_transform(data["PriorityArea"])


"""### 6. Standardization"""

y=data["Category"]
x=data.drop(["Category"],axis=1)

x=x.drop(['Data_Value_Type_Age-Standardized','Data_Value_Type_Crude', 'Break_Out_Category_Age',
       'Break_Out_Category_Gender', 'Break_Out_Category_Overall',
       'Break_Out_Category_Race'],axis=1)
print(x.columns)

"""#Modelling
 Implementing KNN"""

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 12, weights = 'uniform',algorithm = 'brute',metric = 'manhattan')
knn.fit(x, y)

#Saving the model to disk
pickle.dump(knn,open('model.pkl','wb') )

      