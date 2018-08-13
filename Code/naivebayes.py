# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#%%-----------------------------------------------------------------------
#importing Dataset
# read data as panda dataframe
train_data = pd.read_csv('train_users_final.csv')
test_data = pd.read_csv('test_users_final.csv')

# printing the dataset shape
print("Dataset No. of Rows: ", train_data.shape[0])
print("Dataset No. of Columns: ", train_data.shape[1])

label = train_data['country_destination']
target = train_data['country_destination'].values
class_le = LabelEncoder()
y = class_le.fit_transform(target)
train_data = train_data.drop(['id','country_destination'], axis=1)

id_test = test_data['id']
test_data = test_data.drop(['id'], axis=1)


#%%-----------------------------------------------------------------------
# separate the target variable
X = train_data.values

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

#%%-----------------------------------------------------------------------
# perform training
# creating the classifier object
clf = GaussianNB()
# performing training
clf.fit(X_train, y_train)

#%%-----------------------------------------------------------------------
# make predictions on test from train

# predicton on test split from train
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

#%%-----------------------------------------------------------------------
# calculate metrics
print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")


#%%-----------------------------------------------------------------------
# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = label.unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(7,7))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 10}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()


# predicton on test
y_predt = clf.predict(test_data)
y_pred_scoret = clf.predict_proba(test_data)


ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += class_le.inverse_transform(np.argsort(y_pred_score[i])[::-1])[:5].tolist()

# Generating a csv file with the predictions
sub_nb = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub_nb.to_csv('sub_nb1.csv',index=False)
