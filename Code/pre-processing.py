#pre-processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train_users = pd.read_csv('train_users_2.csv')

all_users = pd.concat([train_users, test_users], axis=0, ignore_index=True)
all_users.drop('date_first_booking', axis=1, inplace=True)
all_users.info()
all_users['nans'] = np.sum([
    (all_users['age'] == -1),
    (all_users['gender'] == '-unknown-'),
    (all_users['language'] == '-unknown-'),
    (all_users['first_affiliate_tracked'] == 'untracked'),
    (all_users['first_browser'] == '-unknown-')
], axis=0)

nan_percent = (all_users.isnull().sum() / all_users.shape[0]) * 100
nan_percent[nan_percent > 0]
#The result shows that approximate 42% of age attribute are missing value.
#Dealing with age
all_users.age.describe()
users_year_trans_age = all_users.age > 1000
all_users[all_users.age < 18]['age'].describe()
all_users.loc[users_year_trans_age, 'age'] = 2015 - all_users.loc[users_year_trans_age, 'age']
all_users[all_users.age < 18]['age'].describe()
all_users.loc[all_users.age > 100, 'age'] = np.nan
all_users.loc[all_users.age < 18, 'age'] = np.nan
#create age_group feature
bins = [-1, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 100]
all_users['age_group'] = np.digitize(all_users['age'], bins, right=True)
all_users.fillna(-1, inplace=True)
all_users['date_account_created'] = pd.to_datetime(all_users['date_account_created'], errors='ignore')
all_users['date_first_active'] = pd.to_datetime(all_users['timestamp_first_active'], format='%Y%m%d%H%M%S')
date_account_created = pd.DatetimeIndex(all_users['date_account_created'])
date_first_active = pd.DatetimeIndex(all_users['date_first_active'])
all_users['day_account_created'] = date_account_created.day
all_users['weekday_account_created'] = date_account_created.weekday
all_users['week_account_created'] = date_account_created.week
all_users['month_account_created'] = date_account_created.month
all_users['year_account_created'] = date_account_created.year
all_users['day_first_active'] = date_first_active.day
all_users['weekday_first_active'] = date_first_active.weekday
all_users['week_first_active'] = date_first_active.week
all_users['month_first_active'] = date_first_active.month
all_users['year_first_active'] = date_first_active.year
#create time_lag feature
all_users['time_lag'] = (date_first_active.values - date_account_created.values).astype(int)
all_users.info()
drop_list = [
    'date_account_created',
    'date_first_active',
    'timestamp_first_active'
]

all_users.drop(drop_list, axis=1, inplace=True)
all_users.head()
categorical_features = [
    'gender', 'signup_method', 'signup_flow', 'language',
    'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
    'signup_app', 'first_device_type', 'first_browser'
]
all_users = pd.get_dummies(all_users, columns=categorical_features)
all_users.set_index('id', inplace=True)
all_users.loc[train_users['id']].to_csv('train_users_final.csv')
all_users.loc[test_users['id']].drop('country_destination', axis=1).to_csv('test_users_final.csv')
