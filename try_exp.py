import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import folium

# import data
train_data = pd.read_csv('train_users_2.csv')
test_data = pd.read_csv('test_users.csv')
countries = pd.read_csv('countries.csv')

sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_style("whitegrid", {'axes.edgecolor': '0'})
sns.set_context("poster", font_scale=1.1)


print("We have", train_data.shape[0], "users in the training set and",
      test_data.shape[0], "in the test set.")
print("In total we have", train_data.shape[0] + test_data.shape[0], "users.")


# Remove 'id' feature
train_data.drop('id', axis=1, inplace=True)
train_data.head()
print(train_data.isnull().sum())

#------------------------ Missing value --------------------------
train_data.replace('-unknown-', np.nan, inplace=True)
print(train_data.isnull().sum())

train_nan = (train_data.isnull().sum() / train_data.shape[0]) * 100
train_nan1 = train_nan[train_nan > 0]

print("After check for the missing value, we find that date_first_booking have",
      int((train_data.date_first_booking.isnull().sum() / train_data.shape[0]) * 100),
      "% of missing values in the training data")

width=0.4
train_nan1.plot(kind='bar',width=width, color='c',position=0, label= 'features', rot=0)
plt.xlabel('features')
plt.ylabel('percentage of missing')
plt.show()
plt.clf()

#------------------------ Gender ---------------------------
train_data.gender.value_counts(dropna=False).plot(kind='bar', color='#7D9EC0', rot=0)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
plt.clf()

women = sum(train_data['gender'] == 'FEMALE')
men = sum(train_data['gender'] == 'MALE')
female_destinations =train_data.loc[train_data['gender'] == 'FEMALE', 'country_destination'].value_counts() / women * 100
male_destinations = train_data.loc[train_data['gender'] == 'MALE', 'country_destination'].value_counts() / men * 100
# Bar width
width = 0.4
male_destinations.plot(kind='bar', width=width, color='#96CDCD', position=0, label='Male', rot=0)
female_destinations.plot(kind='bar', width=width, color='#FFA38D', position=1, label='Female', rot=0)
plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
sns.despine()
plt.show()
plt.clf()

#------------------------- Age --------------------------
train_data.age.describe()
print('After checking for age, we find that max of age feature ', train_data.age.max(), 'is year number instead of age number')

# Airbnb only allow 18 or older to create an account.
print(sum(train_data.age > 100))
print(sum(train_data.age < 18))

train_data[train_data.age > 100]['age'].describe()
train_data[train_data.age < 18]['age'].describe()

user_with_year_age_mask = train_data['age'] > 1000
train_data.loc[user_with_year_age_mask, 'age'] = 2015 - train_data.loc[user_with_year_age_mask, 'age']
train_data.loc[(train_data['age'] > 100) | (train_data['age'] < 18), 'age'] = np.nan

age_dist= sns.distplot(train_data.age.dropna(), color='#458B74')
plt.xlabel('Age')
sns.despine()
plt.clf()

data_nN= train_data[train_data['country_destination']!='NDF']
data_nNU = data_nN[data_nN['country_destination']!='US']
sns.boxplot(x='country_destination',y='age' , data=train_data)
plt.ylim(10,70)
plt.xlabel('Destination Country')
plt.ylabel('Age of Users')
sns.despine()
plt.clf()

#-------------------------------language--------------------------------
language_pre = train_data.language.value_counts() / train_data.shape[0] * 100
print('Percentage of users using English is ', language_pre[0].round(2),'%.')

#-----------------------------destination-------------------------------
destinations = train_data.country_destination.value_counts() / train_data.shape[0] * 100
print('Percentage of users have not book yet is ', destinations[0].round(2),
      '%. Percentage of users chose the US as their first destination is ', destinations[1].round(2), '%.')


sns.countplot(x="country_destination", data=train_data, order=list(train_data.country_destination.value_counts().keys()))
plt.xlabel('Destination Country')
plt.ylabel('Count')
sns.despine()
plt.clf()

sns.countplot(x="country_destination", data=data_nNU, order=list(data_nNU.country_destination.value_counts().keys()))
plt.xlabel('Destination Country')
plt.ylabel('Count')
sns.despine()
plt.clf()

destinations1 = train_data.country_destination.value_counts()
dest_f = pd.DataFrame({'country_destination':destinations1.index,'count':destinations1.values})
dest = pd.merge(dest_f, countries, on=('country_destination'))

m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)
for i in range(0, len(dest)):
    folium.Marker([dest.iloc[i]['lat_destination'], dest.iloc[i]['lng_destination']], popup=str(dest.iloc[i]['count'])).add_to(m)
m.save('lx_markers_on_folium_map1.html')


# ------------------------- Date features ---------------------------------
train_data['date_account_created'] = pd.to_datetime(train_data['date_account_created'])
train_data.date_account_created.value_counts().plot(kind='line', linewidth=1, color='#F4D03F')
plt.xlabel('account created date')
plt.show()
plt.clf()

train_data['date_first_active'] = pd.to_datetime(train_data['timestamp_first_active'], format='%Y%m%d%H%M%S')
date_first_active = train_data.date_first_active.apply(lambda x: datetime.datetime(x.year, x.month, x.day))
date_first_active.value_counts().plot(kind='line', linewidth=1, color='#BCEE68')
plt.xlabel('first active date')
plt.show()
plt.clf()


# -------------------------- categorical features ------------------------------
print((train_data.signup_method.value_counts() / train_data.shape[0]) * 100)
train_data.signup_method.value_counts().plot(kind='bar', rot=0, color='#9AC0CD')
plt.xlabel('sign up methods')
plt.ylabel('count')
plt.show()
plt.clf()

print((train_data.signup_flow.value_counts() / train_data.shape[0]) * 100)
train_data.signup_flow.value_counts().plot(kind='bar', rot=0, color='#9AC0CD')
plt.xlabel('sign up flow')
plt.ylabel('count')
plt.show()
plt.clf()

print((train_data.signup_app.value_counts() / train_data.shape[0]) * 100)
train_data.signup_app.value_counts().plot(kind='bar', rot=0, color='#9AC0CD')
plt.xlabel('sign up app')
plt.ylabel('count')
plt.show()
plt.clf()

print((train_data.affiliate_channel.value_counts() / train_data.shape[0]) * 100)
train_data.affiliate_channel.value_counts().plot(kind='bar', rot=0, color='#9AC0CD')
plt.xlabel('affiliate channel')
plt.ylabel('count')
plt.show()
plt.clf()

print((train_data.affiliate_provider.value_counts() / train_data.shape[0]) * 100)
train_data.affiliate_provider.value_counts().plot(kind='bar', rot=25, fontsize=10, color='#9AC0CD')
plt.xlabel('affiliate provider')
plt.ylabel('count')
plt.show()
plt.clf()

print((train_data.first_affiliate_tracked.value_counts() / train_data.shape[0]) * 100)
train_data.first_affiliate_tracked.value_counts().plot(kind='bar', rot=0, color='#9AC0CD')
plt.xlabel('first affiliate tracked')
plt.ylabel('count')
sns.despine()
plt.show()
plt.clf()

print((train_data.first_device_type.value_counts() / train_data.shape[0]) * 100)
train_data.first_device_type.value_counts().plot(kind='bar', rot=25, color='#9AC0CD')
plt.xlabel('first device type')
plt.ylabel('count')
sns.despine()
plt.show()
plt.clf()

print((train_data.first_browser.value_counts() / train_data.shape[0]) * 100)
train_data.first_browser.value_counts().plot(kind='bar', fontsize=10, color='#9AC0CD')
plt.xlabel('first browser')
plt.ylabel('count')
plt.show()
plt.clf()




