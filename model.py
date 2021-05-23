#!/usr/bin/env python
# coding: utf-8

# # AIR QUALITY PREDICTION FOR SMART CITIES USING MACHINE LEARNING

# # DATA COLLECTION

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[5]:


city = pd.read_csv('new.csv')
city.drop(['AQI_Bucket'],axis=1,inplace=True)


# In[6]:


city.describe().T


# In[7]:


city.info()


# In[8]:


city=city.fillna(0)


# #  Checking for Missing Values in the data
# 

# In[9]:


def getMissingValues(data):
    missing_val = data.isnull().sum()
    missing_val_percentage = 100 * data.isnull().sum() / len(data)
    missin_values_array = pd.concat([missing_val, missing_val_percentage], axis=1)
    missin_values_array = missin_values_array.rename(columns = 
                                                     {0 : 'Missing Values', 1 : '% of Total Values'})
    missin_values_array = missin_values_array[
        missin_values_array.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    return missin_values_array


# In[10]:


def mergeColumns(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['BTX'] = data['Benzene'] + data['Toluene'] + data['Xylene']
    data.drop(['Benzene','Toluene','Xylene'], axis=1)
    data['Particulate_Matter'] = data['PM2.5'] + data['PM10']
    return data


# In[11]:


def subsetColumns(data):
    pollutants = ['Particulate_Matter', 'NO2', 'CO','SO2', 'O3', 'BTX']
    columns =  ['Date', 'City', 'AQI'] + pollutants
    data = data[columns]
    return data, pollutants


# In[12]:


def handleMissingValues(data):
    missing_values = getMissingValues(data)
    updatedCityData = mergeColumns(data)
    updatedCityData, pollutants = subsetColumns(updatedCityData)
    return updatedCityData, pollutants


# In[13]:


updatedCityData, newColumns = handleMissingValues(city)


# In[14]:


updatedCityData.fillna(0)


# In[15]:


def visualisePollutants(udata, columns):
    data = udata.copy()
    data.set_index('Date',inplace=True)
    axes = data[columns].plot(marker='.', linestyle='None', figsize=(15, 15), subplots=True)
    for ax in axes:
        ax.set_xlabel('Years')
        ax.set_ylabel('ug/m3')


# In[16]:


visualisePollutants(city, newColumns)


# In[17]:


updatedCityData.groupby('City')[['Particulate_Matter','NO2','CO','SO2','O3','BTX']].mean()


# # CALCULATION OF AQI INDEX BASED ON THE POLLUTANT LEVELS

# In[18]:


def cal_aqi(SO2,NO2,Particulate_Matter,CO,O3,BTX):
    aqi=0
    if(SO2>NO2 and SO2>Particulate_Matter and SO2>CO and SO2>O3 and SO2>BTX):
     aqi=SO2
    if(NO2>SO2 and NO2>Particulate_Matter and NO2>CO and NO2>O3 and NO2>BTX):
     aqi=NO2
    if(Particulate_Matter>SO2 and Particulate_Matter>NO2 and Particulate_Matter>CO and Particulate_Matter>O3 and Particulate_Matter>BTX ):
     aqi=Particulate_Matter
    if(CO>SO2 and CO>NO2 and CO>Particulate_Matter and CO>O3 and CO>BTX):
     aqi=CO
    if(O3>SO2 and O3>NO2 and O3>Particulate_Matter and O3>CO and O3>BTX):
     aqi=O3
    if(BTX>SO2 and BTX>NO2 and BTX>Particulate_Matter and BTX>O3 and BTX>CO):
     aqi=BTX
    return aqi

updatedCityData['AQI_INDEX']=updatedCityData.apply(lambda x:cal_aqi(x['SO2'],x['NO2'],x['Particulate_Matter'],x['CO'],x['O3'],x['BTX']),axis=1)
city_data=updatedCityData[['City','Date','SO2','NO2','Particulate_Matter','CO','O3','BTX','AQI_INDEX']]
city_data


# # CALCULATION OF AQI_RANGE BASED ON THE AQI INDEX

# In[19]:


def AQI_Range(x):
    if x<=50:
        return "Good"
    elif x>50 and x<=100:
        return "Moderate"
    elif x>100 and x<=150:
        return "Unhealthy for Kids"
    elif x>150 and x<=200:
        return "Unhealthy"
    elif x>200 and x<=300:                                #what is SPMI an dRSPMI SOI why do we need it what is AQI and itsrange
        return "Very Unhealthy"
    elif x>300:
        return "Hazardous"

city_data['AQI_RANGE'] = city_data['AQI_INDEX'] .apply(AQI_Range)
city_data.head(150)


# # Effect of Lockdown on AQI
# 

# # a. AQI in the year 2020 - City-wise

# In[20]:


cities = ['Delhi','Lucknow','Bengaluru','Hyderabad']
filtered_city_day = city_data[city_data['Date'] <= '2020-12-31']
AQI_INDEX = filtered_city_day[filtered_city_day.City.isin(cities)][['Date','City','AQI_INDEX','AQI_RANGE']]


# In[21]:


AQI_pivot = AQI_INDEX.pivot(index='Date', columns='City', values='AQI_INDEX')
AQI_pivot.fillna(method='bfill',inplace=True)


# In[22]:


AQI_2020 = AQI_pivot[AQI_pivot.index <'2020-12-31']
AQI_2020 = AQI_2020.resample('M').mean()
# AQI_2020.set_index('Date')
# aqi = aqi.to_numpy()

def getColorBar(city):
    col = []
    for val in AQI_2020[city]:
        if val < 50:
            col.append('royalblue')
        elif val > 50 and val < 101:
            col.append('lightskyblue') #cornflowerblue
        elif val > 100 and val < 201:
            col.append('lightsteelblue')
        elif val > 200 and val < 301:
            col.append('peachpuff')
        elif val > 300 and val < 401:
            col.append('lightcoral')
        else:
            col.append('firebrick')
    return col

for i in range(0, 4, 2):
    city_1 = cities[i]
    city_2 = cities[i+1]
    fig, ((ax1, ax2)) =  plt.subplots(1, 2, sharex='col', sharey='row', figsize=(15,3))
#     ax = fig.add_axes([0,0,1,1])
    ax1.bar(AQI_2020.index, AQI_2020[city_1], width = 25, color=getColorBar(city_1))
    ax1.title.set_text(city_1)
    ax1.set_ylabel('AQI_INDEX')
    
    colors = {'Good':'royalblue', 'Satisfactory':'lightskyblue', 'Moderate':'lightsteelblue', 'Poor':'peachpuff', 'Very Poor':'lightcoral', 'Severe':'firebrick'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    ax1.legend(handles, labels, loc='upper right')
    
    ax2.bar(AQI_2020.index, AQI_2020[city_2], width = 25, color=getColorBar(city_2))
    ax2.title.set_text(city_2)
    ax2.set_ylabel('AQI_INDEX')
    colors = {'Good':'royalblue', 'Satisfactory':'lightskyblue', 'Moderate':'lightsteelblue', 'Poor':'peachpuff', 'Very Poor':'lightcoral', 'Severe':'firebrick'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    ax2.legend(handles, labels, loc='upper right')
    


# # b.AQI before and after LOCKDOWN

# In[23]:


AQI_beforeLockdown = AQI_pivot['2015-01-01':'2020-03-25']
AQI_afterLockdown = AQI_pivot['2020-03-26':'2020-05-01']
limits = [50, 100, 200, 300, 400, 510]
# palette = sns.light_palette("Spectral", len(limits), reverse = True)
palette = sns.color_palette("coolwarm", len(limits))
for city in cities:
    aqi_before = AQI_beforeLockdown[city].mean()
    aqi_after = AQI_afterLockdown[city].mean()
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(27, 2))
    ax1.set_yticks([1])
    ax1.set_yticklabels([city])
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    prev_limit = 0
    for idx, lim in enumerate(limits):
        ax1.barh([1], lim-prev_limit, left=prev_limit, height=15, color=palette[idx])
        prev_limit = lim

    ax1.barh([1], aqi_before, color='black', height=5)
    
    # after lockdown
    ax2.set_yticks([1])
    ax2.set_yticklabels([city])
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    prev_limit = 0
    for idx, lim in enumerate(limits):
        ax2.barh([1], lim-prev_limit, left=prev_limit, height=15, color=palette[idx])
        prev_limit = lim

    ax2.barh([1], aqi_after, color='black', height=5)
    
    ax1.set_title('Before Lockdown')
    ax2.set_title('After Lockdown')
    
    rects = ax1.patches
    labels=["Good", "Satisfactory", "Moderate", "Poor", 'Very Poor', 'Severe']
    
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax1.text(
            rect.get_x() + rect.get_width()/2 ,
            -height * .4,
            label,
            ha='center',
            va='bottom',
            color='black')
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            -height * .7,
            label,
            ha='center',
            va='bottom',
            color='black')


# # Perform Feature Scaling using Min-Max Scaler
# Min-Max Scaling Feature for normalizing the data to a range of (a-b)
# 

# In[24]:


indep_var=city_data[['SO2','NO2','Particulate_Matter','CO','O3','BTX']]
depend_var= city_data['AQI_RANGE']


# In[25]:


from sklearn.preprocessing import MinMaxScaler
scale1=MinMaxScaler()
Xminmax=scale1.fit_transform(indep_var)
Xminmax


# # PREDICTION OF AIR QUALITY USING SVM,KNN,RANDOM FOREST ALGORITHM

# # RANDOM FOREST ALGORITHM

# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(indep_var,depend_var, test_size=0.33, random_state=42)


# In[28]:


X_train


# In[29]:


y_train


# In[30]:


model1 = RandomForestClassifier()
model1.fit(X_train,y_train)


# In[31]:


prediction = model1.predict(X_test)
prediction


# In[32]:


model1.score(X_test,y_test)


# In[33]:


result=model1.predict([[123,45.6,56,78.9,44,9.5]])
result


# # SVM (SUPPORT VECTOR MACHINE) ALGORITHM

# In[34]:


from sklearn.svm import SVC


# In[35]:


model3=SVC(kernel="rbf",random_state=0)


# In[36]:


model3.fit(X_train,y_train)


# In[37]:


model3.score(X_test,y_test)


# In[38]:


model3.predict([[1.9,5.3,33.3,1.45,7.6,15]])


# # KNN (K-NEAREST NEIGHBOURS) ALGORITHM

# In[39]:


from sklearn.neighbors import KNeighborsClassifier
model2=KNeighborsClassifier()


# In[40]:


model2.fit(X_train,y_train)


# In[41]:


model2.score(X_test,y_test)


# In[42]:


answer = model2.predict([[96.8,31.2,541,205,4.1,7.25]])


# In[43]:


print("The air quality is {}".format(answer[0]))


# In[ ]:





# In[44]:


import pickle

file = open('majorproject.pkl', 'wb')

pickle.dump(model1, file)


# In[45]:


X_test


# In[46]:


X_test.columns


# In[ ]:




