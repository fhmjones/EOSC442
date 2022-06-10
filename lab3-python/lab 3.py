#lab 3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime as dt
import numpy as np
import scipy.stats

#converting floating point dates to date time, source:
#https://notebook.community/jonathanrocher/pandas_tutorial/climate_timeseries/climate_timeseries-Part2
import calendar
# Let's first convert the floating point dates in the sea level to timestamps:
def floating_year_to_timestamp(float_date):
    """ Convert a date as a floating point year number to a pandas timestamp object.
    """
    year = int(float_date)
    days_per_year = 366 if calendar.isleap(year) else 365
    remainder = float_date - year
    daynum = 1 + remainder * (days_per_year - 1)
    daynum = int(round(daynum))
    # Convert day number to month and day
    day = daynum
    month = 1
    while month < 13:
        month_days = calendar.monthrange(year, month)[1]
        if day <= month_days:
            return pd.Timestamp(str(year)+"/"+str(month)+"/"+str(day))
        day -= month_days
        month += 1
    raise ValueError('{} does not have {} days'.format(year, daynum))

#%%Part 1 Loading data and plotting time series
lab3_data = pd.read_excel(r"lab3_data.xlsx",index_col=0,parse_dates=True,date_parser=(floating_year_to_timestamp)) #nice way to parse dates
lab3_data = lab3_data.dropna()

print(lab3_data)
print(lab3_data.head(0))


#getting all titles of colums for plots
titles=lab3_data.columns.values.tolist()


 
#plotting each time series in separate subplots on the same fig1ure
'''notes: 
    - figsize is important for making all data readable with so many subplots
and long titles. 
    - linear regression cant be done with datetime. 
use toordinal to translate to integer and fromordinal to translate back

'''



fig1 = plt.figure(figsize=[25,25])
ax11 = fig1.add_subplot(421)
ax12 = fig1.add_subplot(422)
ax13 = fig1.add_subplot(423)
ax14 = fig1.add_subplot(424)
ax15 = fig1.add_subplot(425)
ax16 = fig1.add_subplot(426)
ax17 = fig1.add_subplot(427)
lab3_data.plot(subplots=True,legend=False,title=titles,ax=[ax11,ax12,ax13,ax14,ax15,ax16,ax17]) 





#Linear regression
ubc_temp = lab3_data[r'UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)']                                      
model1 = LinearRegression()
lab3_data.index = lab3_data.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model1.fit(dates_reshape,ubc_temp) #Performing the linear regression
ubc_prediction = model1.predict(dates_reshape) #Calculating the trendline
lab3_data.index = lab3_data.index.map(dt.datetime.fromordinal)
ax11_slope = model1.coef_*12 # C/yr
ax11_score = model1.score(dates_reshape,ubc_temp) # R^2 value
'''ax11_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax11.plot(lab3_data.index.values,ubc_prediction,'k')
ax11.text(1,5,'Slope = '+ str(np.round(ax11_slope,4)) +" C/yr and R^2 = "+str(np.round(ax11_score,4))+"; 95% conf tbd")




global_temp = lab3_data[r'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)']
model2 = LinearRegression()
lab3_data.index = lab3_data.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model2.fit(dates_reshape,global_temp) #Performing the linear regression
global_prediction = model2.predict(dates_reshape) #Calculating the trendline
lab3_data.index = lab3_data.index.map(dt.datetime.fromordinal)
ax12_slope = model2.coef_*12 # C/yr
ax12_score = model2.score(dates_reshape,global_temp) # R^2 value
'''ax12_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax12.plot(lab3_data.index.values,global_prediction,'k')
ax12.text(1,1,'Slope = '+ str(np.round(ax12_slope,4)) +" C/yr and R^2 = "+str(np.round(ax12_score,4))+"; 95% conf tbd")

plt.show()

# %%historgrams for part 1

fig2 = plt.figure(figsize=[15,15])
ax21 = fig2.add_subplot(211)
ax22 = fig2.add_subplot(212)
lab3_data_pre_85 = lab3_data.loc[:'1985-12-31']
ubc_temp_pre_85 = lab3_data_pre_85[r'UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)']
lab3_data_post_85 = lab3_data.loc['1985-12-31':]
ubc_temp_post_85 = lab3_data_post_85[r'UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)']
ax21.hist(ubc_temp_pre_85,alpha=0.5,label="<1985",bins=40,density=True,edgecolor='k')
ax21.hist(ubc_temp_post_85,alpha=0.5,label = ">1985",bins=40,density=True,edgecolor='k')
ax21.plot(title="Histogram of UBC Temperature Anomalies, binsize = 40")
ax21.legend()
ax21.title.set_text("Histogram of UBC Temperature Anomalies, binsize = 40")
ax21.set(xlabel="Temp Anomaly (C)",ylabel = "Count per bin")


global_temp_pre_85 = lab3_data_pre_85[r'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)']

global_temp_post_85 = lab3_data_post_85[r'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)']
ax22.hist(global_temp_pre_85,alpha=0.5,label="<1985",bins=40,density=True,edgecolor='k')
ax22.hist(global_temp_post_85,alpha=0.5,label = ">1985",bins=40,density=True,edgecolor='k')
ax22.legend()
ax22.title.set_text("Histogram of global Temperature Anomalies, binsize = 40")
ax22.set(xlabel="Temp Anomaly (C)",ylabel = "Count per bin")

plt.show()

'''Part 2'''


# %%extracting data

lab3_data_50_60 = lab3_data.loc['1950-1-1':'1960-1-1']
ubc_temp_50_60 = lab3_data_50_60[r'UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)']
global_temp_50_60 = lab3_data_50_60[r'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)']

lab3_data_60_70 = lab3_data.loc['1960-1-1':'1970-1-1']
ubc_temp_60_70 = lab3_data_60_70[r'UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)']
global_temp_60_70 = lab3_data_60_70[r'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)']

lab3_data_70_80 = lab3_data.loc['1970-1-1':'1980-1-1']
ubc_temp_70_80 = lab3_data_70_80[r'UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)']
global_temp_70_80 = lab3_data_70_80[r'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)']

lab3_data_80_90 = lab3_data.loc['1980-1-1':'1990-1-1']
ubc_temp_80_90 = lab3_data_80_90[r'UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)']
global_temp_80_90 = lab3_data_80_90[r'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)']

lab3_data_90_00 = lab3_data.loc['1990-1-1':'2000-1-1']
ubc_temp_90_00 = lab3_data_90_00[r'UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)']
global_temp_90_00 = lab3_data_90_00[r'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)']

lab3_data_00_10 = lab3_data.loc['2000-1-1':'2010-1-1']
ubc_temp_00_10 = lab3_data_00_10[r'UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)']
global_temp_00_10 = lab3_data_00_10[r'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)']

lab3_data_10_20 = lab3_data.loc['2010-1-1':'2020-1-1']
ubc_temp_10_20 = lab3_data_10_20[r'UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)']
global_temp_10_20 = lab3_data_10_20[r'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)']

#%% figure for ubc data

fig3 = plt.figure(figsize=[25,25])
fig3.tight_layout()
ax31 = fig3.add_subplot(421)
ax32 = fig3.add_subplot(422)
ax33 = fig3.add_subplot(423)
ax34 = fig3.add_subplot(424)
ax35 = fig3.add_subplot(425)
ax36 = fig3.add_subplot(426)
ax37 = fig3.add_subplot(427)

#plotting ubc data
ax31.plot(lab3_data_50_60.index.values,ubc_temp_50_60)
ax32.plot(lab3_data_60_70.index.values,ubc_temp_60_70)
ax33.plot(lab3_data_70_80.index.values,ubc_temp_70_80)
ax34.plot(lab3_data_80_90.index.values,ubc_temp_80_90)
ax35.plot(lab3_data_90_00.index.values,ubc_temp_90_00)
ax36.plot(lab3_data_00_10.index.values,ubc_temp_00_10)
ax37.plot(lab3_data_10_20.index.values,ubc_temp_10_20)

#linear regression for UBC data

model_ubc_50_60 = LinearRegression()
lab3_data_50_60.index = lab3_data_50_60.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_50_60.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_ubc_50_60.fit(dates_reshape,ubc_temp_50_60) #Performing the linear regression
ubc_prediction_50_60 = model_ubc_50_60.predict(dates_reshape) #Calculating the trendline
lab3_data_50_60.index = lab3_data_50_60.index.map(dt.datetime.fromordinal)
ax31_slope = model_ubc_50_60.coef_*12 # C/yr
ax31_score = model_ubc_50_60.score(dates_reshape,ubc_temp_50_60) # R^2 value
'''ax32_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax31.plot(lab3_data_50_60.index.values,ubc_prediction_50_60,'k')
ax31.text('1959-01-16T00:00:00.000000000',0.75,'Slope = '+ str(np.round(ax31_slope,4)) +" C/yr and R^2 = "+str(np.round(ax31_score,4))+"; 95% conf tbd")
ax31.set_title("UBC temp anomaly 1950-1960")


model_ubc_60_70 = LinearRegression()
lab3_data_60_70.index = lab3_data_60_70.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_60_70.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_ubc_60_70.fit(dates_reshape,ubc_temp_60_70) #Performing the linear regression
ubc_prediction_60_70 = model_ubc_60_70.predict(dates_reshape) #Calculating the trendline
lab3_data_60_70.index = lab3_data_60_70.index.map(dt.datetime.fromordinal)
ax32_slope = model_ubc_60_70.coef_*12 # C/yr
ax32_score = model_ubc_60_70.score(dates_reshape,ubc_temp_60_70) # R^2 value
'''ax32_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax32.plot(lab3_data_60_70.index.values,ubc_prediction_60_70,'k')
ax32.text('1961-01-16T00:00:00.000000000',2,'Slope = '+ str(np.round(ax32_slope,4)) +" C/yr and R^2 = "+str(np.round(ax32_score,4))+"; 95% conf tbd")
ax32.set_title("UBC temp anomaly 1960-1970")

model_ubc_70_80 = LinearRegression()
lab3_data_70_80.index = lab3_data_70_80.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_70_80.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_ubc_70_80.fit(dates_reshape,ubc_temp_70_80) #Performing the linear regression
ubc_prediction_70_80 = model_ubc_70_80.predict(dates_reshape) #Calculating the trendline
lab3_data_70_80.index = lab3_data_70_80.index.map(dt.datetime.fromordinal)
ax33_slope = model_ubc_70_80.coef_*12 # C/yr
ax33_score = model_ubc_70_80.score(dates_reshape,ubc_temp_70_80) # R^2 value
'''ax33_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax33.plot(lab3_data_70_80.index.values,ubc_prediction_70_80,'k')
ax33.text('1971-01-16T00:00:00.000000000',2,'Slope = '+ str(np.round(ax33_slope,4)) +" C/yr and R^2 = "+str(np.round(ax33_score,4))+"; 95% conf tbd")
ax33.set_title("UBC temp anomaly 1970-1980")


model_ubc_80_90 = LinearRegression()
lab3_data_80_90.index = lab3_data_80_90.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_80_90.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_ubc_80_90.fit(dates_reshape,ubc_temp_80_90) #Performing the linear regression
ubc_prediction_80_90 = model_ubc_80_90.predict(dates_reshape) #Calculating the trendline
lab3_data_80_90.index = lab3_data_80_90.index.map(dt.datetime.fromordinal)
ax34_slope = model_ubc_80_90.coef_*12 # C/yr
ax34_score = model_ubc_80_90.score(dates_reshape,ubc_temp_80_90) # R^2 value
'''ax34_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax34.plot(lab3_data_80_90.index.values,ubc_prediction_80_90,'k')
ax34.text('1981-01-16T00:00:00.000000000',3,'Slope = '+ str(np.round(ax34_slope,4)) +" C/yr and R^2 = "+str(np.round(ax34_score,4))+"; 95% conf tbd")
ax34.set_title("UBC temp anomaly 1980-1990")

model_ubc_90_00 = LinearRegression()
lab3_data_90_00.index = lab3_data_90_00.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_90_00.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_ubc_90_00.fit(dates_reshape,ubc_temp_90_00) #Performing the linear regression
ubc_prediction_90_00 = model_ubc_90_00.predict(dates_reshape) #Calculating the trendline
lab3_data_90_00.index = lab3_data_90_00.index.map(dt.datetime.fromordinal)
ax35_slope = model_ubc_90_00.coef_*12 # C/yr
ax35_score = model_ubc_90_00.score(dates_reshape,ubc_temp_90_00) # R^2 value
'''ax35_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax35.plot(lab3_data_90_00.index.values,ubc_prediction_90_00,'k')
ax35.text('1991-01-16T00:00:00.000000000',3,'Slope = '+ str(np.round(ax35_slope,4)) +" C/yr and R^2 = "+str(np.round(ax35_score,4))+"; 95% conf tbd")
ax35.set_title("UBC temp anomaly 1990-2000")


model_ubc_00_10 = LinearRegression()
lab3_data_00_10.index = lab3_data_00_10.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_00_10.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_ubc_00_10.fit(dates_reshape,ubc_temp_00_10) #Performing the linear regression
ubc_prediction_00_10 = model_ubc_00_10.predict(dates_reshape) #Calculating the trendline
lab3_data_00_10.index = lab3_data_00_10.index.map(dt.datetime.fromordinal)
ax36_slope = model_ubc_00_10.coef_*12 # C/yr
ax36_score = model_ubc_00_10.score(dates_reshape,ubc_temp_00_10) # R^2 value
'''ax36_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax36.plot(lab3_data_00_10.index.values,ubc_prediction_00_10,'k')
ax36.text('2001-01-16T00:00:00.000000000',3,'Slope = '+ str(np.round(ax36_slope,4)) +" C/yr and R^2 = "+str(np.round(ax36_score,4))+"; 95% conf tbd")
ax36.set_title("UBC temp anomaly 2000-2010")

model_ubc_10_20 = LinearRegression()
lab3_data_10_20.index = lab3_data_10_20.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_10_20.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_ubc_10_20.fit(dates_reshape,ubc_temp_10_20) #Performing the linear regression
ubc_prediction_10_20 = model_ubc_10_20.predict(dates_reshape) #Calculating the trendline
lab3_data_10_20.index = lab3_data_10_20.index.map(dt.datetime.fromordinal)
ax37_slope = model_ubc_10_20.coef_*12 # C/yr
ax37_score = model_ubc_10_20.score(dates_reshape,ubc_temp_10_20) # R^2 value
'''ax37_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax37.plot(lab3_data_10_20.index.values,ubc_prediction_10_20,'k')
ax37.text('2011-01-16T00:00:00.000000000',5,'Slope = '+ str(np.round(ax37_slope,4)) +" C/yr and R^2 = "+str(np.round(ax37_score,4))+"; 95% conf tbd")
ax37.set_title("UBC temp anomaly 2010-2020")

plt.show()


#%% figure for global data

fig4 = plt.figure(figsize=[25,25])
fig4.tight_layout()
ax41 = fig4.add_subplot(421)
ax42 = fig4.add_subplot(422)
ax43 = fig4.add_subplot(423)
ax44 = fig4.add_subplot(424)
ax45 = fig4.add_subplot(425)
ax46 = fig4.add_subplot(426)
ax47 = fig4.add_subplot(427)

#plotting global data
ax41.plot(lab3_data_50_60.index.values,global_temp_50_60)
ax42.plot(lab3_data_60_70.index.values,global_temp_60_70)
ax43.plot(lab3_data_70_80.index.values,global_temp_70_80)
ax44.plot(lab3_data_80_90.index.values,global_temp_80_90)
ax45.plot(lab3_data_90_00.index.values,global_temp_90_00)
ax46.plot(lab3_data_00_10.index.values,global_temp_00_10)
ax47.plot(lab3_data_10_20.index.values,global_temp_10_20)

#linear regression for global data

model_global_50_60 = LinearRegression()
lab3_data_50_60.index = lab3_data_50_60.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_50_60.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_global_50_60.fit(dates_reshape,global_temp_50_60) #Performing the linear regression
global_prediction_50_60 = model_global_50_60.predict(dates_reshape) #Calculating the trendline
lab3_data_50_60.index = lab3_data_50_60.index.map(dt.datetime.fromordinal)
ax41_slope = model_global_50_60.coef_*12 # C/yr
ax41_score = model_global_50_60.score(dates_reshape,global_temp_50_60) # R^2 value
'''ax42_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax41.plot(lab3_data_50_60.index.values,global_prediction_50_60,'k')
ax41.text('1959-01-16T00:00:00.000000000',0.075,'Slope = '+ str(np.round(ax41_slope,4)) +" C/yr and R^2 = "+str(np.round(ax41_score,4))+"; 95% conf tbd")
ax41.set_title("global temp anomaly 1950-1960")


model_global_60_70 = LinearRegression()
lab3_data_60_70.index = lab3_data_60_70.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_60_70.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_global_60_70.fit(dates_reshape,global_temp_60_70) #Performing the linear regression
global_prediction_60_70 = model_global_60_70.predict(dates_reshape) #Calculating the trendline
lab3_data_60_70.index = lab3_data_60_70.index.map(dt.datetime.fromordinal)
ax42_slope = model_global_60_70.coef_*12 # C/yr
ax42_score = model_global_60_70.score(dates_reshape,global_temp_60_70) # R^2 value
'''ax42_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax42.plot(lab3_data_60_70.index.values,global_prediction_60_70,'k')
ax42.text('1961-01-16T00:00:00.000000000',0.2,'Slope = '+ str(np.round(ax42_slope,4)) +" C/yr and R^2 = "+str(np.round(ax42_score,4))+"; 95% conf tbd")
ax42.set_title("global temp anomaly 1960-1970")

model_global_70_80 = LinearRegression()
lab3_data_70_80.index = lab3_data_70_80.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_70_80.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_global_70_80.fit(dates_reshape,global_temp_70_80) #Performing the linear regression
global_prediction_70_80 = model_global_70_80.predict(dates_reshape) #Calculating the trendline
lab3_data_70_80.index = lab3_data_70_80.index.map(dt.datetime.fromordinal)
ax43_slope = model_global_70_80.coef_*12 # C/yr
ax43_score = model_global_70_80.score(dates_reshape,global_temp_70_80) # R^2 value
'''ax43_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax43.plot(lab3_data_70_80.index.values,global_prediction_70_80,'k')
ax43.text('1971-01-16T00:00:00.000000000',0.3,'Slope = '+ str(np.round(ax43_slope,4)) +" C/yr and R^2 = "+str(np.round(ax43_score,4))+"; 95% conf tbd")
ax43.set_title("global temp anomaly 1970-1980")

model_global_80_90 = LinearRegression()
lab3_data_80_90.index = lab3_data_80_90.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_80_90.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_global_80_90.fit(dates_reshape,global_temp_80_90) #Performing the linear regression
global_prediction_80_90 = model_global_80_90.predict(dates_reshape) #Calculating the trendline
lab3_data_80_90.index = lab3_data_80_90.index.map(dt.datetime.fromordinal)
ax44_slope = model_global_80_90.coef_*12 # C/yr
ax44_score = model_global_80_90.score(dates_reshape,global_temp_80_90) # R^2 value
'''ax44_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax44.plot(lab3_data_80_90.index.values,global_prediction_80_90,'k')
ax44.text('1981-01-16T00:00:00.000000000',0.4,'Slope = '+ str(np.round(ax44_slope,4)) +" C/yr and R^2 = "+str(np.round(ax44_score,4))+"; 95% conf tbd")
ax44.set_title("global temp anomaly 1980-1990")

model_global_90_00 = LinearRegression()
lab3_data_90_00.index = lab3_data_90_00.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_90_00.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_global_90_00.fit(dates_reshape,global_temp_90_00) #Performing the linear regression
global_prediction_90_00 = model_global_90_00.predict(dates_reshape) #Calculating the trendline
lab3_data_90_00.index = lab3_data_90_00.index.map(dt.datetime.fromordinal)
ax45_slope = model_global_90_00.coef_*12 # C/yr
ax45_score = model_global_90_00.score(dates_reshape,global_temp_90_00) # R^2 value
'''ax45_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax45.plot(lab3_data_90_00.index.values,global_prediction_90_00,'k')
ax45.text('1991-01-16T00:00:00.000000000',0.75,'Slope = '+ str(np.round(ax45_slope,4)) +" C/yr and R^2 = "+str(np.round(ax45_score,4))+"; 95% conf tbd")
ax45.set_title("global temp anomaly 1990-2000")

model_global_00_10 = LinearRegression()
lab3_data_00_10.index = lab3_data_00_10.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_00_10.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_global_00_10.fit(dates_reshape,global_temp_00_10) #Performing the linear regression
global_prediction_00_10 = model_global_00_10.predict(dates_reshape) #Calculating the trendline
lab3_data_00_10.index = lab3_data_00_10.index.map(dt.datetime.fromordinal)
ax46_slope = model_global_00_10.coef_*12 # C/yr
ax46_score = model_global_00_10.score(dates_reshape,global_temp_00_10) # R^2 value
'''ax46_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax46.plot(lab3_data_00_10.index.values,global_prediction_00_10,'k')
ax46.text('2001-01-16T00:00:00.000000000',0.85,'Slope = '+ str(np.round(ax46_slope,4)) +" C/yr and R^2 = "+str(np.round(ax46_score,4))+"; 95% conf tbd")
ax46.set_title("global temp anomaly 2000-2010")

model_global_10_20 = LinearRegression()
lab3_data_10_20.index = lab3_data_10_20.index.map(dt.datetime.toordinal)
dates_reshape = lab3_data_10_20.index.values
dates_reshape = dates_reshape.reshape(-1,1)
model_global_10_20.fit(dates_reshape,global_temp_10_20) #Performing the linear regression
global_prediction_10_20 = model_global_10_20.predict(dates_reshape) #Calculating the trendline
lab3_data_10_20.index = lab3_data_10_20.index.map(dt.datetime.fromordinal)
ax47_slope = model_global_10_20.coef_*12 # C/yr
ax47_score = model_global_10_20.score(dates_reshape,global_temp_10_20) # R^2 value
'''ax47_conf = out of my knowledge what confidence rates are and how to do it python, will go over it later'''
ax47.plot(lab3_data_10_20.index.values,global_prediction_10_20,'k')
ax47.text('2011-01-16T00:00:00.000000000',1.0,'Slope = '+ str(np.round(ax47_slope,4)) +" C/yr and R^2 = "+str(np.round(ax47_score,4))+"; 95% conf tbd")
ax47.set_title("global temp anomaly 2010-2020")


plt.show()
# %% part 3
'''part 3'''

plt.scatter(lab3_data[r'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)'],lab3_data[r'UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)'],s=3)
plt.xlabel("UBC Temp (C)")
plt.ylabel("Global Temp (C)")
pearson = scipy.stats.pearsonr(lab3_data[r'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)'],lab3_data[r'UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)'])
plt.text(-0.4,5,"Correlation = "+ str(np.round(pearson[0],5)) + " and P-value = " + format(pearson[1],'.4e'))

# %% part 4
lab3_data_part4=lab3_data[[ 'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)','TSI (W/m2) (SATIRE project and Lean 2000)','global mean stratospheric aerosol optical depth (GISS, dimensionless)', 'Atmospheric CO2 (ppm, Earth Policy Institute/NOAA)', 'Anthropogenic SO2 emissions (Tg/y, from Pacific Northwest National Laboratory)', 'MEI (NOAA, dimensionless)']].copy()
lab3_data_part4.set_index('global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)',inplace=True)

fig5 = plt.figure(figsize=[25,25])
ax51 = fig5.add_subplot(321)
ax52 = fig5.add_subplot(322)
ax53 = fig5.add_subplot(323)
ax54 = fig5.add_subplot(324)
ax55 = fig5.add_subplot(325)


model_TSI = LinearRegression()

TSI_reshape = lab3_data_part4['TSI (W/m2) (SATIRE project and Lean 2000)'].values
TSI_reshape = TSI_reshape.reshape(-1,1)
model_TSI.fit(TSI_reshape,lab3_data_part4.index.values) #Performing the linear regression
TSIanom_prediction = model_TSI.predict(TSI_reshape) #Calculating the trendline
ax51_slope = model_TSI.coef_ # slope
ax51_score = model_TSI.score(TSI_reshape,lab3_data_part4.index.values) # R^2 value
ax51.text(1366.37,1,'Slope = '+ str(np.round(ax51_slope,4)) +" slope and R^2 = "+str(np.round(ax51_score,4))+"; 95% conf tbd")
ax51.plot(lab3_data_part4['TSI (W/m2) (SATIRE project and Lean 2000)'],TSIanom_prediction ,'k')
ax51.scatter(lab3_data_part4['TSI (W/m2) (SATIRE project and Lean 2000)'],lab3_data_part4.index.values)
ax51.set_title('Global Temperature Anomaly vs. TSI from 1950-2016')
ax51.set_xlabel('Total Solar Irradiance')
ax51.set_ylabel('Global Temp Anomaly (C^{\circ})')



model_AOD = LinearRegression()

AOD_reshape = lab3_data_part4['global mean stratospheric aerosol optical depth (GISS, dimensionless)'].values
AOD_reshape = AOD_reshape.reshape(-1,1)
model_AOD.fit(AOD_reshape,lab3_data_part4.index.values) #Performing the linear regression
AODanom_prediction = model_AOD.predict(AOD_reshape) #Calculating the trendline
ax52_slope = model_AOD.coef_ # slope
ax52_score = model_AOD.score(AOD_reshape,lab3_data_part4.index.values) # R^2 value
ax52.text(0.06,1,'Slope = '+ str(np.round(ax52_slope,4)) +" slope and R^2 = "+str(np.round(ax52_score,4))+"; 95% conf tbd")
ax52.plot(lab3_data_part4['global mean stratospheric aerosol optical depth (GISS, dimensionless)'],AODanom_prediction ,'k')
ax52.scatter(lab3_data_part4['global mean stratospheric aerosol optical depth (GISS, dimensionless)'],lab3_data_part4.index.values)
ax52.set_title('Global Temperature Anomaly vs. AOD from 1950-2016')
ax52.set_xlabel('Aerosol Optical Depth')
ax52.set_ylabel('Global Temp Anomaly (C^{\circ})')



model_CO2 = LinearRegression()

CO2_reshape = lab3_data_part4['Atmospheric CO2 (ppm, Earth Policy Institute/NOAA)'].values
CO2_reshape = CO2_reshape.reshape(-1,1)
model_CO2.fit(CO2_reshape,lab3_data_part4.index.values) #Performing the linear regression
CO2anom_prediction = model_CO2.predict(CO2_reshape) #Calculating the trendline
ax53_slope = model_CO2.coef_ # slope
ax53_score = model_CO2.score(CO2_reshape,lab3_data_part4.index.values) # R^2 value
ax53.text(320,1,'Slope = '+ str(np.round(ax53_slope,4)) +" slope and R^2 = "+str(np.round(ax53_score,4))+"; 95% conf tbd")
ax53.plot(lab3_data_part4['Atmospheric CO2 (ppm, Earth Policy Institute/NOAA)'],CO2anom_prediction ,'k')
ax53.scatter(lab3_data_part4['Atmospheric CO2 (ppm, Earth Policy Institute/NOAA)'],lab3_data_part4.index.values)
ax53.set_title('Global Temperature Anomaly vs. CO2 from 1950-2016')
ax53.set_xlabel('Carbon Dioxide Concentration (ppm)')
ax53.set_ylabel('Global Temp Anomaly (C^{\circ})')

model_SO2 = LinearRegression()

SO2_reshape = lab3_data_part4['Anthropogenic SO2 emissions (Tg/y, from Pacific Northwest National Laboratory)'].values
SO2_reshape = SO2_reshape.reshape(-1,1)
model_SO2.fit(SO2_reshape,lab3_data_part4.index.values) #Performing the linear regression
SO2anom_prediction = model_SO2.predict(SO2_reshape) #Calculating the trendline
ax54_slope = model_SO2.coef_ # slope
ax54_score = model_SO2.score(SO2_reshape,lab3_data_part4.index.values) # R^2 value
ax54.text(50,1,'Slope = '+ str(np.round(ax54_slope,4)) +" slope and R^2 = "+str(np.round(ax54_score,4))+"; 95% conf tbd")
ax54.plot(lab3_data_part4['Anthropogenic SO2 emissions (Tg/y, from Pacific Northwest National Laboratory)'],SO2anom_prediction ,'k')
ax54.scatter(lab3_data_part4['Anthropogenic SO2 emissions (Tg/y, from Pacific Northwest National Laboratory)'],lab3_data_part4.index.values)
ax54.set_title('Global Temperature Anomaly vs. SO2 from 1950-2016')
ax54.set_xlabel('Sulfate Concentration (Tg/yr)')
ax54.set_ylabel('Global Temp Anomaly (C^{\circ})')


model_MEIg = LinearRegression()

MEI_reshape = lab3_data_part4['MEI (NOAA, dimensionless)'].values
MEI_reshape = MEI_reshape.reshape(-1,1)
model_MEIg.fit(MEI_reshape,lab3_data_part4.index.values) #Performing the linear regression
MEIanom_prediction = model_MEIg.predict(MEI_reshape) #Calculating the trendline
MEIg_slope = model_MEIg.coef_ # slope
MEIg_score = model_MEIg.score(MEI_reshape,lab3_data_part4.index.values) # R^2 value
ax55.text(-2,1,'Slope = '+ str(np.round(MEIg_slope,4)) +" slope and R^2 = "+str(np.round(MEIg_score,4))+"; 95% conf tbd")
ax55.plot(lab3_data_part4['MEI (NOAA, dimensionless)'],MEIanom_prediction ,'k')

ax55.scatter(lab3_data_part4['MEI (NOAA, dimensionless)'],lab3_data_part4.index.values)
ax55.set_title('Global Temperature Anomaly vs. MEI from 1950-2016')
ax55.set_xlabel('Multivariate ENSO Index')
ax55.set_ylabel('Global Temp Anomaly (C^{\circ})')

plt.show()



fig6 = plt.figure(figsize=[25,25])
ax61 = fig6.add_subplot(211)
ax62 = fig6.add_subplot(212)

ax61.text(-2,1,'Slope = '+ str(np.round(MEIg_slope,4)) +" slope and R^2 = "+str(np.round(MEIg_score,4))+"; 95% conf tbd")
ax61.plot(lab3_data_part4['MEI (NOAA, dimensionless)'],MEIanom_prediction ,'k')

ax61.scatter(lab3_data_part4['MEI (NOAA, dimensionless)'],lab3_data_part4.index.values)
ax61.set_title('Global Temperature Anomaly vs. MEI from 1950-2016')
ax61.set_xlabel('Multivariate ENSO Index')
ax61.set_ylabel('Global Temp Anomaly (C^{\circ})')


model_MEIu = LinearRegression()

model_MEIu.fit(MEI_reshape,lab3_data['UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)']) #Performing the linear regression
MEIanom_prediction = model_MEIu.predict(MEI_reshape) #Calculating the trendline
MEIu_slope = model_MEIu.coef_ # slope
MEIu_score = model_MEIu.score(MEI_reshape,lab3_data['UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)']) # R^2 value
ax62.text(-2,1,'Slope = '+ str(np.round(MEIu_slope,4)) +" slope and R^2 = "+str(np.round(MEIu_score,4))+"; 95% conf tbd")
ax62.plot(lab3_data_part4['MEI (NOAA, dimensionless)'],MEIanom_prediction ,'k')

ax62.scatter(lab3_data_part4['MEI (NOAA, dimensionless)'],lab3_data['UBC temperature anomaly (Celsius) (measured at UBC weather station,  1961-1990 seasonal cycle used to calculate anomaly)'])
ax62.set_title('UBC Temperature Anomaly vs. MEI from 1950-2016')
ax62.set_xlabel('Multivariate ENSO Index')
ax62.set_ylabel('Global Temp Anomaly (C^{\circ})')

plt.show()
# %% part 5

#same data as part 4. easier to modulate code like this
lab3_data_part5=lab3_data[[ 'global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)','TSI (W/m2) (SATIRE project and Lean 2000)','global mean stratospheric aerosol optical depth (GISS, dimensionless)', 'Atmospheric CO2 (ppm, Earth Policy Institute/NOAA)', 'Anthropogenic SO2 emissions (Tg/y, from Pacific Northwest National Laboratory)', 'MEI (NOAA, dimensionless)']].copy()
lab3_data_part5.set_index('global mean temperature anomaly (Celsius) (HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)',inplace=True)
# TSI_reshape = lab3_data_part4['TSI (W/m2) (SATIRE project and Lean 2000)'].values
# TSI_reshape = TSI_reshape.reshape(-1,1)
# AOD_reshape = lab3_data_part4['global mean stratospheric aerosol optical depth (GISS, dimensionless)'].values
# AOD_reshape = AOD_reshape.reshape(-1,1)
# CO2_reshape = lab3_data_part4['Atmospheric CO2 (ppm, Earth Policy Institute/NOAA)'].values
# CO2_reshape = CO2_reshape.reshape(-1,1)
# SO2_reshape = lab3_data_part4['Anthropogenic SO2 emissions (Tg/y, from Pacific Northwest National Laboratory)'].values
# SO2_reshape = SO2_reshape.reshape(-1,1)
# MEI_reshape = lab3_data_part4['MEI (NOAA, dimensionless)'].values
# MEI_reshape = MEI_reshape.reshape(-1,1)
#data_reshape=[TSI_reshape,AOD_reshape,CO2_reshape,SO2_reshape,MEI_reshape]

multiLinearRegression = LinearRegression()

X=lab3_data_part5[['TSI (W/m2) (SATIRE project and Lean 2000)','global mean stratospheric aerosol optical depth (GISS, dimensionless)', 'Atmospheric CO2 (ppm, Earth Policy Institute/NOAA)', 'Anthropogenic SO2 emissions (Tg/y, from Pacific Northwest National Laboratory)', 'MEI (NOAA, dimensionless)']]
multiLinearRegression.fit(X,lab3_data_part5.index.values.reshape(-1,1))
prediction = multiLinearRegression.predict(X)
MLscore=multiLinearRegression.score(X.values,lab3_data_part5.index.values.reshape(-1,1))
#need coefficient of determination
plt.text(-0.5,1,'Coeff for ML regression = '+ str(np.round(multiLinearRegression.coef_,4)) +" slope and R^2 = "+str(np.round(MLscore,4))+"; 95% conf tbd")

plt.scatter(lab3_data_part5.index.values,prediction,s=3,label="Fit vs Raw")
x=np.linspace(-0.5,1,500)
y=x
plt.plot(x,y,'r',label="y=x")
plt.legend()
plt.show()

#need slope
plt.plot(lab3_data.index.values,prediction,'r',label="ML Fit")
plt.scatter(lab3_data.index.values,lab3_data_part5.index.values, label = "raw",s=3)
plt.legend()
plt.show()
# %%Part 6

#loading data

#had to change data to csv, dont know why
dummyData =  pd.read_csv(r'dummyvariables_lab3.csv')

#checking data
print(dummyData.head(0))

trumpAge = dummyData['Age of Donald Trump (yr)'].dropna()
globalTempAnomaly = dummyData['global mean temperature anomaly (Celsius, from HadCRUT, 1961-1990 seasonal cycle)'].dropna()
plt.scatter(trumpAge,globalTempAnomaly,s=3,label="Trump vs Raw")
pearsonTrump = scipy.stats.pearsonr(trumpAge,globalTempAnomaly)
plt.text(10,1,"Correlation = "+ str(np.round(pearsonTrump[0],5)) + " and P-value = " + format(pearsonTrump[1],'.4e'))
plt.xlabel("Trump Age")
plt.ylabel("Gloabl Temp Anomaly (C)")
plt.show()

truncCO2 = dummyData['Truncated imaginary CO2 (ppm)'].dropna()
truncTempAnam=dummyData['Truncated imaginary temperature anomaly (Celsius)'].dropna()
fullCO2 = dummyData['Imaginary CO2 (ppm)']
fullTemp = dummyData['Imaginary temperature anomaly (Celsius)']

#regression of the data

imaginaryModel = LinearRegression()
imaginaryModel.fit(truncCO2.values.reshape(-1,1),truncTempAnam)
predictionImaginary = imaginaryModel.predict(truncCO2.values.reshape(-1,1))
imaginarySlope = imaginaryModel.coef_
imaginaryR2 = imaginaryModel.score(truncCO2.values.reshape(-1,1),predictionImaginary)
pearsonImag = scipy.stats.pearsonr(truncCO2,predictionImaginary)

#plotting data
fig7 = plt.figure(figsize=[25,25])
ax71 = fig7.add_subplot(211)
ax72 = fig7.add_subplot(212)

#first plot
ax71.set_title('Linear Regression Extrapolation Using Imaginary Data')
ax71.plot(truncCO2,predictionImaginary,'k',label='Predicted')
ax71.plot(truncCO2,truncTempAnam,'r',label='Raw')
ax71.legend()
ax71.set_xlabel('Truncated imaginary CO2 (ppm)')
ax71.set_ylabel('Truncated imaginary temperature anomaly (Celsius)')
ax71.text(300,4,"Slope: "+str(imaginarySlope)+" R^2: "+str(imaginaryR2)+" P-value = "+str(pearsonImag[1]))
#second plot
ax72.plot(fullCO2,fullTemp,'r',label='Raw')
ax72.plot(fullCO2[:len(predictionImaginary)],predictionImaginary,'k',label="Predicted")
ax72.set_xlabel('Imaginary CO2 (ppm)')
ax72.set_ylabel('Imaginary temperature anomaly (Celsius)')
ax72.legend()

plt.show()
