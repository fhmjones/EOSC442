#lab 2

import pandas as pd
import matplotlib.pyplot as plt


#part 1


#loading data
#regex a bit different to deal with spacing
#using parse_dates to save as date time
data = pd.read_csv('DailyAirTempUBC19592018.txt', sep='\s+ ',header=None,parse_dates=[0],names=['date','temp'])

#checking data
print(data.head(0))
print(data)
#checking parsing worked for dates
print(data.dtypes)


#plotting data
plt.plot(data['date'],data['temp'])
plt.xlabel('Date')
plt.ylabel('Temperature (Celsius)')
plt.title('Daily Temperature at UBC from 1959-2017')
plt.show()

#part 2

#checking if grouping data works
#average
print(data.groupby(data.date.dt.month)['temp'].mean())
#standard deviation
print(data.groupby(data.date.dt.month)['temp'].std())
#minumum recorded value
print(data.groupby(data.date.dt.month)['temp'].min())
#maximum recorded value
print(data.groupby(data.date.dt.month)['temp'].max())


#plotting all data on one graph
plt.plot(data.groupby(data.date.dt.month)['temp'].mean(),'b-s',label="mean")
plt.plot(data.groupby(data.date.dt.month)['temp'].mean()+data.groupby(data.date.dt.month)['temp'].std(),'r--',label="mean+std")
plt.plot(data.groupby(data.date.dt.month)['temp'].mean()-data.groupby(data.date.dt.month)['temp'].std(),'r--',label="mean-std")
plt.plot(data.groupby(data.date.dt.month)['temp'].min(),'k:',label="min")
plt.plot(data.groupby(data.date.dt.month)['temp'].max(),'k:',label="max")
plt.title('UBC Totem Station Mean Monthly Air Temperature 1959-2016')
plt.ylabel('Air Temperature (\circC)')
plt.xlabel('Month of Year')
plt.legend()
plt.show()


'''part 3,4,5'''


''' tried some things here'''
#monthly average
#print(data.groupby(data.date.dt.to_period("M"))['temp'].mean())
#plt.plot(data.groupby(data.date.dt.to_period("M"))['temp'].mean().to_timestamp(),'s',label="overal mean")

#monthl avearge by year
#plt.plot(data.groupby(data.date.dt.month)['temp'].mean(),'b-s','b',label="mean")
#plt.show()
#calculate different between monthley mean time-series and monthly average

''' actual part 3 4 and 5'''
print(data.groupby(data.date.dt.to_period("M"))['temp'].mean())

#create dataframe from grouped by month and year data
part2_data=pd.DataFrame(data.groupby(data.date.dt.to_period("M"))['temp'].mean())

#create list from monthly average
avg_month_list=data.groupby(data.date.dt.month)['temp'].mean().to_list()
#helper variables for loop
lengthpart2_data=len(part2_data['temp'])
year=709//12
#lengthen list avg_month_list to be the length of part2_data
for i in range(year):
    for j in range(12):
        avg_month_list.append(avg_month_list[j])
        
#add monthly mean to data frame
part2_data["monthly mean"]=avg_month_list[:709]

#add difference to data frame
part2_data["diff"]=part2_data["temp"]-part2_data["monthly mean"]


#added some style to graph, can remove style input to make graph look like original lab
part2_data.plot(style=['-','b.','y-'])
plt.show()


#plotting everything together as subplots
part2_data.plot(subplots=True)




