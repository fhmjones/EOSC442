#lab 1 part 2
import numpy as np
import numpy.ma as ma
import pandas
import matplotlib.pyplot as plt

#Part 1: Loading Mauna Loa Data
muana_loa_data = pandas.read_csv(r"monthly_maunaloa_co2.csv", header= 54, skiprows = [55,56])
print(muana_loa_data) #checking file is fine
print(muana_loa_data.head(0)) #checking header names
muana_loa_data.columns = muana_loa_data.columns.str.replace(' ','') #striping whitespace from headers
print(muana_loa_data.head(0)) #making sure whitespaces are stripped

print(muana_loa_data.loc[0,'Yr']) #making sure no issues with accessing data using column names
print(muana_loa_data.dtypes) #checking what data types are stored

#pulling out ndarrays from dataframe.
co2_date = muana_loa_data["Date"].values
co2_date = co2_date[:,1] #since we have two co2_date columns I will only be using the first. we need to find a solution for this
co2 = muana_loa_data['CO2'].values
co2=co2[:,0]#same as co2_date

#lab instructions only ask for above 2 columns, but TA sheet also uses below columns
co2sa = muana_loa_data["seasonally"].values
co2sa = co2sa[:,] # https://stackoverflow.com/questions/40557910/plt-plot-meaning-of-0-and-1 explanation of this notation, good to use in lab 1a
co2fit = muana_loa_data["fit"].values
co2safit = muana_loa_data["seasonally"].values
co2safit = co2safit[:,1]


#plotting raw data
plt.plot(co2_date,co2,'r')
plt.title('Raw CO2 Data vs. Time with missing data')
plt.xlabel('Date')
plt.ylabel('CO2 (ppm)')
plt.show()

#masking empty entries
co2 = ma.masked_where(co2<0,co2) #masking data
co2_date = ma.masked_where(co2_date<0,co2_date) #TA solution doesnt mask data data. May remove.
co2sa = ma.masked_where(co2sa<0,co2sa)
co2fit = ma.masked_where(co2fit<0,co2fit)
co2safit = ma.masked_where(co2safit<0,co2safit)
#Plotting all raw CO2 data with fitted curve overplotted
plt.plot(co2_date,co2,'r-',label='CO2 Raw') #plotting data
plt.plot(co2_date,co2fit,'k-',label='CO2 Fit')
plt.title('Raw CO2 Data vs. Time')
plt.legend()
plt.ylabel('CO2 (ppm)')
plt.xlabel('Date')
plt.show()

#plotting seasonally adjusted CO2 data with fitted curve overplotted
plt.plot(co2_date,co2sa,'g-',label='co2sa') #plotting data
plt.plot(co2_date,co2safit,'k-',label='co2sa')
plt.legend()
plt.title('Seasonally Adjusted CO2 Data vs. Time')
plt.ylabel('CO2 SA fit (ppm)')
plt.xlabel('Date')
plt.show()