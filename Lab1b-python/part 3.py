#lab 1 part 3
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt

#reading data, used regex for separator, this may be an issue for students
M =  pd.read_csv('MSL_Global_1993_2016.txt', sep='\s* ',header=None)

#checking the data
print(M)
print(M.head(0))

#getting dates
float_dates = M[0].values
print(float_dates)


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
    
    
dates = pd.Series(float_dates).apply(floating_year_to_timestamp)

#getting MSL data in mm
MSL = M[1].values
print(MSL)

plt.plot(dates,MSL)
plt.xlabel("Year")
plt.ylabel("Mean Sea Level (mm)")
plt.title("Mean Sea Level (MSL) from 1993-2016")
