#lab 1 part 2
import pandas
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


#loading data from .txt file
M =  pandas.read_csv('MEI_data_only.txt', sep='\t',index_col=0,header=None)
#making sure data was read correctly
#print(M.head(0))
#print(M)
#flatten the data to 1d array
mei=M.to_numpy().flatten()
#print(mei)
#print(mei[13])

#possible uses for date time
#x=dt.datetime(2022, 2, 2)
#print(dt.datetime.now()) 
#print(dt.date.today())
#print(x.replace(year=2019))

#creating array of dates corresponding to data
dates =[]
for i in range(1950,2013):
    for j in range(1,13):
        dates.append(dt.datetime(i, j, 1))
        
#print(dates)

#ploting data vs dates
plt.plot(dates,mei)
plt.title("Monthly MEI index from 1950-2012")
plt.ylabel("MEI index")
plt.xlabel("Date")
plt.show()






