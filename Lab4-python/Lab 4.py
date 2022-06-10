#lab 4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
part 0, unclear for now. need to see what the objective of this part. 
Essentially, python already has builtin function that does this...
'''
#mymean() function

#input: series or dataframe or list or array with null values and returns the mean of the non-null values
#output: mean 
def mean(x):
    if type(x) == list or type(x) == np.ndarray:
        x = np.array(x)
        x = x[~np.isnan(x)]
        return np.mean(x)
    elif type(x) == pd.DataFrame:
        x = x.dropna()
        return x.mean()
    if type(x) == pd.Series:
        return x.dropna().mean()
    else:
        print("Invalid input")

# %% part 1 loading data

data = pd.read_excel("STRATOGEM_plankton.xls",index_col=0)

# %% part 2 plot data

#summing data along rows
summedData = data.sum(axis=1)


#creating subplots
fig1 = plt.figure(figsize=[25,15])
ax11 = fig1.add_subplot(211)
ax12 = fig1.add_subplot(212)



ax11.plot(summedData.index,summedData[:])
ax11.set_title("Total Phytoplankton Count Linear Scale")
ax11.set_ylabel("Phytoplankton Count")

ax12.plot(summedData.index,summedData[:])
ax12.set_yscale("log")
ax12.set_title("Total Phytoplankton Count Log Scale")
ax12.set_ylabel("Phytoplankton Count")

plt.xlabel("Date")
plt.show()

# %% part 3 function to calculate Shannon-Weiner diversity index



#function to calculate Shannon-Wiener Index given a series or dataframe or list or array
def shannon_wiener(x):  # x is a series or dataframe or list or array   
    #if x is a dataframe or series,convert to array
    if type(x) == pd.DataFrame or type(x) == pd.Series:
        x = x.values 
    x=x[x>0]
    sample_species = np.sum(x) #N - Number of species in the sample
    H=0
    for i in x:
        p_i = (i/sample_species)
        H-=p_i*np.log(p_i)
    return H


#testing function
print(shannon_wiener(np.array([1,1,1])))
print(shannon_wiener(np.array([1,2,3])))
print(shannon_wiener(np.array([0,0,0])))
print(shannon_wiener(np.array([1,np.nan,1])))


# %% part 4 calculate and plot the Shannon-Wiener index for the STRATOGEM data

sw_array = []

for i in range(len(data)):
    sw_array.append(shannon_wiener(data.iloc[[i]]))
    
plt.plot(data.index,sw_array)
plt.xlabel("SWDI")
plt.ylabel("Date")
plt.title("Shannon Wiener Diversity Index from Apr, 2002 to June 2005")
plt.xticks(rotation=45)
plt.show()

# %% part 5 Calculate the SW index and total phytoplankton by year


#2003 data


#creating figure
fig2 = plt.figure(figsize=[25,15])
ax21 = fig2.add_subplot(211)
ax22 = fig2.add_subplot(212)


data03=data.loc['2003-01-1 00:00:00':'2003-12-31 11:59:59']
summed03data = data03.sum(axis=1)

sw_array_03 = []

for i in range(len(data03)):
    sw_array_03.append(shannon_wiener(data03.iloc[[i]]))

    
    
ax21.plot(summed03data.index,summed03data[:])
ax21.set_xlabel("Phytoplanton Count")
ax21.set_ylabel("Date")
ax21.set_yscale("log") #wasn't asked for but matlalb code asked for this
ax21.set_title("Phytoplanton Count for 2003")
 

ax22.plot(data03.index,sw_array_03)
ax22.set_xlabel("SWDI")
ax22.set_ylabel("Date")
ax22.set_title("Shannon Wiener Diversity Index for 2003")



plt.show()


#2004 data

#creating figure
fig3 = plt.figure(figsize=[25,15])
ax31 = fig3.add_subplot(211)
ax32 = fig3.add_subplot(212)


data04=data.loc['2004-01-1 00:00:00':'2004-12-31 11:59:59']
summed04data = data04.sum(axis=1)

sw_array_04 = []

for i in range(len(data04)):
    sw_array_04.append(shannon_wiener(data04.iloc[[i]]))

    
    
ax31.plot(summed04data.index,summed04data[:])
ax31.set_xlabel("Phytoplanton Count")
ax31.set_ylabel("Date")
ax31.set_yscale("log") #wasn't asked for but matlalb code asked for this
ax31.set_title("Phytoplanton Count for 2004")
 

ax32.plot(data04.index,sw_array_04)
ax32.set_xlabel("SWDI")
ax32.set_ylabel("Date")
ax32.set_title("Shannon Wiener Diversity Index for 2004")



plt.show()
