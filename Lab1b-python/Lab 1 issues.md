### Part 1
#### first cell
 - There are instructions for where to save files for lab session. I am still not sure how these notebooks will be distributed so I dont know exactly what to put there and I will need to change this later on. **me**
 - Why does the original PDF recomend viewing the csv in vim? **TA**
 
#### between code cell 13 and 14
 - the way the data is organized overcomplicates things a bit.
  - I can either spend some more time to find a more elgent way to wrangle the data, or we can leave this part already done for the students, or we can have different data/ask students to clean it up in excel before importing **discuss with francis/TA**
  
#### between code cell 14 and 15
- lab pdf only askfs for co2 vs date plot, TA code also does plots for seasonally adjusted values. I added parts for graphing as the TA did. will need to see if we omit this or not. **TA**  

#### between code cell 18 and 19
- running pyplot on Jupyter NB does not require using the plot.show() command, should we leave this as it is important to know when coding in other environemnts, and if so, should we explain that it is not necessary in jupyter but in other environments it is and why. **francis/TA**

#### between code cell 20 and 21
- it seems like the masking library doesnt work on jupyter, it seems data is being removed and not masked. this is not the case in other environemnts, I will have to spend time looking in to this. If I dont find solution, either maybe phil can help or we can leave the technique I used as is. If we use my logic, we may not need to intoroduce the concept of masking and just frame this as a data wrangling excercise. **me/phil/TA**

#### between code cell 27 and 29
- TA code plots fitted and seasonally adjusted data, lab instructins doesnt ask for it. I put it in anyways. **TA**

### Part 2
- a second method of reading data is used here. Should I introduce a second method? I have used pandas throughout my solutions for now.
- dateimte: should I introduce some functionality in prelab as opposed to lab?

### Part 3
- there is a mention of an example figure, which does not exist.


```python

```
