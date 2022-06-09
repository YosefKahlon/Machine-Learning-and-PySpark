import findspark
from pyspark.sql import SQLContext
from pyspark.sql.functions import *

findspark.init()
sc = SparkContext("local[*]", "Simple App")

sqlContext = SQLContext(sc)

""" ----------------------------------- q1.1 ------------------------------------------------------"""

# read data from json
df = sqlContext.read.json("books.json", multiLine=True)

# SQL Query in Spark
df.registerTempTable("book_table")
starting_with_f = sqlContext.sql(
    "SELECT title, author,(2022 - year)  AS since_published  FROM book_table WHERE author LIKE 'F%'")

print("------------------- q1.1 -------------------------")
starting_with_f.show()

""" ----------------------------------- q1.2 ------------------------------------------------------"""

print("------------------- q1.2 -------------------------")
##The average number of pages written
# by each author but only books in English
df.filter(df.language == 'English').groupBy(df.author).agg(avg(df.pages)).show()

"""----------------------------------------------------------------------------------------------- """

import random
import numpy as np
import pandas as pd
import csv
import math

print("------------------- q2.1 -------------------------")
# Create CSV to get the data easily
user1 = pd.read_csv('prices.txt')
user1.to_csv("data.csv")

# https://www.geeksforgeeks.org/delete-a-csv-column-in-python/
# open input CSV file as source
# open output CSV file as result
with open("data.csv", "r") as source:
    reader = csv.reader(source)

    with open("output.csv", "w") as result:
        writer = csv.writer(result)
        for r in reader:
            # add feature (1 - 11 )
            writer.writerow((r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]))

with open("data.csv", "r") as source:
    reader = csv.reader(source)

    with open("price.csv", "w") as result:
        writer = csv.writer(result)
        for r in reader:
            # add feature (12)
            writer.writerow((r[12],))

items = []
y = []
Prediction = []
actual = []
with open('output.csv') as csvfile:
    csvReader = csv.reader(csvfile)
    csvReader.__next__()
    i = 0
    for row in csvReader:

        # empty row in my csv
        if row.__len__() == 0:
            i = i + 1
            continue

        # Saving 25% of my data as a Test set
        if i == 1 or i == 7 or i == 17 or i == 27 or i == 37 or i == 47 or i == 55:

            Prediction.append(list(np.float_(row)))
            i = i + 1
            continue
        # rest for Training set
        else:
            # data x
            items.append(list(np.float_(row)))
            i = i + 1

with open('price.csv') as csvfile:
    csvReader = csv.reader(csvfile)
    csvReader.__next__()
    i = 0
    for row in csvReader:

        # empty row in my csv
        if row.__len__() == 0:
            i = i + 1
            continue

        # Saving 25% of my data as a Test set
        if i == 1 or i == 7 or i == 17 or i == 27 or i == 37 or i == 47 or i == 55:
            actual.append(float(row[0]))

            i = i + 1
            continue
        # rest for Training set
        else:
            # data y
            y.append(float(row[0]))
            i = i + 1

data_x = np.array(items)
data_y = np.array(y)
b = 0
wi = []
deriv_w = []
for i in range(11):
    wi.append(0)
    deriv_w.append(0)

# I found  for this number of iterations and  alfa give the best result
alpha = 0.001
for iteration in range(10000):
    sum = 0
    i = 0

    # Sum for derivative
    while i < len(wi):
        sum += (wi[i] * data_x[:, i])
        i = i + 1

    # Derived  for 𝑏
    deriv_b = np.mean(1 * ((sum + b) - data_y))

    i = 0
    # Derived for w
    while i < len(wi):
        deriv_w[i] = np.dot(((sum + b) - data_y), data_x[:, i]) * 1.0 / len(data_y)
        i = i + 1

    # set new w and b
    b -= alpha * deriv_b
    i = 0
    while i < len(wi):
        wi[i] -= (alpha * deriv_w[i])
        i = i + 1

i = 0
while i < len(Prediction):
    print("Estimated price : ",
          np.dot(np.array(Prediction[i]), np.array(wi)) + b, " actual : ", actual[i])

    i = i + 1

""" ----------------------------------- q2.2 ------------------------------------------------------"""

print("------------------- q2.2 -------------------------")

# Lost function
i = 0
a = 1 / (2 * len(data_y))
sum = 0

while i < len(data_y):
    sum = sum + math.pow((np.dot(np.array(data_x[i]), np.array(wi)) + b - data_y[i]), 2)

    i = i + 1

lost = a * sum

print("mean squared-error ", lost)

""" ----------------------------------- q2.3 ------------------------------------------------------"""

print("------------------- q2.3 -------------------------")

with open("data.csv", "r") as source:
    reader = csv.reader(source)

    with open("q3.csv", "w") as result:
        writer = csv.writer(result)
        for r in reader:
            # add all necessary feature
            writer.writerow((r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[12]))

with open("data.csv", "r") as source:
    reader = csv.reader(source)

    with open("fire.csv", "w") as result:
        writer = csv.writer(result)
        for r in reader:
            writer.writerow((r[11],))

Prediction = []
actual = []
items = []
y = []
with open('q3.csv') as csvfile:
    csvReader = csv.reader(csvfile)

    csvReader.__next__()
    i = 0

    for row in csvReader:

        # empty row
        if row.__len__() == 0:
            i = i + 1
            continue

        # 15% for Test set
        if i == 1 or i == 7 or i == 17 or i == 27 or i == 37:

            Prediction.append(list(np.float_(row)))
            # print(list(np.float_(row)))
            i = i + 1
            continue
        # 85% for Train set
        else:
            # data x
            items.append(list(np.float_(row)))

            i = i + 1

with open('fire.csv') as csvfile:
    csvReader = csv.reader(csvfile)

    csvReader.__next__()
    i = 0

    for row in csvReader:
        # empty row
        if row.__len__() == 0:
            i = i + 1
            continue
        # 15% for Test set
        if i == 1 or i == 7 or i == 17 or i == 27 or i == 37:
            actual.append(float(row[0]))

            i = i + 1
            continue
        # 85% for Train set
        else:
            # data y
            y.append(float(row[0]))

            i = i + 1

data_x = np.array(items)
data_y = np.array(y)


# The prediction function
def h(x, w, b):
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))


wi = []
for i in range(11):
    wi.append(0.0)  # random.random() * 1)

w = np.array(wi)
b = 0  # random.random() * 1

alpha = 0.001

for iteration in range(100000):
    deriv_b = np.mean(1 * ((h(data_x, w, b) - data_y)))
    i = 0
    deriv_w = np.dot((h(data_x, w, b) - data_y), data_x) * 1 / len(data_y)

    b -= alpha * deriv_b
    w -= alpha * deriv_w

i = 0
while i < len(Prediction):
    print("prob of have firemen: ",
          h(np.array([Prediction[i]]), w, b), "actual :", actual[i])
    i = i + 1
print("------------------- q2.4 -------------------------")

print("+--------------------------+---------------------------------- + \n"
      "|                          |class as fire | class as not fire  | \n"
      "| rly positive - fire      |       1      |        1           | \n"
      "| ------------------------ + --------------------------------- | \n"
      "| rly negative  - not fire |      0       |        3           | \n"
      "+--------------------------+-----------------------------------+")

print("accuracy = 1 + 3       4    \n"
      "         -------- =   --    \n"
      "          1+1+0+3      5    \n")

print("reacall  =   1         1    \n"
      "         -------- =   --    \n"
      "            1+1        2    \n")

print("precision  =   1         1    \n"
      "            -------- =   --   \n"
      "              1+0        1    \n")

print("F-measure  = 2 * 1 * 1/2     2    \n"
      "             ---------- =   --    \n"
      "               1 + 1/2       3    \n")

print("------------------- END -------------------------")

"""
[5.0208, 1.0, 3.531, 1.5, 2.0, 7.0, 4.0, 62.0, 1.0, 1.0, 29.5]
[5.0597, 1.0, 4.455, 1.121, 1.0, 6.0, 3.0, 42.0, 3.0, 1.0, 29.9]
[14.4598, 2.5, 12.8, 3.0, 2.0, 9.0, 5.0, 14.0, 4.0, 1.0, 82.9]
[5.05, 1.0, 5.0, 1.02, 0.0, 5.0, 2.0, 46.0, 4.0, 1.0, 30.0]
[9.0384, 1.0, 7.8, 1.5, 1.5, 7.0, 3.0, 23.0, 3.0, 3.0, 43.9]

prob of have firemen:  [0.32246753] actual : 0.0
prob of have firemen:  [0.07899615] actual : 0.0
prob of have firemen:  [0.75711239] actual : 1.0
prob of have firemen:  [0.18051465] actual : 1.0
prob of have firemen:  [0.00373377] actual : 0.0



---------------------------------------------------------------
|                           class as fire | class as not fire  |
| rly positive - fire      |       1              1            |
| ----------               | --------------------------------- |
| rly negative  - not fire |      0               3            |
|---------------------------------------------------------------

accuracy = 1 + 3     4 
          -------- = --
           1+1+0+3    5 

reacall : The amount of actual positivity in relation to all those who are actually positive

reacall = 1     1 
        ---- = ----
        1 + 1   2
        
precision = how accurate we are in the category.

precision =   1 
            ------  = 1 
            1 + 0
            
F-measure  = 2 * 1 * 1/2      2  
            ------------  = -----
              1 + 1/2         3      
"""
