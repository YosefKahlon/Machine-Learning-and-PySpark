import findspark
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql import SQLContext
from pyspark.sql import functions
from pyspark.sql.functions import lit  # lit for literal
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

findspark.init()
sc = SparkContext("local[*]", "Simple App")

sqlContext = SQLContext(sc)

""" ----------------------------------- q1.1 ------------------------------------------------------"""


# read data from json
df = sqlContext.read.json("books.json", multiLine=True)

# SQL Query in Spark
df.registerTempTable("book_table")
starting_with_f = sqlContext.sql(
    "SELECT title, author,(2020 - year)  AS since_published  FROM book_table WHERE author LIKE 'F%'")

print("------------------- q1.1 -------------------------")
# Add a column of the difference
# Between 2022 and the year of publication
# starting_with_ff = starting_with_f.withColumn('num of year',
#                                               year(current_date()).cast(IntegerType()) - df.year.cast(IntegerType()))

starting_with_f.show()

"""----------------------------------------------------------------------------------------------- """

""" ----------------------------------- q1.2 ------------------------------------------------------"""

print("------------------- q1.2 -------------------------")

df = sqlContext.read.json("books.json", multiLine=True)
# df.show()

##The average number of pages written
# by each author but only books in English
df.filter(df.language == 'English').groupBy(df.author).agg(avg(df.pages)).show()

"""----------------------------------------------------------------------------------------------- """

import random
import numpy as np
import pandas as pd
import csv
import math

""" ----------------------------------- q2.1 ------------------------------------------------------"""

print("------------------- q2.1 -------------------------")

user1 = pd.read_csv('prices.txt')
user1.to_csv("data.csv")

items = []
y = []
i = 0

# https://www.geeksforgeeks.org/delete-a-csv-column-in-python/
# open input CSV file as source
# open output CSV file as result
with open("data.csv", "r") as source:
    reader = csv.reader(source)

    with open("output.csv", "w") as result:
        writer = csv.writer(result)
        for r in reader:  # add feature (1 - 11 )
            writer.writerow((r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]))

with open("data.csv", "r") as source:
    reader = csv.reader(source)

    with open("price.csv", "w") as result:
        writer = csv.writer(result)
        for r in reader:
            writer.writerow((r[12],))

k = 0
t = 0

Prediction = []
actual = []
with open('output.csv') as csvfile:
    csvReader = csv.reader(csvfile)

    csvReader.__next__()
    i = 0

    for row in csvReader:

        if row.__len__() == 0:
            i = i + 1
            continue

        if i == 1 or i == 7 or i == 17 or i == 27 or i == 37 or i == 47 or i == 55:

            Prediction.append(list(np.float_(row)))
            i = i + 1
            continue
        else:

            items.append(list(np.float_(row)))

            i = i + 1

with open('price.csv') as csvfile:
    csvReader = csv.reader(csvfile)

    csvReader.__next__()
    i = 0

    for row in csvReader:

        if row.__len__() == 0:
            i = i + 1
            continue

        if i == 1 or i == 7 or i == 17 or i == 27 or i == 37 or i == 47 or i == 55:
            actual.append(float(row[0]))

            i = i + 1
            continue
        else:

            y.append(float(row[0]))

            i = i + 1

data_x = np.array(items)

data_y = np.array(y)

wi = []
deriv = []
for i in range(11):
    wi.append(0)
    deriv.append(0)

b = 0
alpha = 0.0001

for iteration in range(1000):
    sum = 0
    i = 0
    while i < len(wi):
        sum += (wi[i] * data_x[:, i])
        i = i + 1

    deriv_b = np.mean(1 * ((sum + b) - data_y))

    i = 0

    while i < len(wi):
        deriv[i] = np.dot(((sum + b) - data_y), data_x[:, i]) * 1.0 / len(data_y)
        i = i + 1

    b -= alpha * deriv_b

    i = 0
    while i < len(wi):
        wi[i] -= (alpha * deriv[i])
        i = i + 1

i = 0
while i < len(Prediction):
    print("Estimated price : ",
          np.dot(np.array(Prediction[i]), np.array(wi)) + b, " actual : ", actual[i])

    i = i + 1

""" ----------------------------------- q2.2 ------------------------------------------------------"""

print("------------------- q2.2 -------------------------")
i = 0
a = 1 / 2 * len(data_y)
sum = 0

i = 0
while i < len(data_y):
    sum += (np.dot(np.array(data_x[i]), np.array(wi)) + b - data_y[i])
    i = i + 1

lost = a * math.pow(sum, 2)
print("mean squared-error ", lost)

""" ----------------------------------- q2.3 ------------------------------------------------------"""

print("------------------- q2.3 -------------------------")

with open("data.csv", "r") as source:
    reader = csv.reader(source)

    with open("q3.csv", "w") as result:
        writer = csv.writer(result)
        for r in reader:
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

        if row.__len__() == 0:
            i = i + 1
            continue

        if i == 1 or i == 7 or i == 17 or i == 27 or i == 37:

            Prediction.append(list(np.float_(row)))
            #print(list(np.float_(row)))
            i = i + 1
            continue
        else:

            items.append(list(np.float_(row)))

            i = i + 1

with open('fire.csv') as csvfile:
    csvReader = csv.reader(csvfile)

    csvReader.__next__()
    i = 0

    for row in csvReader:

        if row.__len__() == 0:
            i = i + 1
            continue

        if i == 1 or i == 7 or i == 17 or i == 27 or i == 37:
            actual.append(float(row[0]))

            i = i + 1
            continue
        else:

            y.append(float(row[0]))

            i = i + 1

data_x = np.array(items)

data_y = np.array(y)


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

    deriv = np.dot((h(data_x, w, b) - data_y), data_x) * 1 / len(data_y)

    b -= alpha * deriv_b

    w -= alpha * deriv

i = 0
while i < len(Prediction):
    print("prob of have firemen: ",
          h(np.array([Prediction[i]]), w, b), "actual :", actual[i])
    i = i + 1

""" ----------------------------------- q2.4 ------------------------------------------------------"""

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
0
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
        
precision = accuracy, how accurate we are in the category.

precision =   1 
            ------  = 1 
            1 + 0
            
F-measure  = 2 * 1 * 1/2      2  
            ------------  = -----
              1 + 1/2         3      
"""
