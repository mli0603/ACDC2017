# script to generate text file for training, validation and testing test dataset 
import random
import numpy as np
import os

# split test:(training+validation) = 2:8
train_percentage = 0.8
validation_percentage = 0.2

# generate testing index
f = open("test_data.txt","w+")
for index in range (101,151):
    f.write(" %3d \n" % (index))
f.close()
print("Test data successfully generated")

# generate training+validation index
total_num = 100
total_id = range(total_num) # 0-indexed
print("Total number of training frames")
print(total_num)

# validattion
validation_num = int(validation_percentage*(total_num))
print('Number of validation data')
print(validation_num)
validation_id = random.sample(range(total_num), validation_num)
validation_id = np.array(validation_id)

f = open("validation_data.txt","w+")
for index in np.nditer(validation_id):
    f.write(" %3d \n" % (index))
f.close()
print("Validation data successfully generated")

# train
train_num = total_num - validation_num
train_id = np.delete(total_id,validation_id)
print('Number of train data')
print(train_num)

# generate list of train images
f = open("train_data.txt","w+")
for index in np.nditer(train_id):
    f.write(" %3d \n" % (index))
f.close()
print("Train data successfully generated")