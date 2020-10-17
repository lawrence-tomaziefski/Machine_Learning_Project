# Predicting Exercise Type with Random Forest Machine Learning
Lawrence A. Tomaziefski  


***Executive Summary***
Personal fitness devices such as, *Fitbit* and *Nike FuelBand*, have become more and more prevelanet everyday fitness activities.  The immense amount of data has given rise to research in Human Activity Recogition.  The fitness devices are very useful in tracking the amount of activity. However, could these devices be used to inform people about the quality of thier activity? 

Vellaso, et al. conducted a study where they asked participants to wear a personal fitness device and conduct a set of excercises in six different manners.  The researchers instructed the participants to conduct the activity correctly, and then asked them to perform the activity using five common mistakes.  Thier report *Qualitative Activity Recognition of Weight Lifting Exercises* and accompanying dataset can be found at, http://groupware.les.inf.puc-rio.br/har.

The purpose of this project is to determine if the manner of how the participants conducted the exercise can be predicted.  In order determine this, a random forest machine learning model was built using 5 variables and 2 fold cross validation.  When the model was applied to a test dataset the *out of sample error rate* between 0% and .001%.  Additionaly the model was able to predict the manner of exercise in a validation dataset with 100% accuracy.  Therefore, it appears promising that exercise quailty can be determined using personal fitness devices.  

***Data Processing***
Note that the dataset that is labeled as test, was used as the validation dataset.  The training dataset is split later in the process into training and test datasets.  Proceeding in this manner allowed for the estimation of out of sample error.


```r
#Packages required for processing and analysis:
require(dplyr)
require(caret)
require(mlbench)
require(parallel)
require(doParallel)

data_sets = c("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

download.file(data_sets,destfile = c("training.csv", "testing.csv"),method = "libcurl")

training = read.csv("training.csv", header = TRUE, stringsAsFactors = FALSE)
validation = read.csv("testing.csv", header = TRUE, stringsAsFactors = FALSE)
```

***Cleaning the Data*** 
After doing a quick visual inspection and summary of the testing data there appears to be several variables that have nothing but NA values.  Therefore, we want to remove those variables because of the lack of predictive ability.  Removing the variables of class *logical* signficantly reduces the dataset from 160 to 58 variables.  The same data processing steps were applied to the training and validation datasets.  The validation dataset must contain only the variables are used for prediction later in the process and all factor variables need to have the same number of levels in the training and validation datasets.


```r
validation_logical = validation %>% select_if(is.logical)
validation_logical_names = names(validation_logical)
validation = validation %>% select(-one_of(validation_logical_names),-cvtd_timestamp, -1) %>% select(-58)
training = training %>% select(-one_of(validation_logical_names),-cvtd_timestamp, -1)
training$user_name = as.factor(training$user_name)
training$new_window = as.factor(training$new_window)
training$classe = as.factor(training$classe)
validation$user_name = as.factor(validation$user_name)
validation$new_window = as.factor(validation$new_window)
levels(validation$user_name ) = levels(training$user_name)
levels(validation$new_window ) = levels(training$new_window)
```

***Developing Random Forest Model***

Random Forests are excellent for classification problems such as this one, and are fairly straight-forward to implement.  The draw back is sometimes they can be very expenisive computationally, tend to overfit, and hard to interpret.  The following is the strategy implemented to overcome some of these issues.



+ K-fold cross validation with several numbers of folds were testing to determine trade-off between accuracy and speed.
+ Variable importance was explored to determine if the number of variables included in the model could be reduced.

**Building the models**

```r
#Split the training data set into a training and testing dataset in order to determine out of sample error rate.  
set.seed(5001)
in_train = createDataPartition(y = training$classe, p = .75, list = FALSE)
training = training[in_train, ]
testing = training[-in_train, ]

x = training[,-58]
y = as.factor(training[,58])

cluster = makeCluster(detectCores() - 1) #Required for parallel processing
registerDoParallel(cluster)

fit_control2 = trainControl(method = "cv", number = 2, allowParallel = TRUE)
fit_control3 = trainControl(method = "cv", number = 3, allowParallel = TRUE)
fit_control5 = trainControl(method = "cv", number = 5, allowParallel = TRUE) 
two_fold = train(x, y, method = "rf", trControl = fit_control2, importance = TRUE, na.action = na.omit)
three_fold = train(x, y, method = "rf", trControl = fit_control3, importance = TRUE, na.action = na.omit) 
five_fold = train(x, y, method = "rf", trControl = fit_control5, importance = TRUE, na.action = na.omit)
stopCluster(cluster)
registerDoSEQ()
```

The follwing table summarizes the results of the 3 models, note that *Time* is the total processing time in seconds.  



```
   Model Accuracy   Time Variables
1 2 Fold   0.9878 206.53        29
2 3 Fold   0.9985 341.80        29
3 5 Fold   0.9989 609.28        29
```

Clearly the 2 Fold model is the advantageous because it requires one-third of the processing time, with little sacrifice in accuracy.  The number of variables used in the model, remains a concern.  The following is a list of the twenty of the most important variables to the model.  There appears to be a seperation between the fifth and sixth variables on the list.  The potential exists to reduce the model from fifty seven variables to five without sacrificing accuracy.  




```
rf variable importance

  variables are sorted by maximum importance across the classes
  only 20 most important variables shown (out of 57)

                          A      B     C     D     E
raw_timestamp_part_1 75.074 100.00 76.98 85.99 43.85
roll_belt            43.171  61.29 53.47 52.86 58.23
num_window           37.942  59.73 53.56 41.77 42.17
magnet_dumbbell_z    50.617  37.90 42.76 36.59 36.26
pitch_forearm        33.757  39.25 49.08 41.12 36.26
magnet_dumbbell_y    36.285  36.01 41.86 38.24 35.40
yaw_belt             23.976  33.23 33.59 38.05 22.65
pitch_belt           20.756  30.43 31.09 27.19 28.43
roll_forearm         26.483  22.50 24.80 20.80 21.68
accel_dumbbell_y     17.190  18.89 23.52 20.18 19.52
roll_dumbbell        15.227  18.56 23.21 21.18 20.26
accel_forearm_x      13.548  20.89 18.00 23.07 23.18
accel_dumbbell_z     11.239  19.10 17.22 22.13 16.76
gyros_belt_z          9.585  15.98 18.01 16.83 21.34
magnet_belt_z        12.012  18.59 16.07 21.10 18.86
total_accel_dumbbell 12.549  19.55 16.08 20.55 18.40
magnet_belt_y        12.077  17.68 16.36 16.68 18.65
magnet_belt_x         8.777  15.62 15.97 11.72 18.24
gyros_dumbbell_y     18.190  14.09 16.69 16.19 12.24
yaw_dumbbell          9.660  17.49 15.97 13.86 13.37
```

A random forest model with two fold cross validation and five variables is trained.


```r
training1 = training %>% select(raw_timestamp_part_1, roll_belt, num_window, magnet_dumbbell_z, pitch_forearm, classe)
testing1 = testing %>% select(raw_timestamp_part_1, roll_belt, num_window, magnet_dumbbell_z, pitch_forearm, classe)
validation1 = validation %>% select(raw_timestamp_part_1, roll_belt, num_window, magnet_dumbbell_z, pitch_forearm)

x1 = training1[,-6]
y1 = as.factor(training1[,6])

fit_less = train(x1, y1, method = "rf", trControl = fit_control2, importance = TRUE, na.action = na.omit)
```

Running the model with a reduced number of variables increased both accuracy and computation speed.  





```
   Model Accuracy  Time Variables
1 2 Fold    0.999 34.39         5
```

Running the model on the test data set results in an accuracy between .999 and 1, therefore the out of sample error rate between 0 and .001. 


```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1034    0    0    0    0
         B    0  717    0    0    0
         C    0    0  642    0    0
         D    0    0    0  588    0
         E    0    0    0    0  715

Overall Statistics
                                    
               Accuracy : 1         
                 95% CI : (0.999, 1)
    No Information Rate : 0.2798    
    P-Value [Acc > NIR] : < 2.2e-16 
                                    
                  Kappa : 1         
 Mcnemar's Test P-Value : NA        

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000    1.000   1.0000   1.0000   1.0000
Specificity            1.0000    1.000   1.0000   1.0000   1.0000
Pos Pred Value         1.0000    1.000   1.0000   1.0000   1.0000
Neg Pred Value         1.0000    1.000   1.0000   1.0000   1.0000
Prevalence             0.2798    0.194   0.1737   0.1591   0.1935
Detection Rate         0.2798    0.194   0.1737   0.1591   0.1935
Detection Prevalence   0.2798    0.194   0.1737   0.1591   0.1935
Balanced Accuracy      1.0000    1.000   1.0000   1.0000   1.0000
```

Running the model on the validation dataset produces the following results:





```
   id pred
1   1    B
2   2    A
3   3    B
4   4    A
5   5    A
6   6    E
7   7    D
8   8    B
9   9    A
10 10    A
11 11    B
12 12    C
13 13    B
14 14    A
15 15    E
16 16    E
17 17    A
18 18    B
19 19    B
20 20    B
```


***Conclusions***
It appears that exercise quality can be determined using personal fitness devices.  A random forest machine learning model built with 5 variables and using 2 fold cross validation supports this conclusion.  Applying the model to a test dataset yielded an *out of sample error rate* between 0 and .001.  Additionaly the model was able to predict the manner of exercise in a validation dataset with 100% accuracy.  




