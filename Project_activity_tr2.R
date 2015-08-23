# Resources
# http://topepo.github.io/caret/modelList.html


## The goal of your project is to predict the manner in which they did the exercise.
# This is the "classe" variable in the training set.
# You may use any of the other variables to predict with.
# You should create a report describing
# - how you built your model
# - how you used cross validation
# - what you think the expected out of sample error is
# - why you made the choices you did.
# - also use your prediction model to predict 20 different test cases. 
# 
# 1. Your submission should consist of
# - a link to a Github repo
# - with your R markdown
# - compiled HTML file describing your analysis
# - text of the writeup to < 2000 words and the number of figures to be less than 5.
# - It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online
# 
# 2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above.
# - Please submit your predictions in appropriate format to the programming assignment for automated grading.
# - See the programming assignment for additional details. 

## Download and load files
setwd("/Users/yingjiang/Dropbox/Learnings/Stats_data/Coursework/Data_science_spec/Data_science_C8/Project")
setwd("C:/Users/jiangy/Dropbox/Learnings/Stats_data/Coursework/Data_science_spec/Data_science_C8/Project")

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
              destfile = "pmltrain.csv", method = "curl")
pmltrain <- read.csv("pmltrain.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
              destfile = "pmltest.csv", method = "curl")
pmltest <- read.csv("pmltest.csv")

colnames(pmltrain)
# [1] "X"                        "user_name"                "raw_timestamp_part_1"    
# [4] "raw_timestamp_part_2"     "cvtd_timestamp"           "new_window"              
# [7] "num_window"               "roll_belt"                "pitch_belt"              
# [10] "yaw_belt"                 "total_accel_belt"         "kurtosis_roll_belt"      
# [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"        "skewness_roll_belt"      
# [16] "skewness_roll_belt.1"     "skewness_yaw_belt"        "max_roll_belt"           
# [19] "max_picth_belt"           "max_yaw_belt"             "min_roll_belt"           
# [22] "min_pitch_belt"           "min_yaw_belt"             "amplitude_roll_belt"     
# [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"       "var_total_accel_belt"    
# [28] "avg_roll_belt"            "stddev_roll_belt"         "var_roll_belt"           
# [31] "avg_pitch_belt"           "stddev_pitch_belt"        "var_pitch_belt"          
# [34] "avg_yaw_belt"             "stddev_yaw_belt"          "var_yaw_belt"            
# [37] "gyros_belt_x"             "gyros_belt_y"             "gyros_belt_z"            
# [40] "accel_belt_x"             "accel_belt_y"             "accel_belt_z"            
# [43] "magnet_belt_x"            "magnet_belt_y"            "magnet_belt_z"           
# [46] "roll_arm"                 "pitch_arm"                "yaw_arm"                 
# [49] "total_accel_arm"          "var_accel_arm"            "avg_roll_arm"            
# [52] "stddev_roll_arm"          "var_roll_arm"             "avg_pitch_arm"           
# [55] "stddev_pitch_arm"         "var_pitch_arm"            "avg_yaw_arm"             
# [58] "stddev_yaw_arm"           "var_yaw_arm"              "gyros_arm_x"             
# [61] "gyros_arm_y"              "gyros_arm_z"              "accel_arm_x"             
# [64] "accel_arm_y"              "accel_arm_z"              "magnet_arm_x"            
# [67] "magnet_arm_y"             "magnet_arm_z"             "kurtosis_roll_arm"       
# [70] "kurtosis_picth_arm"       "kurtosis_yaw_arm"         "skewness_roll_arm"       
# [73] "skewness_pitch_arm"       "skewness_yaw_arm"         "max_roll_arm"            
# [76] "max_picth_arm"            "max_yaw_arm"              "min_roll_arm"            
# [79] "min_pitch_arm"            "min_yaw_arm"              "amplitude_roll_arm"      
# [82] "amplitude_pitch_arm"      "amplitude_yaw_arm"        "roll_dumbbell"           
# [85] "pitch_dumbbell"           "yaw_dumbbell"             "kurtosis_roll_dumbbell"  
# [88] "kurtosis_picth_dumbbell"  "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
# [91] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"    "max_roll_dumbbell"       
# [94] "max_picth_dumbbell"       "max_yaw_dumbbell"         "min_roll_dumbbell"       
# [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"         "amplitude_roll_dumbbell" 
# [100] "amplitude_pitch_dumbbell" "amplitude_yaw_dumbbell"   "total_accel_dumbbell"    
# [103] "var_accel_dumbbell"       "avg_roll_dumbbell"        "stddev_roll_dumbbell"    
# [106] "var_roll_dumbbell"        "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
# [109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"         "stddev_yaw_dumbbell"     
# [112] "var_yaw_dumbbell"         "gyros_dumbbell_x"         "gyros_dumbbell_y"        
# [115] "gyros_dumbbell_z"         "accel_dumbbell_x"         "accel_dumbbell_y"        
# [118] "accel_dumbbell_z"         "magnet_dumbbell_x"        "magnet_dumbbell_y"       
# [121] "magnet_dumbbell_z"        "roll_forearm"             "pitch_forearm"           
# [124] "yaw_forearm"              "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
# [127] "kurtosis_yaw_forearm"     "skewness_roll_forearm"    "skewness_pitch_forearm"  
# [130] "skewness_yaw_forearm"     "max_roll_forearm"         "max_picth_forearm"       
# [133] "max_yaw_forearm"          "min_roll_forearm"         "min_pitch_forearm"       
# [136] "min_yaw_forearm"          "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
# [139] "amplitude_yaw_forearm"    "total_accel_forearm"      "var_accel_forearm"       
# [142] "avg_roll_forearm"         "stddev_roll_forearm"      "var_roll_forearm"        
# [145] "avg_pitch_forearm"        "stddev_pitch_forearm"     "var_pitch_forearm"       
# [148] "avg_yaw_forearm"          "stddev_yaw_forearm"       "var_yaw_forearm"         
# [151] "gyros_forearm_x"          "gyros_forearm_y"          "gyros_forearm_z"         
# [154] "accel_forearm_x"          "accel_forearm_y"          "accel_forearm_z"         
# [157] "magnet_forearm_x"         "magnet_forearm_y"         "magnet_forearm_z"        
# [160] "classe"                  
sapply(pmltrain, class)
colnames(pmltest)

## Clean
# Remove the following columns:
# - 1st 7 (contains user names and other non-numerical data)
# - Blank columns ("")
# - NA columns
pmltrain1 <- pmltrain[-c(1:7, which(pmltrain[1, ] == ""), which(is.na(pmltrain[1, ])))]
ncol(pmltrain1) # 53


## Explore

# Convert all to numeric and do correlation
# First, use correlation to generate a "pairs" plot
pmltrain2 <- as.data.frame(sapply(pmltrain1, as.numeric))
pmltrain2$classe <- pmltrain$classe

M1 <- abs(cor(pmltrain2))
diag(M1) <- 0
image(M1) # FIGURE
# The last column represents the variable "classe"'s correlation with all predictors
which(M1[, ncol(M1)] > .1)
# magnet_belt_y       magnet_belt_z           pitch_arm     total_accel_arm         accel_arm_x        magnet_arm_x 
# 12                  13                  15                  17                  21                  24 
# magnet_arm_y        magnet_arm_z    accel_dumbbell_x   magnet_dumbbell_z       pitch_forearm total_accel_forearm 
# 25                  26                  34                  39                  41                  43 
# accel_forearm_x    magnet_forearm_x    magnet_forearm_y 
# 47                  50                  51 

# Choose the set of predictors, since they have relatively more obvious correlations with classe (> 0.1).


## Correlation matrix among the predictors to further narrow down variables
train_small <- pmltrain2[, c(which(M1[, ncol(M1)] > .1), ncol(pmltrain2))]
train_small$classe <- pmltrain$classe
M2 <- abs(cor(train_small[, -16]))
diag(M2) <- 0
ind <- which(M2 > 0.8, arr.ind=T)
#                row col
# magnet_arm_x   6   5
# accel_arm_x    5   6
# magnet_arm_z   8   7
# magnet_arm_y   7   8


## Build model:

library(caret)
library(e1071)
library(party)

# Splice data for k-fold cross-validation
set.seed(9510)
# Use 5 folds to get around 2000 samples in each fold
folds <- createFolds(y = train_small$classe,
                     k = 5,
                     list = T,
                     returnTrain = F)

# To each of the folds, apply preProcess with and without PCA.
# Get model, fit to the remaining fold as the "test" set.
# Get Accuracy (in sample errors) of resulting fit.
for(i in 1:5) {
    # Create validation folds
    validation <- train_small[-folds[[i]], ]
    holdout <- pmltrain1[folds[[i]], ]
    # 
    preProc <- preProcess(train_small[, -53], method = "pca", thresh = 0.9)
}

# 1. Build models including all predictors (without feature selection)
validation_full <- pmltrain2[-folds[[1]], ]
holdout_full <- pmltrain2[folds[[1]], ]

# 2. Build model with Decision Trees wo PCA
modelFit_DT_full <- ctree(classe ~ ., data = validation_full)
predicted_DT_full <- predict(modelFit_DT, newdata = holdout_full)
confusionMatrix(predicted_DT_full, holdout_full$classe)

# 3. Include just the predictors that are relatively more correlated with classe.
validation <- train_small[-folds[[1]], ]
holdout <- train_small[folds[[1]], ]

# Again build model with Decision Trees wo PCA
modelFit_DT <- ctree(classe ~ ., data = validation)
predicted_DT <- predict(modelFit_DT, newdata = holdout)
confusionMatrix(predicted_DT, holdout$classe)
# Get accuracy
confusionMatrix(predicted_DT, holdout$classe)$overall[1] # Note: this figure is exactly the same as predicted_DT_full!

# Build model with less predictors, with Decision Trees, including a PCA
# Note: 9 variables are needed to achieve 90% explanation of variance
preProc <- preProcess(validation[, -16], method = "pca", thresh = 0.9)
trainPC <- predict(preProc, validation[, -16])
modelFitPC <- ctree(validation$classe ~ ., data = trainPC)
testPC <- predict(preProc, holdout[, -16])
predictedPC <- predict(modelFitPC, testPC)
confusionMatrix(predictedPC, holdout$classe)
# Get accuracy
confusionMatrix(predictedPC, holdout$classe)$overall[1] # 73% Accurate. Not as good as without using PCA!

# Build model with even less predictors, discarding "accel_arm_x" (correlated to magnet_arm_x) and magnet_arm_y (correlated to magnet_arm_z)
train_smaller <- train_smaller[-c(5, 7)]
validation_smaller <- train_smaller[-folds[[1]], ]
holdout_smaller <- train_smaller[folds[[1]], ]
modelFit_DT_sml <- ctree(classe ~ ., data = validation_smaller)
predicted_DT_sml <- predict(modelFit_DT_sml, newdata = holdout_smaller)
confusionMatrix(predicted_DT_sml, holdout_smaller$classe) # Accuracy doesn't change much. Became slightly smaller.


# ?? How to get what the PCs represent? How do we predict while making physical sense?



