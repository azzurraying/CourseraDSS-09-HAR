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

# Scrutinize the variables. Group the similar
# 5 variables measured through weightlifting:
# roll, pitch, yaw, accel, magnet, gyros
# 4 objects were applied with these actions:
# belt, arm, dumbbell, forearm
grep("roll", colnames(pmltrain1))
# [1]  1 14 27 40
colnames(pmltrain1)[grep("roll", colnames(pmltrain1))]
# [1] "roll_belt"     "roll_arm"      "roll_dumbbell" "roll_forearm" 
grep("pitch", colnames(pmltrain1))
# [1]  2 15 28 41
colnames(pmltrain1)[grep("pitch", colnames(pmltrain1))]
# [1] "pitch_belt"     "pitch_arm"      "pitch_dumbbell" "pitch_forearm" 
grep("yaw", colnames(pmltrain1))
# [1]  3 16 29 42
colnames(pmltrain1)[grep("yaw", colnames(pmltrain1))]
# [1] "yaw_belt"     "yaw_arm"      "yaw_dumbbell" "yaw_forearm" 
grep("accel", colnames(pmltrain1))
# [1]  4  8  9 10 17 21 22 23 30 34 35 36 43 47 48 49
colnames(pmltrain1)[grep("accel", colnames(pmltrain1))]
# [1] "total_accel_belt"     "accel_belt_x"         "accel_belt_y"         "accel_belt_z"        
# [5] "total_accel_arm"      "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
# [9] "total_accel_dumbbell" "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
# [13] "total_accel_forearm"  "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
grep("magnet", colnames(pmltrain1))
# [1] 11 12 13 24 25 26 37 38 39 50 51 52
colnames(pmltrain1)[grep("magnet", colnames(pmltrain1))]
# [1] "magnet_belt_x"     "magnet_belt_y"     "magnet_belt_z"     "magnet_arm_x"      "magnet_arm_y"     
# [6] "magnet_arm_z"      "magnet_dumbbell_x" "magnet_dumbbell_y" "magnet_dumbbell_z" "magnet_forearm_x" 
# [11] "magnet_forearm_y"  "magnet_forearm_z"
grep("gyros", colnames(pmltrain1))
# [1]  5  6  7 18 19 20 31 32 33 44 45 46
colnames(pmltrain1)[grep("gyros", colnames(pmltrain1))]
# [1] "gyros_belt_x"     "gyros_belt_y"     "gyros_belt_z"     "gyros_arm_x"      "gyros_arm_y"     
# [6] "gyros_arm_z"      "gyros_dumbbell_x" "gyros_dumbbell_y" "gyros_dumbbell_z" "gyros_forearm_x" 
# [11] "gyros_forearm_y"  "gyros_forearm_z" 

###############
# This part is optional
length <- numeric(ncol(pmltrain1))
for(i in 1:ncol(pmltrain1)) {
    length[i] <- length(unique(pmltrain1[, i]))
}
# Or:
length <- sapply(pmltrain1, function(x) length(unique(x)))

# Simple linear regression to determine which variables are significant.
y <- as.numeric(pmltrain1$class)
modelFit_lm <- lm(y ~ ., data = pmltrain1[, -53])
summary(lm(y ~ pmltrain1$total_accel_belt))$coeff # Pretty much all variables are significant!
##########################

## Explore
# Convert all to numeric and do correlation

# First, use correlation to generate a "pairs" plot
pmltrain3 <- as.data.frame(sapply(pmltrain1, as.numeric))


#######################################
## Check for and clean outliers: 3 sds outside the mean
out <- list()
pmltrain_out <- pmltrain3
for(i in 1:52) {
    out[[i]] <- which(pmltrain3[, i] < mean(pmltrain3[, i])-3*sd(pmltrain3[, i]) |
              pmltrain3[, i] > mean(pmltrain3[, i])+3*sd(pmltrain3[, i]))
    pmltrain_out[, i][out[[i]]] <- NA
}

# # Number of outliers in each predictor variable
# [1,]  1   0
# [2,]  2   0
# [3,]  3   0
# [4,]  4   0
# [5,]  5 194
# [6,]  6 295
# [7,]  7 420
# [8,]  8   2
# [9,]  9   6
# [10,] 10   0
# [11,] 11  69
# [12,] 12 467
# [13,] 13 270
# [14,] 14   0
# [15,] 15   2
# [16,] 16   0
# [17,] 17  58
# [18,] 18   8
# [19,] 19 131
# [20,] 20  78
# [21,] 21   0
# [22,] 22   4
# [23,] 23 179
# [24,] 24   0
# [25,] 25   0
# [26,] 26   0
# [27,] 27   0
# [28,] 28  94
# [29,] 29   0
# [30,] 30   1
# [31,] 31   1
# [32,] 32 104
# [33,] 33   1
# [34,] 34 296
# [35,] 35   9
# [36,] 36 186
# [37,] 37   0
# [38,] 38   1
# [39,] 39   0
# [40,] 40   0
# [41,] 41   0
# [42,] 42   0
# [43,] 43 255
# [44,] 44  92
# [45,] 45   1
# [46,] 46   5
# [47,] 47   0
# [48,] 48  16
# [49,] 49   0
# [50,] 50   0
# [51,] 51   0
# [52,] 52  99
# [53,] 53   0
###########################################


# Plot correlation
M1 <- abs(cor(pmltrain3))
diag(M1) <- 0
image(M1) # FIGURE
# The last column represents the variable "classe"'s correlation with all predictors
which(M1[, ncol(M1)] > .3)
# pitch_forearm 
# 41
plot(pmltrain3$pitch_forearm, pmltrain3$classe, color = typeColor)
which(M1[, ncol(M1)] > .2)
# magnet_belt_y   accel_arm_x  magnet_arm_x  magnet_arm_y pitch_forearm 
# 12            21            24            25            41 
which(M1[, ncol(M1)] > .1)
# magnet_belt_y       magnet_belt_z           pitch_arm     total_accel_arm         accel_arm_x        magnet_arm_x 
# 12                  13                  15                  17                  21                  24 
# magnet_arm_y        magnet_arm_z    accel_dumbbell_x   magnet_dumbbell_z       pitch_forearm total_accel_forearm 
# 25                  26                  34                  39                  41                  43 
# accel_forearm_x    magnet_forearm_x    magnet_forearm_y 
# 47                  50                  51 

# Choose the last set of predictors, since they have relatively more obvious correlations with classe.


## Correlation matrix among the predictors to further narrow down variables
train_small <- pmltrain3[, c(which(M1[, ncol(M1)] > .1), ncol(pmltrain3))]
M2 <- abs(cor(train_small))
diag(M2) <- 0
ind <- which(M2 > 0.8, arr.ind=T)
#                row col
# magnet_arm_x   6   5
# accel_arm_x    5   6
# magnet_arm_z   8   7
# magnet_arm_y   7   8


#################################
# Optional:
# This is an example of a full PCA
pmltrain2 <- sapply(pmltrain1[1:(ncol(pmltrain1)-1)], as.numeric)
pmltrain2 <- as.numeric(pmltrain2)
# pmltrain2 <- sapply(pmltrain1[1:(ncol(pmltrain1)-1)], function(x) {if (is.numeric(x) != T) {x <- as.numeric(x)}})
M <- abs(cor(pmltrain2))
diag(M) <- 0
corind <- which(M > 0.8, arr.ind=T)
# There are lots:
# row col
# yaw_belt           3   1
# total_accel_belt   4   1
# accel_belt_y       9   1
# accel_belt_z      10   1
# accel_belt_x       8   2
# magnet_belt_x     11   2
# roll_belt          1   3
# roll_belt          1   4
# accel_belt_y       9   4
# accel_belt_z      10   4
# pitch_belt         2   8
# magnet_belt_x     11   8
# roll_belt          1   9
# total_accel_belt   4   9
# accel_belt_z      10   9
# roll_belt          1  10
# total_accel_belt   4  10
# accel_belt_y       9  10
# pitch_belt         2  11
# accel_belt_x       8  11
# gyros_arm_y       19  18
# gyros_arm_x       18  19
# magnet_arm_x      24  21
# accel_arm_x       21  24
# magnet_arm_z      26  25
# magnet_arm_y      25  26
# accel_dumbbell_x  34  28
# accel_dumbbell_z  36  29
# gyros_dumbbell_z  33  31
# gyros_forearm_z   46  31
# gyros_dumbbell_x  31  33
# gyros_forearm_z   46  33
# pitch_dumbbell    28  34
# yaw_dumbbell      29  36
# gyros_forearm_z   46  45
# gyros_dumbbell_x  31  46
# gyros_dumbbell_z  33  46
# gyros_forearm_y   45  46
M[33, 46] # E.g., cell [33, 46] has a high correlation number.
rownames(M)[33] # row 33 = gyros_dumbbell_z
colnames(M)[46] # col 46 = gyros_forearm_z
# This means the gyros of dumbbell and forearm are correlated in the z direction!
M[33, 31] # Cell [33, 31] is also  highly correlated.
rownames(M)[33] # row 33 = gyros_dumbbell_z
colnames(M)[31] # row 31 = gyros_dumbbell_x
# This means the gyros of dumbell in x and z directions is correlated!

# 1 is correlated w 3, 4, 9
# 3 is correlated w 1
# 4 is correlated w 1, 9, 10
# 9 is correlated w 1, 4, 10
# 10 is correlated w 1, 4, 9
# 2, 8, 11 are correlated w the other 2
# 18, 19 are correlated w each other
# 21, 24 are correlated w each other
# 25, 26
# 28, 34
# 29, 36
# 31, 33, 46 are correlated w the other 2
# 45 is correlated w 46

# Conclusion from this section: It's necessary to do PCA.
# Conclusion from this section wo PCA:
# - Discard the variable "yaw_belt
# - Choose 1 variable from the set 1, 4, 9, 10, etc etc


# More optional PCA exploration:
# Plot the gyros_dumbell of z vs x
# First remove 1 outlier
a1 <- pmltrain2[, 31]
a1[5373] <- 0
a2 <- pmltrain2[, 33]
a2[5373] <- 0
plot(a1, a2) # Indeed look correlated!

# Do a pairs featureplot for one of the actions (e.g. roll)
featurePlot(x = pmltrain2[, c("roll_belt", "roll_arm", "roll_dumbbell", "roll_forearm")],
            y = pmltrain1$classe,
            plot = "pairs")
# Do a pairs featureplot for one of the action objects (e.g. belt)
featurePlot(x = pmltrain2[, c("roll_belt", "pitch_belt", "yaw_belt")],
            y = pmltrain1$classe,
            plot = "pairs")

# Do a PCA. See if any could be compressed
smallpml <- pmltrain2[, unique(corind[, 2])]
prComp <- prcomp(smallpml)
# The components in prcomp object are basically the result of linear combinations of all the cols included in PCA.
plot(prComp$x[,1],prComp$x[,2])
prComp$rotation # shows the coefficients

typeColor <- as.numeric(pmltrain1$classe)*1
# prComp2 <- prcomp(log10(pmltrain1[, -53] + 1))
prComp2 <- prcomp(pmltrain1[, -53])
plot(prComp$x[, 1], prComp$x[, 2], col = typeColor, xlab = "PC1", ylab = "PC2") # Black, red, green: some separation...
plot(prComp$x[, 2], prComp$x[, 3], col = typeColor, xlab = "PC1", ylab = "PC2")

# The above is optional.
###########################################


## ??? Build model:

library(caret)
library(e1071)

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
    validation <- train_small[-folds[[i]], ]
    preProc <- preProcess(train_small[, -53], method = "pca", thresh = 0.9)
}

validation <- train_small[-folds[[1]], ]
holdout <- train_small[folds[[1]], ]

# Build model without PCA
# This DOESN'T WORK because glm is logistic regression, for binary classification only.
modelFit1 <- train(classe ~.,
                   data = validation
                   method = "glm")
test1 <- predict(modelFit1, newdata = holdout) # use one of the folds
confusionMatrix(test1, holdout$classe)

# Build model with PCA
# Note that for a full data PCA,
# preProc <- preProcess(train_small[, -16], method = "pca", thresh = 0.9)
# 9 variables are needed to achieve 90% explanation of variance
preProc <- preProcess(validation[, -16], method = "pca", thresh = 0.9)

trainPC <- predict(preProc, validation[, -16])
modelFitPC <- train(validation$classe ~ .,
                    method = "glm",
                    data = trainPC)
testPC <- predict(preProc, holdout[, -16])
confusionMatrix(predict(modelFitPC, testPC), holdout$classe)$overall[1]


# ?? How to get what the PCs represent? How do we predict while making physical sense?

modelFit <- train(classe ~.,
                  data = train_small,
                  method = "glm")
# To find the actual fitted values from the final model:
modelFit$finalModel
########


