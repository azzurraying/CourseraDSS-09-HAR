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
sapply(pmltrain, class)
colnames(pmltest)

## Clean
# Remove the following columns:
# - 1st 7 (contains user names and other non-numerical data)
# - Blank columns ("")
# - NA columns
pmltrain1 <- pmltrain[-c(1:7, which(pmltrain[1, ] == ""), which(is.na(pmltrain[1, ])))]
ncol(pmltrain1) # 53
# Convert all to numeric and do correlation
pmltrain2 <- as.data.frame(sapply(pmltrain1, as.numeric))
pmltrain2$classe <- pmltrain$classe

pmltest1 <- pmltest[-c(1:7, which(pmltest[1, ] == ""), which(is.na(pmltest[1, ])))]
pmltest2 <- as.data.frame(sapply(pmltest1, as.numeric))
pmltest2$classe <- pmltest$classe


## Explore
# First, use correlation to generate a "pairs" plot
M1 <- abs(cor(pmltrain2))
diag(M1) <- 0
image(M1) # FIGURE
# The last column represents the variable "classe"'s correlation with all predictors
which(M1[, ncol(M1)] > .1)
# Choose the set of predictors, since they have relatively more obvious correlations with classe (> 0.1).


## Correlation matrix among the predictors to further narrow down variables
train_small <- pmltrain2[, c(which(M1[, ncol(M1)] > .1), ncol(pmltrain2))]
train_small$classe <- pmltrain$classe
M2 <- abs(cor(train_small[, -16]))
diag(M2) <- 0
ind <- which(M2 > 0.8, arr.ind=T)

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

aFull <- numeric(5)
aSel <- numeric(5)
aPCA <- numeric(5)
aSel2 <- numeric(5)

for(i in 1:5) {
  
  # 1. Build models including all predictors (without feature selection)
  # Use Decision Trees, without PCA
  # Create validation folds
  validation_full <- pmltrain2[-folds[[i]], ]
  holdout_full <- pmltrain2[folds[[i]], ]
  # Fit models
  modelFit_DT_full <- ctree(classe ~ ., data = validation_full)
  predicted_DT_full <- predict(modelFit_DT_full, newdata = holdout_full)
  aFull[i] <- confusionMatrix(predicted_DT_full, holdout_full$classe)$overall[1]
  
  # 2. Include just the predictors that are relatively more correlated with classe.
  # Use Decision Trees, without PCA
  validation <- train_small[-folds[[i]], ]
  holdout <- train_small[folds[[i]], ]
  # Fit models
  modelFit_DT <- ctree(classe ~ ., data = validation)
  predicted_DT <- predict(modelFit_DT, newdata = holdout)
  confusionMatrix(predicted_DT, holdout$classe)
  # Get accuracy
  aSel[i] <- confusionMatrix(predicted_DT, holdout$classe)$overall[1] # Note: this figure is exactly the same as predicted_DT_full!
  
  # 3. Build model with less predictors, with Decision Trees, including a PCA
  # Note: 9 variables are needed to achieve 90% explanation of variance
  preProc <- preProcess(validation[, -16], method = "pca", thresh = 0.9)
  trainPC <- predict(preProc, validation[, -16])
  # Fit models
  modelFitPC <- ctree(validation$classe ~ ., data = trainPC)
  testPC <- predict(preProc, holdout[, -16])
  predictedPC <- predict(modelFitPC, testPC)
  # Get accuracy
  aPCA[i] <- confusionMatrix(predictedPC, holdout$classe)$overall[1] # 73% Accurate. Not as good as without using PCA!
  
  # 4. Build model with even less predictors, discarding "accel_arm_x" (correlated to magnet_arm_x) and magnet_arm_y (correlated to magnet_arm_z)
  train_smaller <- train_small[-c(5, 7)]
  validation_smaller <- train_small[-folds[[i]], ]
  holdout_smaller <- train_small[folds[[i]], ]
  # Fit models
  modelFit_DT_sml <- ctree(classe ~ ., data = validation_smaller)
  predicted_DT_sml <- predict(modelFit_DT_sml, newdata = holdout_smaller)
  aSel2[i] <- confusionMatrix(predicted_DT_sml, holdout_smaller$classe)$overall[1] # Accuracy doesn't change much. Became slightly smaller.
  
}

aFull; mean(aFull)
aSel; mean(aSel)
aPCA; mean(aPCA)
aSel2; mean(aSel2)

modelFit_DT_full <- ctree(classe ~ ., data = pmltrain2)
predicted_DT_full <- predict(modelFit_DT_full, newdata = pmltest2)

answers = predicted_DT_full
    pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
            filename = paste0("problem_id_",i,".txt")
            write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
    }
setwd("/Users/yingjiang/Dropbox/Learnings/Stats_data/Coursework/Data_science_spec/Data_science_C8/Project/Answers")
pml_write_files(answers)