#Importing the dataset
setwd("C:\\Users\\User\\Desktop")
getwd()
dataset = read.csv("House_Price_Dataset.csv")

#Dropping id and date columns
dataset = dataset[3:21]
summary(dataset)
str(dataset)

#Checking for missing values
colSums(is.na(dataset))

#Encoding categorical data
dataset$waterfront = factor(dataset$waterfront, 
                            levels = c("No", "Yes"), 
                            labels = c(0, 1))
#Changing back waterfront to a numeric variable
dataset$waterfront = as.numeric(as.character(dataset$waterfront))

#Finding the correlation coefficient matrix of the all the variables with price
round(cor(dataset), 2)
#From the correlation coefficient matrix, it was discovered that only bathrooms, sqft_living, grade, sqft_above, sqft_living15 has strong correlations with price

#Dropping zipcode, latitude and longitude
dataset$zipcode = NULL
dataset$lat = NULL
dataset$long = NULL

#Removing records with bedrooms & bathrooms = 0
dataset = dataset[!(dataset$bedrooms == 0 | dataset$bathrooms == 0),]

#Changing yr_built to the number of years since built to 2022
dataset$yrs_since_built = 2022 - dataset$yr_built
dataset$yr_built = NULL #dropping yr_built

#Changing yr_renovated to the number of years since renovated to 2022
dataset$yrs_since_renovated = 2022 - dataset$yr_renovated
dataset$yr_renovated = NULL #dropping yr_renovated
#However, some of the houses have never been renovated, 
#hence the new column created will have values of "2022", which does not make sense.
#Therefore, changing "2022" to the number of years since built.
dataset$yrs_since_renovated = ifelse(dataset$yrs_since_renovated == 2022, 
                                     dataset$yrs_since_built, 
                                     dataset$yrs_since_renovated)
dataset = dataset[c(2:16, 1)]

#Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$price, SplitRatio = 0.8)
training_set = subset(dataset, split == T)
test_set = subset(dataset, split == F)

#Feature scaling
training_set[,1:15] = scale(training_set[,1:15])
test_set[,1:15] = scale(test_set[,1:15])

#Fitting Linear Regression to the Training set
linear_regression = lm(formula = price ~ .,
                      data = training_set)
summary(linear_regression)

#Backward elimination of Linear Regression model
linear_regression2 = lm(formula = price ~ bedrooms + bathrooms + sqft_living + floors + waterfront + view + condition + grade + sqft_living15 + sqft_lot15 + yrs_since_built,
                        data = training_set)
summary(linear_regression2)

#Predicting the Test set results
y_pred1 = predict(linear_regression2, newdata = test_set)
table(y_pred1, test_set$price)
summary(y_pred1)

library(caret)
library(MLmetrics)
MAE(y_pred1, test_set$price)
RMSE(y_pred1, test_set$price)
MAPE(y_pred1, test_set$price)

#Fitting the Random Forest Regression to the dataset
library(randomForest)
set.seed(123)
random_forest = randomForest(x = dataset[1:15], 
                             y = dataset$price,
                             ntree = 100)

#Prediction a new result
y_pred2 = predict(random_forest, dataset)
table(y_pred2, dataset$price)
summary(y_pred2)

MAE(y_pred2, dataset$price)
RMSE(y_pred2, dataset$price)
MAPE(y_pred2, dataset$price)

#Fitting XGBoost to the Training set
#install.packages("xgboost")
library(xgboost)
xgboost_regression = xgboost(data = as.matrix(training_set[-16]), 
                     label = training_set$price,
                     nrounds = 10)
summary(xgboost_regression)

y_pred3 = predict(xgboost_regression, newdata = as.matrix(test_set[-16]))
table(y_pred3, test_set$price)
summary(y_pred3)

MAE(y_pred3, test_set$price)
RMSE(y_pred3, test_set$price)
MAPE(y_pred3, test_set$price)

#-------------------------------------------------------------------------

#Cross validation Linear Regression
library(caret)
set.seed(123)
train.control <- trainControl(method = "cv", number = 10, verboseIter = T)

lm_cv <- train(price ~ ., data = training_set, method = "lm", 
               trControl = train.control)
summary(lm_cv)

lm_cv2 <- train(price ~ bedrooms + bathrooms + sqft_living + floors + waterfront + view + condition + grade + sqft_living15 + sqft_lot15 + yrs_since_built,
                data = training_set, method = "lm", 
                trControl = train.control)
summary(lm_cv2)

y_pred7 = predict(lm_cv2, test_set)
table(y_pred7, test_set$price)
summary(y_pred7)

MAE(y_pred7, test_set$price)
RMSE(y_pred7, test_set$price)
MAPE(y_pred7, test_set$price)

#Cross validation XGBoost
xgb_cv <- train(price ~ ., data = training_set, method = "xgbDART", 
                trControl = train.control)
summary(xgb_cv)

y_pred8 = predict(xgb_cv, test_set)
table(y_pred8, test_set$price)
summary(y_pred8)

MAE(y_pred8, test_set$price)
RMSE(y_pred8, test_set$price)
MAPE(y_pred8, test_set$price)

library(xgboost)
xgb = train(price ~ ., training_set, method = "xgbTree", metric = "RMSE")

#-------------------------------Applying PCA-------------------------------

library(caret)
library(e1071)
pca = preProcess(x = training_set[-16], method = "pca", pcaComp = 2)
training_set_pca = predict(pca, training_set)
training_set_pca = training_set_pca[c(2,3,1)]
test_set_pca = predict(pca, test_set)
test_set_pca = test_set_pca[c(2,3,1)]
dataset_pca = rbind(training_set_pca, test_set_pca)
dataset_pca = dataset_pca[c(2,3,1)]

#Fitting Simple Linear Regression to the Training set
linear_regression_pca = lm(formula = price ~ .,
                           data = training_set_pca)
summary(linear_regression_pca)

#Predicting the Test set results
y_pred4 = predict(linear_regression_pca, newdata = test_set_pca)
table(y_pred4, test_set_pca$price)
summary(y_pred4)

library(caret)
library(MLmetrics)
MAE(y_pred4, test_set_pca$price)
RMSE(y_pred4, test_set_pca$price)
MAPE(y_pred4, test_set_pca$price)

#Fitting the Random Forest Regression to the dataset
library(randomForest)
set.seed(123)
random_forest_pca = randomForest(x = training_set_pca[1:2], 
                                 y = training_set_pca$price,
                                 ntree = 100)

#Prediction a new result
y_pred5 = predict(random_forest_pca, test_set_pca)
table(y_pred5, test_set_pca$price)
summary(y_pred5)

MAE(y_pred5, test_set_pca$price)
RMSE(y_pred5, test_set_pca$price)
MAPE(y_pred5, test_set_pca$price)

#Fitting XGBoost to the Training set
#install.packages("xgboost")
library(xgboost)
xgboost_regression_pca = xgboost(data = as.matrix(training_set_pca[-3]), 
                                 label = training_set_pca$price,
                                 nrounds = 10)
summary(xgboost_regression_pca)

y_pred6 = predict(xgboost_regression_pca, newdata = as.matrix(test_set_pca[-3]))
table(y_pred6, test_set_pca$price)
summary(y_pred6)

MAE(y_pred6, test_set_pca$price)
RMSE(y_pred6, test_set_pca$price)
MAPE(y_pred6, test_set_pca$price)

#--------------------------------------------------------------------------

#Cross validation XGBoost - PCA reduced dataset
library(caret)
set.seed(123)
train.control <- trainControl(method = "cv", number = 10, verboseIter = T)

lm_cv_pca <- train(price ~ ., data = training_set_pca, method = "lm", 
                   trControl = train.control)
summary(lm_cv_pca)

y_pred9 = predict(lm_cv_pca, test_set_pca)
table(y_pred9, test_set_pca$price)
summary(y_pred9)

MAE(y_pred9, test_set_pca$price)
RMSE(y_pred9, test_set_pca$price)
MAPE(y_pred9, test_set_pca$price)

#Cross validation XGBoost - PCA reduced dataset
xgb_cv_pca <- train(price ~ ., data = training_set_pca, method = "xgbDART", 
                    trControl = train.control)
summary(xgb_cv_pca)

y_pred10 = predict(xgb_cv_pca, test_set_pca)
table(y_pred10, test_set_pca$price)
summary(y_pred10)

MAE(y_pred10, test_set_pca$price)
RMSE(y_pred10, test_set_pca$price)
MAPE(y_pred10, test_set_pca$price)
