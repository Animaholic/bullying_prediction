### Authors: Hongmin Huang, Guohao Shen ###

rm(list=ls()); cat("\014") # Clear workspace and console
# libraries
library(modeest)
library(caret)
library(FSelector)
library(RWeka)
library(data.table)
library(Boruta)
library(ggcorrplot)
library(GGally)
library(rsample)
library(e1071)
library(class)
library(randomForest)
library(gbm)
library(glmnet)

###############################################################################
### data preprocessing
df <- read.csv("project_data.csv", na.strings = "?") # load the dataset
dim(df)

# get the number of missing values
sum(is.na(df))

## data cleaning
na_cols <- colnames(df)[colSums(is.na(df)) > 0] # identify columns with missing values

# replace missing values with column modes because all attributes are nominal
for (col in na_cols) {
  df[[col]][is.na(df[[col]])] <- mfv(df[[col]], na_rm = TRUE)
}

sum(is.na(df)) # check the number of missing values is zero

## data reduction

# remove duplicate attributes
df <- df[!duplicated(as.list(df))]
dim(df)

# dimensionality reduction
# near zero variance
nearZeroVar(df, names = TRUE)
df <- df[, -nearZeroVar(df)]
dim(df)

# collinearity
corr <- cor(df[1:ncol(df)])
highCorr <- findCorrelation(corr, cutoff = 0.7, names = TRUE)
length(highCorr)
highCorr
df <- df[, -findCorrelation(corr, cutoff = 0.7)]
dim(df)

# # cfs
# subset <- cfs(class ~., df)
# df.cfs <- as.simple.formula(subset, "class")
# df.cfs
att1 <- c("vs007","vs030","vs064a","vs066","vs068","vs131","SchCultureRecode")
# 
# # info gain
# df2 <- copy(df)
# df2 <- as.data.frame(unclass(df2), stringsAsFactors = TRUE)
# df2$class <- factor(df2$class)
# df2.infogain <- InfoGainAttributeEval(class ~., data = df2)
# sorted.features <- sort(df2.infogain, decreasing = TRUE)
# sorted.features[1:10]
# att2 <- c("vs129","SchCultureRecode","vs066","vs068","vr16","vs130","vs046","vs061","vs064a","vs060")
# 
# # Boruta
# df.boruta <- Boruta(class ~., data = df)
# df.boruta
# att3 <- getSelectedAttributes(df.boruta, withTentative=FALSE)
# att3
# 
# # find common elements
# attShared <- intersect(intersect(att1,att2), att3)
# attShared
# attShared <- c(attShared, "class")

# select important attributes
# create attCopy for temporary use (no need to run feature selection)
attCopy <- c(att1, "class")
df <- subset(df, select = attCopy)
head(df)
table(df$class)

# over and undersampling
library(ROSE)
oversampled_data <- ovun.sample(class ~ ., data = df, method = "both")
oversampled_data <- oversampled_data$data
head(oversampled_data)

df <- data.frame(oversampled_data)
head(df)
table(df$class)

# save the preprocessed data
#write.csv(df, "preprocessed_data.csv", row.names = FALSE)

# correlation plot
sub_df <- subset(df, select = att1)
cor(sub_df)
ggpairs(sub_df)
ggcorrplot(cor(sub_df), method = "square", lab = TRUE)


###############################################################################
### classification
df$class <- factor(df$class)
df[1:4] <- scale(df[1:4]) # standardize all 4 features

# train-test split
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)

# 10-fold cross-validation
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)


## Model 1: knn
model_1 <- function(train,test) {
  knn_model <- train(class ~ ., data = train, method = "knn", trControl=train_control, preProcess = c("center", "scale"), tuneLength = 100)
  predictions <- predict(knn_model, test)
  cm <- confusionMatrix(predictions, test$class)
  return(list(table=cm$table, overall=cm$overall, byClass=cm$byClass))
}

result1 <- model_1(train, test)
result1

## Model 2: SVM
tuneGrid <- expand.grid(sigma = seq(0.1, 0.4, by = 0.05), C = seq(1.0, 2.0, by = 0.1))

model_2 <- function(train,test) {
  svm_model <- train(class ~ ., data = train, method = "svmRadial", trControl = train_control, tuneGrid = tuneGrid)
  predictions <- predict(svm_model, test)
  cm <- confusionMatrix(predictions, test$class)
  return(list(table=cm$table, overall=cm$overall, byClass=cm$byClass))
}

result2 <- model_2(train, test)
result2

## Model 3: Random Forest
tuneGrid <- expand.grid(.mtry = c(1:10))

model_3 <- function(train, test) {
  set.seed(31)
  rf_model <- train(class ~ ., data = train, method = "rf", trControl = train_control, tuneGrid = tuneGrid)
  predictions <- predict(rf_model, test)
  cm <- confusionMatrix(predictions, test$class)
  return(list(table=cm$table, overall=cm$overall, byClass=cm$byClass))
}

result3 <- model_3(train, test)
result3

## Model 4: Gradient Boosting
tuneGrid <- expand.grid(.interaction.depth = c(1, 5, 10), .n.trees = seq(100, 500, by = 50), .shrinkage = c(0.01, 0.1), .n.minobsinnode = 20)

model_4 <- function(train, test) {
  set.seed(31)
  gbm_model <- train(class ~ ., data = train, method = "gbm", trControl = train_control, tuneGrid = tuneGrid, verbose = FALSE)
  predictions <- predict(gbm_model, test)
  cm <- confusionMatrix(predictions, test$class)
  return(list(table=cm$table, overall=cm$overall, byClass=cm$byClass))
}

result4 <- model_4(train, test)
result4

## Model 5: Logistic Regression
tuneGrid <- expand.grid(.penalty = c(0.01, 0.1, 1, 10, 100), .tol = c(1e-8, 1e-6, 1e-4))

model_5 <- function(train, test) {
  set.seed(31)
  log_model <- train(class ~ ., data = train, method = "glmnet", trControl = train_control, tuneGrid = tuneGrid)
  predictions <- predict(log_model, test)
  cm <- confusionMatrix(predictions, test$class)
  return(list(table=cm$table, overall=cm$overall, byClass=cm$byClass))
}

result5 <- model_5(train, test)
result5
