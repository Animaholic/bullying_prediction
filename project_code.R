### Authors: Hongmin Huang, Guohao Shen ###

rm(list=ls()); cat("\014") # Clear workspace and console
start_time <- Sys.time() # start time
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
library(ROSE)
library(e1071)
library(class)
library(randomForest)
library(gbm)
library(glmnet)
library(naivebayes)
library(ROCR)
library(mlbench)
library(pROC)

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

# save the preprocessed data
write.csv(df, "preprocessed_data.csv", row.names = FALSE)

###############################################################################
### classification
df$class <- factor(df$class) # apply the factor function.

# train-test split
set.seed(31)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)

# over and under sampling train set
oversampled_data <- ovun.sample(class ~ ., data = train, method = "both")
oversampled_data <- oversampled_data$data
train <- as.data.frame(oversampled_data) # convert list to data frame
table(train$class)
dim(train)

# cfs
subset <- cfs(class ~., train)
train.cfs <- as.simple.formula(subset, "class")
train.cfs

att1 <- c("v3020", "v3081", "vs030", "vs129", "vs130", "vs045", "vs046", 
          "vs049", "vs051", "vs064a", "vs066", "vs067", "vs068", "SchCultureRecode")

# info gain
train2 <- train
train2 <- as.data.frame(unclass(train2), stringsAsFactors = TRUE) # convert to data frame
train2$class <- factor(train2$class) # apply the factor function
train2.infogain <- InfoGainAttributeEval(class ~., data = train2) # evaluation
sorted.features <- sort(train2.infogain, decreasing = TRUE) # sort the features
sorted.features[1:20] # pick the first 20 features

att2 <- c("SchCultureRecode","vs129","vs066","vs045","vs068","vs130","vs049",
          "vs046","vs064a","v3081","vr16","vs051","vs061","vs067","v3020",
          "vs060","vs057","vs007","vs047","vs126")

# # Boruta
# set.seed(31)
# df.boruta <- Boruta(class ~., data = df)
# df.boruta
# att3 <- getSelectedAttributes(df.boruta, withTentative=FALSE)
# att3

# find common elements
# attShared <- intersect(intersect(att1,att2), att3)
attShared <- c("v3020","v3081","vs129","vs130","vs045","vs046","vs049","vs051",
               "vs064a","vs066","vs067","vs068","SchCultureRecode")

# correlation plot
corr <- subset(train, select = attShared)
ggcorrplot(cor(corr), method = "square", lab = TRUE)

attShared <- c(attShared, "class") # add class attribute

# select important attributes
train <- subset(train, select = attShared)
test <- subset(test, select = attShared)

# scale
train[1:13] <- scale(train[1:13])
test[1:13] <- scale(test[1:13])

# save best_training.csv and best_testing.csv
write.csv(train, "best_training.csv", row.names = FALSE)
write.csv(test, "best_testing.csv", row.names = FALSE)

# 10-fold cross-validation
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)


# Model 1: knn
model_1 <- function(train, test) {
  knn_model <<- train(class ~ ., data = train, method = "knn", trControl=train_control, 
                      preProcess = c("center", "scale"), tuneLength = 100) # build the model
  predictions <- predict(knn_model, test) # make prediction
  predictions <- factor(predictions, levels = levels(test$class)) # apply the factor function
  pred_perf <- prediction(as.numeric(predictions), labels = as.numeric(test$class)) # convert to numeric
  auc=as.numeric(performance(pred_perf, measure = "auc")@y.values) # auc
  cm1 <- confusionMatrix(predictions, test$class,positive="1") # confusion matrix of class 1
  tb1=cm1$table
  TP1<-cm1$table[1,1]
  TN1<-cm1$table[2,2]
  FP1<-cm1$table[1,2]
  FN1<-cm1$table[2,1]
  MCC1 <- ((TP1*TN1)-(FP1*FN1))/((TP1+FP1)^0.5*(TP1+FN1)^0.5*(TN1+FP1)^0.5*(TN1+FN1)^0.5) # MCC of class 1's cm
  cm2 <- confusionMatrix(predictions, test$class,positive="2") # confusion matrix of class 2
  TP2<-cm2$table[1,1]
  TN2<-cm2$table[2,2]
  FP2<-cm2$table[1,2]
  FN2<-cm2$table[2,1]
  MCC2 <- ((TP2*TN2)-(FP2*FN2))/((TP2+FP2)^0.5*(TP2+FN2)^0.5*(TN2+FP2)^0.5*(TN2+FN2)^0.5)

  # Create a data frame with performance measures of Class 1
  performance_df1 <- data.frame(Measure = c("TP rate", "FP rate", "Precision",
                                            "Recall", "F1-Score", "MCC"),
                                Value = c(cm1$byClass["Sensitivity"],
                                          cm1$byClass["Specificity"],
                                          cm1$byClass["Precision"],
                                          cm1$byClass["Recall"],
                                          cm1$byClass["F1"],
                                          MCC1))

  # Create a bar plot
  performance_df1$Measure <- factor(performance_df1$Measure,
                                    levels = unique(performance_df1$Measure))
  plot1 <- ggplot(performance_df1, aes(x = Measure, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label=sprintf("%.3f", Value)), vjust=1.6, color="white", size=3.5) +
    labs(title = "Performance Measures of Class 1", x="Measure", y="Value")

  print(plot1) # Display the plot

  # Create a data frame with performance measures of Class 2
  performance_df2 <- data.frame(Measure = c("TP rate", "FP rate", "Precision",
                                            "Recall", "F1-Score", "MCC"),
                                Value = c(cm2$byClass["Sensitivity"],
                                          cm2$byClass["Specificity"],
                                          cm2$byClass["Precision"],
                                          cm2$byClass["Recall"],
                                          cm2$byClass["F1"],
                                          MCC2))

  # Create a bar plot
  performance_df2$Measure <- factor(performance_df2$Measure,
                                    levels = unique(performance_df2$Measure))
  plot2 <- ggplot(performance_df2, aes(x = Measure, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label=sprintf("%.3f", Value)), vjust=1.6, color="white", size=3.5) +
    labs(title = "Performance Measures of Class 2", x="Measure", y="Value")

  print(plot2) # Display the plot

  # Calculate the ROC curve
  perf <- performance(pred_perf,"tpr","fpr")
  plot(perf,colorize=TRUE)

  # print the results
  print(list(table1=cm1$table, overall1=cm1$overall, byClass1=cm1$byClass,
             table2=cm2$table, overall2=cm2$overall, byClass2=cm2$byClass,
             MCC1=MCC1,MCC2=MCC2,auc=auc))
}

result1 <- model_1(train, test)


## Model 2: SVM
tuneGrid <- expand.grid(sigma = seq(0.1, 0.4, by = 0.05), C = seq(1.0, 2.0, by = 0.1))

model_2 <- function(train,test) {
  svm_model <<- train(class ~ ., data = train, method = "svmRadial", trControl = train_control, tuneGrid = tuneGrid) # build the model
  predictions <- predict(svm_model, test)
  predictions <- factor(predictions, levels = levels(test$class)) # apply the factor function
  pred_perf <- prediction(as.numeric(predictions), labels = as.numeric(test$class)) # convert to numeric
  auc=as.numeric(performance(pred_perf, measure = "auc")@y.values) # auc
  cm1 <- confusionMatrix(predictions, test$class,positive="1") # confusion matrix of class 1
  tb1=cm1$table
  TP1<-cm1$table[1,1]
  TN1<-cm1$table[2,2]
  FP1<-cm1$table[1,2]
  FN1<-cm1$table[2,1]
  MCC1 <- ((TP1*TN1)-(FP1*FN1))/((TP1+FP1)^0.5*(TP1+FN1)^0.5*(TN1+FP1)^0.5*(TN1+FN1)^0.5) # MCC of class 1's cm
  cm2 <- confusionMatrix(predictions, test$class,positive="2") # confusion matrix of class 2
  TP2<-cm2$table[1,1]
  TN2<-cm2$table[2,2]
  FP2<-cm2$table[1,2]
  FN2<-cm2$table[2,1]
  MCC2 <- ((TP2*TN2)-(FP2*FN2))/((TP2+FP2)^0.5*(TP2+FN2)^0.5*(TN2+FP2)^0.5*(TN2+FN2)^0.5)
  
  # Create a data frame with performance measures of Class 1
  performance_df1 <- data.frame(Measure = c("TP rate", "FP rate", "Precision",
                                            "Recall", "F1-Score", "MCC"),
                                Value = c(cm1$byClass["Sensitivity"],
                                          cm1$byClass["Specificity"],
                                          cm1$byClass["Precision"],
                                          cm1$byClass["Recall"],
                                          cm1$byClass["F1"],
                                          MCC1))
  
  # Create a bar plot
  performance_df1$Measure <- factor(performance_df1$Measure,
                                    levels = unique(performance_df1$Measure))
  plot1 <- ggplot(performance_df1, aes(x = Measure, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label=sprintf("%.3f", Value)), vjust=1.6, color="white", size=3.5) +
    labs(title = "Performance Measures of Class 1", x="Measure", y="Value")
  
  print(plot1) # Display the plot
  
  # Create a data frame with performance measures of Class 2
  performance_df2 <- data.frame(Measure = c("TP rate", "FP rate", "Precision",
                                            "Recall", "F1-Score", "MCC"),
                                Value = c(cm2$byClass["Sensitivity"],
                                          cm2$byClass["Specificity"],
                                          cm2$byClass["Precision"],
                                          cm2$byClass["Recall"],
                                          cm2$byClass["F1"],
                                          MCC2))
  
  # Create a bar plot
  performance_df2$Measure <- factor(performance_df2$Measure,
                                    levels = unique(performance_df2$Measure))
  plot2 <- ggplot(performance_df2, aes(x = Measure, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label=sprintf("%.3f", Value)), vjust=1.6, color="white", size=3.5) +
    labs(title = "Performance Measures of Class 2", x="Measure", y="Value")
  
  print(plot2) # Display the plot
  
  # Calculate the ROC curve
  perf <- performance(pred_perf,"tpr","fpr")
  plot(perf,colorize=TRUE)
  
  # print the results
  print(list(table1=cm1$table, overall1=cm1$overall, byClass1=cm1$byClass,
             table2=cm2$table, overall2=cm2$overall, byClass2=cm2$byClass,
             MCC1=MCC1,MCC2=MCC2,auc=auc))
}

result2 <- model_2(train, test)


## Model 3: Random Forest
tuneGrid <- expand.grid(.mtry = c(1:10))

model_3 <- function(train, test) {
  set.seed(31)
  rf_model <<- train(class ~ ., data = train, method = "rf", trControl = train_control, tuneGrid = tuneGrid) # build the model
  predictions <- predict(rf_model, test)
  predictions <- factor(predictions, levels = levels(test$class)) # apply the factor function
  pred_perf <- prediction(as.numeric(predictions), labels = as.numeric(test$class)) # convert to numeric
  auc=as.numeric(performance(pred_perf, measure = "auc")@y.values) # auc
  cm1 <- confusionMatrix(predictions, test$class,positive="1") # confusion matrix of class 1
  tb1=cm1$table
  TP1<-cm1$table[1,1]
  TN1<-cm1$table[2,2]
  FP1<-cm1$table[1,2]
  FN1<-cm1$table[2,1]
  MCC1 <- ((TP1*TN1)-(FP1*FN1))/((TP1+FP1)^0.5*(TP1+FN1)^0.5*(TN1+FP1)^0.5*(TN1+FN1)^0.5) # MCC of class 1's cm
  cm2 <- confusionMatrix(predictions, test$class,positive="2") # confusion matrix of class 2
  TP2<-cm2$table[1,1]
  TN2<-cm2$table[2,2]
  FP2<-cm2$table[1,2]
  FN2<-cm2$table[2,1]
  MCC2 <- ((TP2*TN2)-(FP2*FN2))/((TP2+FP2)^0.5*(TP2+FN2)^0.5*(TN2+FP2)^0.5*(TN2+FN2)^0.5)
  
  # Create a data frame with performance measures of Class 1
  performance_df1 <- data.frame(Measure = c("TP rate", "FP rate", "Precision",
                                            "Recall", "F1-Score", "MCC"),
                                Value = c(cm1$byClass["Sensitivity"],
                                          cm1$byClass["Specificity"],
                                          cm1$byClass["Precision"],
                                          cm1$byClass["Recall"],
                                          cm1$byClass["F1"],
                                          MCC1))
  
  # Create a bar plot
  performance_df1$Measure <- factor(performance_df1$Measure,
                                    levels = unique(performance_df1$Measure))
  plot1 <- ggplot(performance_df1, aes(x = Measure, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label=sprintf("%.3f", Value)), vjust=1.6, color="white", size=3.5) +
    labs(title = "Performance Measures of Class 1", x="Measure", y="Value")
  
  print(plot1) # Display the plot
  
  # Create a data frame with performance measures of Class 2
  performance_df2 <- data.frame(Measure = c("TP rate", "FP rate", "Precision",
                                            "Recall", "F1-Score", "MCC"),
                                Value = c(cm2$byClass["Sensitivity"],
                                          cm2$byClass["Specificity"],
                                          cm2$byClass["Precision"],
                                          cm2$byClass["Recall"],
                                          cm2$byClass["F1"],
                                          MCC2))
  
  # Create a bar plot
  performance_df2$Measure <- factor(performance_df2$Measure,
                                    levels = unique(performance_df2$Measure))
  plot2 <- ggplot(performance_df2, aes(x = Measure, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label=sprintf("%.3f", Value)), vjust=1.6, color="white", size=3.5) +
    labs(title = "Performance Measures of Class 2", x="Measure", y="Value")
  
  print(plot2) # Display the plot
  
  # Calculate the ROC curve
  perf <- performance(pred_perf,"tpr","fpr")
  plot(perf,colorize=TRUE)
  
  # print the results
  print(list(table1=cm1$table, overall1=cm1$overall, byClass1=cm1$byClass,
             table2=cm2$table, overall2=cm2$overall, byClass2=cm2$byClass,
             MCC1=MCC1,MCC2=MCC2,auc=auc))
}

result3 <- model_3(train, test)


## Model 4: Gradient Boosting
tuneGrid <- expand.grid(.interaction.depth = c(1, 5, 10), .n.trees = seq(100, 500, by = 50), .shrinkage = c(0.01, 0.1), .n.minobsinnode = 20) 

model_4 <- function(train, test) {
  set.seed(31)
  gbm_model <<- train(class ~ ., data = train, method = "gbm", trControl = train_control, tuneGrid = tuneGrid, verbose = FALSE)  # build the model
  predictions <- predict(gbm_model, test)
  predictions <- factor(predictions, levels = levels(test$class)) # apply the factor function
  pred_perf <- prediction(as.numeric(predictions), labels = as.numeric(test$class)) # convert to numeric
  auc=as.numeric(performance(pred_perf, measure = "auc")@y.values) # auc
  cm1 <- confusionMatrix(predictions, test$class,positive="1") # confusion matrix of class 1
  tb1=cm1$table
  TP1<-cm1$table[1,1]
  TN1<-cm1$table[2,2]
  FP1<-cm1$table[1,2]
  FN1<-cm1$table[2,1]
  MCC1 <- ((TP1*TN1)-(FP1*FN1))/((TP1+FP1)^0.5*(TP1+FN1)^0.5*(TN1+FP1)^0.5*(TN1+FN1)^0.5) # MCC of class 1's cm
  cm2 <- confusionMatrix(predictions, test$class,positive="2") # confusion matrix of class 2
  TP2<-cm2$table[1,1]
  TN2<-cm2$table[2,2]
  FP2<-cm2$table[1,2]
  FN2<-cm2$table[2,1]
  MCC2 <- ((TP2*TN2)-(FP2*FN2))/((TP2+FP2)^0.5*(TP2+FN2)^0.5*(TN2+FP2)^0.5*(TN2+FN2)^0.5)
  
  # Create a data frame with performance measures of Class 1
  performance_df1 <- data.frame(Measure = c("TP rate", "FP rate", "Precision",
                                            "Recall", "F1-Score", "MCC"),
                                Value = c(cm1$byClass["Sensitivity"],
                                          cm1$byClass["Specificity"],
                                          cm1$byClass["Precision"],
                                          cm1$byClass["Recall"],
                                          cm1$byClass["F1"],
                                          MCC1))
  
  # Create a bar plot
  performance_df1$Measure <- factor(performance_df1$Measure,
                                    levels = unique(performance_df1$Measure))
  plot1 <- ggplot(performance_df1, aes(x = Measure, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label=sprintf("%.3f", Value)), vjust=1.6, color="white", size=3.5) +
    labs(title = "Performance Measures of Class 1", x="Measure", y="Value")
  
  print(plot1) # Display the plot
  
  # Create a data frame with performance measures of Class 2
  performance_df2 <- data.frame(Measure = c("TP rate", "FP rate", "Precision",
                                            "Recall", "F1-Score", "MCC"),
                                Value = c(cm2$byClass["Sensitivity"],
                                          cm2$byClass["Specificity"],
                                          cm2$byClass["Precision"],
                                          cm2$byClass["Recall"],
                                          cm2$byClass["F1"],
                                          MCC2))
  
  # Create a bar plot
  performance_df2$Measure <- factor(performance_df2$Measure,
                                    levels = unique(performance_df2$Measure))
  plot2 <- ggplot(performance_df2, aes(x = Measure, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label=sprintf("%.3f", Value)), vjust=1.6, color="white", size=3.5) +
    labs(title = "Performance Measures of Class 2", x="Measure", y="Value")
  
  print(plot2) # Display the plot
  
  # Calculate the ROC curve
  perf <- performance(pred_perf,"tpr","fpr")
  plot(perf,colorize=TRUE)
  
  # print the results
  print(list(table1=cm1$table, overall1=cm1$overall, byClass1=cm1$byClass,
             table2=cm2$table, overall2=cm2$overall, byClass2=cm2$byClass,
             MCC1=MCC1,MCC2=MCC2,auc=auc))
}

result4 <- model_4(train, test)


## Model 5: Logistic Regression
tuneGrid <- expand.grid(alpha = 0:1, lambda = c(0.01, 0.1, 1, 10, 100))

model_5 <- function(train, test) {
  set.seed(31)
  log_model <<- train(class ~ ., data = train, method = "glmnet", trControl = train_control, tuneGrid = tuneGrid) # build the model
  predictions <- predict(log_model, test)
  predictions <- factor(predictions, levels = levels(test$class)) # apply the factor function
  pred_perf <- prediction(as.numeric(predictions), labels = as.numeric(test$class)) # convert to numeric
  auc=as.numeric(performance(pred_perf, measure = "auc")@y.values) # auc
  cm1 <- confusionMatrix(predictions, test$class,positive="1") # confusion matrix of class 1
  tb1=cm1$table
  TP1<-cm1$table[1,1]
  TN1<-cm1$table[2,2]
  FP1<-cm1$table[1,2]
  FN1<-cm1$table[2,1]
  MCC1 <- ((TP1*TN1)-(FP1*FN1))/((TP1+FP1)^0.5*(TP1+FN1)^0.5*(TN1+FP1)^0.5*(TN1+FN1)^0.5) # MCC of class 1's cm
  cm2 <- confusionMatrix(predictions, test$class,positive="2") # confusion matrix of class 2
  TP2<-cm2$table[1,1]
  TN2<-cm2$table[2,2]
  FP2<-cm2$table[1,2]
  FN2<-cm2$table[2,1]
  MCC2 <- ((TP2*TN2)-(FP2*FN2))/((TP2+FP2)^0.5*(TP2+FN2)^0.5*(TN2+FP2)^0.5*(TN2+FN2)^0.5)
  
  # Create a data frame with performance measures of Class 1
  performance_df1 <- data.frame(Measure = c("TP rate", "FP rate", "Precision",
                                            "Recall", "F1-Score", "MCC"),
                                Value = c(cm1$byClass["Sensitivity"],
                                          cm1$byClass["Specificity"],
                                          cm1$byClass["Precision"],
                                          cm1$byClass["Recall"],
                                          cm1$byClass["F1"],
                                          MCC1))
  
  # Create a bar plot
  performance_df1$Measure <- factor(performance_df1$Measure,
                                    levels = unique(performance_df1$Measure))
  plot1 <- ggplot(performance_df1, aes(x = Measure, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label=sprintf("%.3f", Value)), vjust=1.6, color="white", size=3.5) +
    labs(title = "Performance Measures of Class 1", x="Measure", y="Value")
  
  print(plot1) # Display the plot
  
  # Create a data frame with performance measures of Class 2
  performance_df2 <- data.frame(Measure = c("TP rate", "FP rate", "Precision",
                                            "Recall", "F1-Score", "MCC"),
                                Value = c(cm2$byClass["Sensitivity"],
                                          cm2$byClass["Specificity"],
                                          cm2$byClass["Precision"],
                                          cm2$byClass["Recall"],
                                          cm2$byClass["F1"],
                                          MCC2))
  
  # Create a bar plot
  performance_df2$Measure <- factor(performance_df2$Measure,
                                    levels = unique(performance_df2$Measure))
  plot2 <- ggplot(performance_df2, aes(x = Measure, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label=sprintf("%.3f", Value)), vjust=1.6, color="white", size=3.5) +
    labs(title = "Performance Measures of Class 2", x="Measure", y="Value")
  
  print(plot2) # Display the plot
  
  # Calculate the ROC curve
  perf <- performance(pred_perf,"tpr","fpr")
  plot(perf,colorize=TRUE)
  
  # print the results
  print(list(table1=cm1$table, overall1=cm1$overall, byClass1=cm1$byClass,
             table2=cm2$table, overall2=cm2$overall, byClass2=cm2$byClass,
             MCC1=MCC1,MCC2=MCC2,auc=auc))
}

result5 <- model_5(train, test)


## Model 6: Naive Bayes
tuneGrid <- expand.grid(.laplace = seq(0, 1, by = 0.1), .usekernel = c(FALSE, TRUE), .adjust = seq(1, 1.5, by = 0.1))

model_6 <- function(train, test) {
  set.seed(31)
  nb_model <<- train(class ~ ., data = train, method = "naive_bayes", trControl = train_control, tuneGrid = tuneGrid) # build the model
  predictions <- predict(nb_model, test)
  predictions <- factor(predictions, levels = levels(test$class)) # apply the factor function
  pred_perf <- prediction(as.numeric(predictions), labels = as.numeric(test$class)) # convert to numeric
  auc=as.numeric(performance(pred_perf, measure = "auc")@y.values) # auc
  cm1 <- confusionMatrix(predictions, test$class,positive="1") # confusion matrix of class 1
  tb1=cm1$table
  TP1<-cm1$table[1,1]
  TN1<-cm1$table[2,2]
  FP1<-cm1$table[1,2]
  FN1<-cm1$table[2,1]
  MCC1 <- ((TP1*TN1)-(FP1*FN1))/((TP1+FP1)^0.5*(TP1+FN1)^0.5*(TN1+FP1)^0.5*(TN1+FN1)^0.5) # MCC of class 1's cm
  cm2 <- confusionMatrix(predictions, test$class,positive="2") # confusion matrix of class 2
  TP2<-cm2$table[1,1]
  TN2<-cm2$table[2,2]
  FP2<-cm2$table[1,2]
  FN2<-cm2$table[2,1]
  MCC2 <- ((TP2*TN2)-(FP2*FN2))/((TP2+FP2)^0.5*(TP2+FN2)^0.5*(TN2+FP2)^0.5*(TN2+FN2)^0.5)
  
  # Create a data frame with performance measures of Class 1
  performance_df1 <- data.frame(Measure = c("TP rate", "FP rate", "Precision",
                                            "Recall", "F1-Score", "MCC"),
                                Value = c(cm1$byClass["Sensitivity"],
                                          cm1$byClass["Specificity"],
                                          cm1$byClass["Precision"],
                                          cm1$byClass["Recall"],
                                          cm1$byClass["F1"],
                                          MCC1))
  
  # Create a bar plot
  performance_df1$Measure <- factor(performance_df1$Measure,
                                    levels = unique(performance_df1$Measure))
  plot1 <- ggplot(performance_df1, aes(x = Measure, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label=sprintf("%.3f", Value)), vjust=1.6, color="white", size=3.5) +
    labs(title = "Performance Measures of Class 1", x="Measure", y="Value")
  
  print(plot1) # Display the plot
  
  # Create a data frame with performance measures of Class 2
  performance_df2 <- data.frame(Measure = c("TP rate", "FP rate", "Precision",
                                            "Recall", "F1-Score", "MCC"),
                                Value = c(cm2$byClass["Sensitivity"],
                                          cm2$byClass["Specificity"],
                                          cm2$byClass["Precision"],
                                          cm2$byClass["Recall"],
                                          cm2$byClass["F1"],
                                          MCC2))
  
  # Create a bar plot
  performance_df2$Measure <- factor(performance_df2$Measure,
                                    levels = unique(performance_df2$Measure))
  plot2 <- ggplot(performance_df2, aes(x = Measure, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label=sprintf("%.3f", Value)), vjust=1.6, color="white", size=3.5) +
    labs(title = "Performance Measures of Class 2", x="Measure", y="Value")
  
  print(plot2) # Display the plot
  
  # Calculate the ROC curve
  perf <- performance(pred_perf,"tpr","fpr")
  plot(perf,colorize=TRUE)
  
  # print the results
  print(list(table1=cm1$table, overall1=cm1$overall, byClass1=cm1$byClass,
             table2=cm2$table, overall2=cm2$overall, byClass2=cm2$byClass,
             MCC1=MCC1,MCC2=MCC2,auc=auc))
}

result6 <- model_6(train, test)

end_time <- Sys.time() # end time
end_time - start_time # time difference

# Collect resamples: This function checks that the models are comparable 
# and that they used the same training scheme (trainControl configuration). 
# This object contains the evaluation metrics for each fold and each repeat 
# for each algorithm to be evaluated.
results <- resamples(list(kNN=knn_model, SVM=svm_model, RF=rf_model, 
                          GB=gbm_model, LR=log_model, NB=nb_model))
summary(results)

# box and whisker plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

# dot plots of accuracy
dotplot(results, scales=scales)
