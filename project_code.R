### Authors: Hongmin Huang, Guohao Shen ###

# libraries
library(modeest)
library(caret)
library(FSelector)
library(RWeka)
library(data.table)
library(Boruta)
library(ggcorrplot)
library(GGally)

# load the dataset
df <- read.csv("project_data.csv", na.strings = "?")
dim(df)

# get the number of missing values
sum(is.na(df))

### data preprocessing

## data cleaning

# identify columns with missing values
na_cols <- colnames(df)[colSums(is.na(df)) > 0]

# replace missing values with column modes because all attributes are nominal
for (col in na_cols) {
  df[[col]][is.na(df[[col]])] <- mfv(df[[col]], na_rm = TRUE)
}

# check the number of missing values is zero
sum(is.na(df))

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

# cfs
subset <- cfs(class ~., df)
df.cfs <- as.simple.formula(subset, "class")
df.cfs
att1 <- c("vs007","vs030","vs064a","vs066","vs068","vs131","SchCultureRecode")

# info gain
df2 <- copy(df)
df2 <- as.data.frame(unclass(df2), stringsAsFactors = TRUE)
df2$class <- factor(df2$class)
df2.infogain <- InfoGainAttributeEval(class ~., data = df2)
sorted.features <- sort(df2.infogain, decreasing = TRUE)
sorted.features[1:10]
att2 <- c("vs129","SchCultureRecode","vs066","vs068","vr16","vs130","vs046","vs061","vs064a","vs060")

# Boruta
df.boruta <- Boruta(class ~., data = df)
df.boruta
att3 <- getSelectedAttributes(df.boruta, withTentative=FALSE)
att3

# find common elements
attShared <- intersect(intersect(att1,att2), att3)
attShared
attShared <- c(attShared, "class")

# select important attributes
# create attCopy for temporary use (no need to run feature selection)
attCopy <- c("vs064a", "vs066", "vs068", "SchCultureRecode", "class")
df <- subset(df, select = attCopy)
head(df)

# save the preprocessed data
#write.csv(df, "preprocessed_data.csv", row.names = FALSE)

# correlation plot
sub_df <- subset(df, select = c("vs064a", "vs066", "vs068", "SchCultureRecode"))
cor(sub_df)
ggpairs(sub_df)
ggcorrplot(cor(sub_df), method = "square", lab = TRUE)

df$class <- factor(df$class)

split <- initial_split(df, prop = 0.7, strata = class)
train <- training(split)
test <- testing(split)
#10-fold cross-validation
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)

model_1 <- function(train,test) {
  set.seed(31)
  j48ml <- J48(class~., data=train)
  predictions <- predict(j48ml, test)
  cm <- confusionMatrix(predictions, test$class)
  return(list(table=cm$table, overall=cm$overall, byClass=cm$byClass))
}

result <- model_1(train, test)

model_2 <- function(train,test) {
  svmml <- svm(class ~ ., data = train)
  predictions <- predict(svmml, test)
  cm <- confusionMatrix(predictions, test$class)
  return(list(table=cm$table, overall=cm$overall, byClass=cm$byClass))
}

result <- model_2(train, test)

