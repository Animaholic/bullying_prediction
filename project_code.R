### Authors: Hongmin Huang, Guohao Shen ###

# libraries
library(modeest)
library(caret)
library(FSelector)
library(RWeka)
library(data.table)
library(Boruta)

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

# info gain
df2 <- copy(df)
df2 <- as.data.frame(unclass(df2), stringsAsFactors = TRUE)
df2$class <- factor(df2$class)
df2.infogain <- InfoGainAttributeEval(class ~., data = df2)
sorted.features <- sort(df2.infogain, decreasing = TRUE)
sorted.features[1:10]

# Boruta
df.boruta <- Boruta(class ~., data = df)
df.boruta

# select important attributes
# df <- subset(df, select = c(class))

#write.csv(df, "preprocessed_data.csv", row.names = FALSE)


