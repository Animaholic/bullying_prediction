### Authors: Hongmin Huang, Guohao Shen ###

## load the dataset
df <- read.csv("project_data.csv", na.strings = "?")

#replance the ? sign to na
df[df=="?"]<-NA

# get the number of missing values
sum(is.na(df))

## data preprocessing

library(modeest)

# identify columns with missing values
na_cols <- colnames(df)[colSums(is.na(df)) > 0]

# replace missing values with column modes
# becasue there are nominal variables, so we use modes rather than mean
for (col in na_cols) {
  df[[col]][is.na(df[[col]])] <- mfv(df[[col]], na_rm = TRUE)
}

# check the number of missing values is zero
sum(is.na(df))

#write.csv(df, "preprocessed_data.csv", row.names = FALSE)


