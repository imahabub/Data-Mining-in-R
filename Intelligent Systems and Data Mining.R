# Load necessary libraries
library(tidyverse)   # For data manipulation
library(caret)       # For data splitting and model training
library(dplyr)       # For data manipulation

# Load the dataset (replace with actual path if using a local file)
data <- read.csv("D:/bank-additional.csv")

# View the structure of the dataset
str(data)

# Convert categorical variables to factors
data$job <- as.factor(data$job)
data$marital <- as.factor(data$marital)
data$education <- as.factor(data$education)
data$default <- as.factor(data$default)
data$housing <- as.factor(data$housing)
data$loan <- as.factor(data$loan)
data$contact <- as.factor(data$contact)
data$month <- as.factor(data$month)
data$day_of_week <- as.factor(data$day_of_week)
data$poutcome <- as.factor(data$poutcome)
data$y <- as.factor(data$y)


# Descriptive statistics for numeric variables (age and duration as examples)
summary(data$age)
summary(data$duration)

# Standard deviation
sd(data$age)
sd(data$duration)

# Descriptive statistics for the entire dataset
summary(data)


# Scatter plot of Age vs Duration
plot(data$age, data$duration,
     main = "Scatter Plot of Age vs Duration",
     xlab = "Age",
     ylab = "Duration",
     col = "blue", pch = 16)   # pch: point character shape


# Histogram of Age
hist(data$age,
     main = "Histogram of Age",
     xlab = "Age",
     col = "lightblue",
     border = "black")



# Bar plot of Job categories
job_counts <- table(data$job)
barplot(job_counts,
        main = "Bar Plot of Job Categories",
        xlab = "Job",
        ylab = "Count",
        col = "lightgreen",
        las = 2)  # las: label orientation


# Split the data into training and testing sets
set.seed(123)
splitIndex <- createDataPartition(data$y, p = 0.7, list = FALSE)
train_data <- data[splitIndex, ]
test_data <- data[-splitIndex, ]

# Hypothesis: Logistic regression and decision trees may perform well in predicting customer subscription.




# Logistic Regression Model
set.seed(123)
logistic_model <- train(y ~ ., data = train_data, method = "glm", family = "binomial",
                        trControl = trainControl(method = "cv", number = 10),
                        tuneLength = 5)

# Predict and evaluate on the test data
logistic_pred <- predict(logistic_model, newdata = test_data)
confusionMatrix(logistic_pred, test_data$y)


# Load rpart for decision tree
library(rpart)
library(rpart.plot)

# Train the decision tree model
set.seed(123)
tree_model <- rpart(y ~ ., data = train_data, method = "class")

# Plot the tree
rpart.plot(tree_model)

# Predict and evaluate on the test data
tree_pred <- predict(tree_model, newdata = test_data, type = "class")
confusionMatrix(tree_pred, test_data$y)




# Load random forest library
library(randomForest)

# Train Random Forest model
set.seed(123)
rf_model <- randomForest(y ~ ., data = train_data, ntree = 100)

# Predict and evaluate on the test data
rf_pred <- predict(rf_model, newdata = test_data)
confusionMatrix(rf_pred, test_data$y)



# Evaluate metrics for logistic regression
logistic_accuracy <- confusionMatrix(logistic_pred, test_data$y)$overall['Accuracy']
logistic_accuracy

# Evaluate metrics for decision tree
tree_accuracy <- confusionMatrix(tree_pred, test_data$y)$overall['Accuracy']
tree_accuracy

# Evaluate metrics for random forest
rf_accuracy <- confusionMatrix(rf_pred, test_data$y)$overall['Accuracy']
rf_accuracy





# You can also calculate other performance metrics like precision, recall, and F1-score:
logistic_confusion <- confusionMatrix(logistic_pred, test_data$y)
tree_confusion <- confusionMatrix(tree_pred, test_data$y)
rf_confusion <- confusionMatrix(rf_pred, test_data$y)

# Precision, Recall, F1-score for Logistic Regression
precision_logistic <- logistic_confusion$byClass['Pos Pred Value']
recall_logistic <- logistic_confusion$byClass['Sensitivity']
f1_score_logistic <- (2 * precision_logistic * recall_logistic) / (precision_logistic + recall_logistic)

# Precision, Recall, F1-score for Decision Tree
precision_tree <- tree_confusion$byClass['Pos Pred Value']
recall_tree <- tree_confusion$byClass['Sensitivity']
f1_score_tree <- (2 * precision_tree * recall_tree) / (precision_tree + recall_tree)

# Precision, Recall, F1-score for Random Forest
precision_rf <- rf_confusion$byClass['Pos Pred Value']
recall_rf <- rf_confusion$byClass['Sensitivity']
f1_score_rf <- (2 * precision_rf * recall_rf) / (precision_rf + recall_rf)

# Print out the results
cat("Logistic Regression F1-Score: ", f1_score_logistic, "\n")
cat("Decision Tree F1-Score: ", f1_score_tree, "\n")
cat("Random Forest F1-Score: ", f1_score_rf, "\n")
