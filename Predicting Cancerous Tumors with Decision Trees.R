#*** Coding Challenge # 2 ***


#(10pts) Build a basic Decision Tree model to predict if a tumor is benign or malignant.
# (5pts)Plot our Decision Tree. What is the error rate for your model? Which attributes are used for the splits?
# (10pts) How can you improve your model? 
# (10pts) Build a basic Random Forest model to predict if a tumor is benign or malignant.
# (10pts) How can you improve your model? 
# (5pts) Compare the Decision Tree model and Random Forest model. Which model is better?
# Submit your code, scripts and above deliverables on Canvas. You can create one Word/PDF file with all the details and submit as well.


library(caret)
library(gmodels)
library(MASS)
library(C50)
library(randomForest)
library(gmodels)
library(psych)

data(biopsy)

str(biopsy)
# 699 observations, 11 variables, 1 outcome variable benign / malignant 

boxplot(biopsy$V1, main="Boxplot of clump thickness",
        ylab="Thickness")
# Slightly negatively skewed
boxplot(biopsy$V2, main="Boxplot of uniformity of cell size",
        ylab="Uniformity")
# Negatively skewed with some larger outliers 
boxplot(biopsy$V3, main="Boxplot of uniformity of cell shape",
        ylab="Uniformity")
# Negatively skewed with some larger outliers 
boxplot(biopsy$V4, main="Boxplot of marginal adhesion",
        ylab="Adhesion")
# has outliers
boxplot(biopsy$V5, main="Boxplot of epithelial cell size",
        ylab="size")
# has outliers 
boxplot(biopsy$V6, main="Boxplot of bare nuclei",
        ylab="size")
# negatively skewed
boxplot(biopsy$V7, main="Boxplot of bland chromatin",
        ylab="size")
# has outliers but almost normally distributed
boxplot(biopsy$V8, main="Boxplot of normal nucleoli",
        ylab="size")
# has outliers 
boxplot(biopsy$V9, main="Boxplot of epithelial cell size",
        ylab="size")
# has a lot of outliers 

# many of these variables should be candidates for logging 

# Check distributions


pairs.panels(biopsy[c("V1","V2","V3","V4","V5","V6","V7","V8","V9","class")])
# V2 and V3 look highly correlated, might consider dropping from the final model

summary(biopsy)
# There are 16 missing variables in V6 

# drop the ID variable and V6, because I don't want to make the dataset smaller, and I can always add it bad
drops <- c("ID","V6")
biopsy <- biopsy[ , !(names(biopsy) %in% drops)]

# Create training and testing groups 

set.seed(123)
train_sample <- sample(699, 525)

str(train_sample)

# split the data frames
bioposy_train <- biopsy[train_sample, ]
biopsy_test  <- biopsy[-train_sample, ]

# check the proportion of class variable
prop.table(table(biopsy$class)) # 65% to 34%
prop.table(table(bioposy_train$class)) # 66% to 33%
prop.table(table(biopsy_test$class)) # 62% to 37%

# Build decision tree 
biopsy_model <- C5.0(bioposy_train[-9], bioposy_train$class)

summary(biopsy_model)
# The decision tree is composed of 4 splits 
# Training model incorrectly predicted 17 cases (3.2%)
# The top attributes are  Uniformity of Cell Size and Epitheial of Cell Size

biopsy_pred <- predict(biopsy_model, biopsy_test)


CrossTable(biopsy_test$class, biopsy_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual cancer', 'predicted cancer'))
# We predict malignant as benign 4 times (2.3%) and benign as malignant 4 times (2.3%). So a total of a 4.6% error rate 
# This is slightly higher than the error on the training set. 
# This is a dataset where false positives are deadly, so reducing that error is important. 
# To help with this, I will do some boosting trials on the dataset in order to get a diff

biopsy_boost10 <- C5.0(bioposy_train[-10], bioposy_train$class,
                       trials = 10,rules = TRUE)

summary(biopsy_boost10)

biopsy_boost_pred10 <- predict(biopsy_boost10, biopsy_test)
CrossTable(biopsy_test$class, biopsy_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual cancer', 'predicted cancer'))

# The rules-based approach led to a final model predicting accurately 100% of the time. 

# Build Random Forest Model 

random_forest <- randomForest(bioposy_train[-9], bioposy_train$class)
summary(random_forest)  

rd_predict <- predict(random_forest, biopsy_test)

CrossTable(biopsy_test$class, rd_predict,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual cancer', 'predicted cancer'))

#This model already performs better than the basic decision tree. 
#We see much lower actual malignant predicted benign cases (only 1)
# and the same number of benign predicted malignant (4)

# To improve the model, I'll pass additional parameters into the classifier 
# correlation bias is an experimental feature that corrects for regression within the model.
random_forest <- randomForest(bioposy_train[-9], bioposy_train$class, corr.bias=FALSE)
summary(random_forest)  

rd_predict <- predict(random_forest, biopsy_test)

CrossTable(biopsy_test$class, rd_predict,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual cancer', 'predicted cancer'))
# That actually worsened the model - we now have 2 cases of actual malignant predicted benign
# and 4 cases of benign actually predicted malignant 

# This leads me to believe that the random forest model is better than the decision tree. 
# This would make sense as a random forest model is a compliation of decision trees. 