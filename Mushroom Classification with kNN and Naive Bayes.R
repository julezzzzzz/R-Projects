# Opening required libraries
# Library for recoding factors
library(plyr)
# Library to look at correlations between the variables
library(corrgram)
# This package gives tables for model classification 
library(gmodels)
# Library for Naive Bayes 
library(e1071) 
# Library for more ML models/ functions  
library(caret)


# Reading in the data frame 
theUrl <- "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
mushrooms <- read.table(file = theUrl, header = FALSE, sep = ",")

# Creating column names 
col_names = c('class', 'cap_shape', 'cap_surface','cap_color', 'bruises',
            'odor', 'gill_attachment', 'gill_spacing', 'gill_size',
            'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
            'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring',
            'veil_type', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color',
            'population', 'habitat')

# Renaming columns
colnames(mushrooms)<-col_names

# Looking at the top 5 rows to make sure column naming worked 
head(mushrooms)

# Looking at structure of data, 23 variables with 8,124 observations 
str(mushrooms)

# Summarizing Data
summary(mushrooms)
# veil_type only has one distinct value in the column. I will drop this variable as including it will over cause over-fitting.
## Many of the variables have underrepresented classes in the DF. This should be considered in the final model, as 
##  those variables might be very strong indicators of poisonous, but since they're underrepresented, they might get ignored, 
##   watch out for Type 2 Errors in confusion matrix. 
## stalk_root has 2,480 missing values, as noted in the documentation, missing are marked with a question mark. 

# Check for missing values 
any(is.na(mushrooms))
# None, but the documentation said that missing values are marked with a question mark


#Calculating the percentage of total for the variable we're trying to predict 
round(prop.table(table(mushrooms$class))*100,digit=1) 
# Results:
# 51.8% Edible, 48.2% Poisonous 


# Since veil_type only has one value, I'll drop it. 
drops <- c("veil_type")
mushrooms <- mushrooms[ , !(names(mushrooms) %in% drops)]
# For now, I'm going to leave in the columns with missings (containing a questions mark)

# Converting entire DF into numbers
mushroom_numbers <- data.frame(lapply(mushrooms, function(x) as.numeric(as.factor(x))))


n.point <- nrow(mushroom_numbers) # Number of rows in mushroom DF 
sampling.rate <- 0.75 # Percentage of data I am sampling 


training <- sample(1:n.point, sampling.rate*n.point, replace = FALSE) # Creating basis for training set
testing <- setdiff(1:n.point,training) # Creating basis for testing set 


# Subsetting data per the above specifications 
mushroom_train <- subset(mushroom_numbers[training,])
mushroom_test <- subset(mushroom_numbers[testing,])

# These are the target columns 
spc_train <-  mushroom_train[,1]
spc_test <-  mushroom_test[,1]

# Factorizing the target columns 
spc_train <- revalue(as.factor(spc_train), c("1"="Edible", "2"="Poisonous"))
spc_test <- revalue(as.factor(spc_test), c("1"="Edible", "2"="Poisonous"))

# Drop the target columns from the training and testing dataset, as we don't want to include in our final model
mushroom_train <- mushroom_train[,-1]
mushroom_test <- mushroom_test[,-1]

# Check SUmmary Statistics
summary(mushroom_train)
summary(spc_train)

# Check distribution of outcome variable for our training set- 52.2% E and 47.8% poisonous- close to our overall DF
round(prop.table(table(spc_train))*100,digit=1) 


# Create correlation matrix 
corrgram(mushroom_numbers , lower.panel=panel.shade,
         upper.panel=panel.pie, text.panel=panel.txt,
         main="Mushroom Data Correlations")

# Gill color and gill size are highly positively correlated, gill color is also correlated with target variable class 
# Bruises is highly positively correlated with target variable class
# Gill attachment is highly negatively correlated with veil color 
# Gill size is highly negatively correlated with target variable class 
# Ring type is negatively correlated with gill color and bruises 


## Now that the data is ready to model, I'll start with the kNN Algorithm. 

set.seed(1) # This will allow me to reproduce the results


mushroom_test_pred <- knn(train = mushroom_train, test = mushroom_test,
                      cl = spc_train, k = 5) # starting with k=5 because it's the approx square root of number of variables


# Misclassification Rate 
misclass.rate <- mean(spc_test != mushroom_test_pred)


# Write a For loop to minimize the Misclassification Rate (n=100)
predict_species <- NULL 
error.rate <- NULL
for (i in 1:100){
  set.seed(1)
  predict_species <- knn(mushroom_train,mushroom_test, spc_train,k=i)
  error.rate[i] <- mean(spc_test != mushroom_test_pred)
}

# Show error rates 
k<- 1:5
error.df<- data.frame(k, error.rate)
error.df

# The error rates are all the same, which makes me think we're overfitting the model, and need to consider the # of variables. 
# I'll use the rule of thumb sqrt(# of variables) ~5 
predict_species_knn <- knn(mushroom_train, mushroom_test, spc_train, k=5)

## evaluate model for misclassification rate 
CrossTable(spc_test, predict_species, prop.chisq = FALSE)
# We incorrectly identified 93 mushrooms as edible when they are poisonous, which is the type 2 error I was worried about
# we predict edible as edible correctly 98% of the time, and poisonous as poisonous 90% of the time 

# kNN Optimization

# As I saw above, it appears there is some multicollinearity in the model, I'll calculate the relative 
# importance of each feature. 


# calculate correlation matrix
correlationMatrix <- cor(mushroom_numbers)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

# Dropping highly correlated variables 
mushroom_numbersclean <- mushroom_numbers[,-c(highlyCorrelated)] 

# Subsetting data per the above specifications 
mushroom_train <- subset(mushroom_numbersclean[training,])
mushroom_test <- subset(mushroom_numbersclean[testing,])

# Run the new model
mushroom_clean_knn <- knn(train = mushroom_train, test = mushroom_test,
                          cl = spc_train, k = 5)


# Misclassification Rate 
misclass.rate <- mean(spc_test != mushroom_clean_knn)


# Write a For loop to minimize the Misclassification Rate (n=100)
predict_species <- NULL 
error.rate <- NULL
for (i in 1:100){
  set.seed(1)
  predict_species <- knn(mushroom_train,mushroom_test, spc_train,k=i)
  error.rate[i] <- mean(spc_test != mushroom_clean_knn)
}

# Show error rates 
k<- 1:5
error.df<- data.frame(k, error.rate)
error.df

# seems like the k=5 value is fine, as the values are constant 
predict_species <- knn(mushroom_train, mushroom_test, spc_train, k=5)


## evaluate model for misclassification rate 
CrossTable(spc_test, predict_species, prop.chisq = FALSE)
# removing variables greatly improved the model, we minimized the type 2 error rate to zero
# we predict edible = edible 99% of the time, and only 9 times we predicted edible when it was poisonous
# 10x better than the original model. 
# Dropping the excess variables greatly improved model performance. 





## Naive Bayes Model ## 

# I'll start with the original data frame, not dropping the excess columns

mushroom_train <- subset(mushrooms[training,])
mushroom_test <- subset(mushrooms[testing,])

# Creating a model 
mushroom_classifier <- naiveBayes(class~.,data=mushroom_train)

# use the created classifer to predict outcome. Need to transform the outcome variable to a factor because 
# Naive Bayes calculates the probability of an event occuring or not, so it has to be binary. 
mushroom_test_predict <- predict(mushroom_classifier, mushroom_test)

# see how the model works

CrossTable(mushroom_test_predict, spc_test, prop.chisq = TRUE, prop.t = TRUE, prop.r = TRUE)
# We predict edible = edible 90% of the time, and .7% of the time we classify poisonous as edible, 
## Much smaller type 2 error than the kNN inital model
## More incorrect predictions for edible > poisonous (109 instances where actually edible, classified as poisonous)

## Naive Bayes Optimization

# Trying the same model with the same dropped columns as before

# Need to keep the first column though
highly_correlated <- c(19,20,9,10,8,7)
mushroom_clean <- mushrooms[,-c(highly_correlated)] 

mushroom_train <- subset(mushroom_clean[training,])
mushroom_test <- subset(mushroom_clean[testing,])

clean_mushroom_classifier <- naiveBayes(class~.,data=mushroom_train)

# use the created classifer to predict outcome. Need to transform the outcome variable to a factor because 
# Naive Bayes calculates the probability of an event occuring or not, so it has to be binary. 
mushroom_test_predict <- predict(mushroom_classifier, mushroom_test)

# see how the model works

CrossTable(mushroom_test_predict, spc_test, prop.chisq = TRUE, prop.t = TRUE, prop.r = TRUE)
## Improvement in all categories when dropping the excess columns
## we only miscategorize poisonous mushrooms as edible 8 times (previously 9)
# also improve actually edible categorized as poisonous (65 down from 109)






# Conclusions 
## kNN provides strong results, especially when dropping excessive variables,
#  but I worry that generalizing to more data might prove that the model overfits. 

# Naive Bayes inital model had much lower type 2 error rate than kNN, although the overall classification rate was worse.
# Removing excess columns from the models greatly improved performance. 
# There are still a lot of variables, and it should be noted that in the kNN model, since I converted the DF to integers,
# I'm inherently weighting observations. A good step might be to normalize this effect, but I don't know if that's 
# a fair assumption to make, as I'd be turning categorical variables into continous. Definitely worth investigating.


## Additionally, some dimensionality reduction techniques might be worthwhile to explore. There are a lot of variables, and
# I still think there is a degree of overfitting. 


