
# Read in data/ required packages 
library(archdata)
library(gmodels)
library(psych)


data(Handaxes)

summary(Handaxes)

# In looking at the range of values:
# L1: 15-139 
# B: 34-123
# B2: 21-102
# T: 20-69
# T1: 7-35 
# No missing variables.
# Outliers in L, L1, otherwise the distributions of the data look relatively normal.
# I will drop 'Catalog' because it's just a name for the type 


pairs.panels(Handaxes[c("L", "L1", "B", "B1",  "B2","T","T1")])
# Many of the variables are highly correlated with each other, which is going to lead to overfitting,
# But the variables look pretty well distributed around the mean
# B is highly positively correlated to the target variable 
# B is also highly positively correlated to B2, so only one of those should proably be used in the final mode


# Transforming the data 
Handaxes$L1 <- log(Handaxes$L1)
Handaxes$B <- log(Handaxes$B)
Handaxes$B1 <- log(Handaxes$B1)
Handaxes$B2 <- log(Handaxes$B2)
Handaxes$T <- log(Handaxes$T)
Handaxes$T1 <- log(Handaxes$T1)

# Dropping the catalog column, but preserving the original DF so calling a new one 
Handaxes_DF <- Handaxes[,-1]

# Calculating the correlation coefficients  
correl <- cor(Handaxes_DF) 
# They are the same as from the pairs plot.


findCorrelation(correl,cutoff = .6) # B and B1 are highly correlated to the target variable, and
# as such I will only use one of them, but test on both to see what performs better. 

# Sampling the data 
set.seed(230)

handaxe_rand <- Handaxes_DF[order(runif(600)),]

s<- sample(600,550)

handaxe_trainX <- handaxe_rand[s,]
handaxe_trainY <- as.data.frame(handaxe_trainX$L)
handaxe_trainX <- handaxe_trainX[,-1]
handaxe_testX <- handaxe_rand[-s,]
handaxe_testY <-  as.data.frame(handaxe_testX$L)
handaxe_testX <- handaxe_testX[,-1]


# Build model using B instead of B1 
lm <- lm(handaxe_trainY$`handaxe_trainX$L` ~handaxe_trainX$L1 + handaxe_trainX$B + handaxe_trainX$B2 + handaxe_trainX$T + handaxe_trainX$T1)
summary(lm)  # all statistically significant, Adj. R2 of 66.6%





# predict the ranking using the above model 
pred <- predict(lm, newdata = handaxe_testX)

# Look at how the actuals line up to predicted
par(mfrow = c(2, 2))
plot(lm)


# Build model using B1 instead of B
lm <- lm(handaxe_trainY$`handaxe_trainX$L` ~handaxe_trainX$L1 + handaxe_trainX$B1 + handaxe_trainX$B2 + handaxe_trainX$T + handaxe_trainX$T1)
summary(lm)  # all statistically significant, Adj. R2 of 66.6%
# Slightly improved model, now Adj. R2 is 68.6% 


# predict the ranking using the above model 
pred <- predict(lm, newdata = handaxe_testX)

# Look at how the actuals line up to predicted
par(mfrow = c(2, 2))
plot(lm)


# Coefficients stay consistent across the models, which is good to see,
# One way to improve the model would be to change the sampling technique,
# we could instead do cross validation, and see how that improves the model performance. 

