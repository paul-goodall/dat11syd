## Author: Andrew Szwec (AIM)
## Date: 2016-04-01 
## Title: Cognitive 101 R Script
## Description: Build a random forest predicting likelihood of survival

## titanic data
require(reshape2)
require(caret)

# Read in the Titanic Data
setwd("~/Cognitive 101")
df <- read.csv('titanic3.csv')

# Change the survived status from the word "yes" into the number one as a factor
df$survivedFlag <- as.factor(ifelse(df$Survived == 'Yes', 1, 0))
df$count <- 1

# Pivot each column to make dummy variables
tickets <- dcast(data=df, PassengerId ~ Ticket, fun.aggregate = sum, value.var = "count")
names(tickets) <- ifelse(names(tickets)!='PassengerId', paste("Ticket",names(tickets)), names(tickets))

embarked <- dcast(data=df, PassengerId ~ Embarked, fun.aggregate = sum, value.var = "count")
names(embarked) <- ifelse(names(embarked)!='PassengerId', paste("Embarked",names(embarked)), names(embarked))

cabin <- dcast(data=df, PassengerId ~ Cabin, fun.aggregate = sum, value.var = "count")
names(cabin) <- ifelse(names(cabin)!='PassengerId', paste("Cabin",names(cabin)), names(cabin))

# Join the dummy variables back onto the main table
dd <- merge(df, tickets, by='PassengerId', all.x=TRUE)
dd <- merge(dd, embarked, by='PassengerId', all.x=TRUE)
dd <- merge(dd, cabin, by='PassengerId', all.x=TRUE)

# make a variable containing  average age
avgAge <- mean(dd$Age, na.rm = TRUE)

# impute missing age values using the mean age value
dd$Age <- ifelse(is.na(dd$Age), avgAge ,dd$Age)

# Turn Pclass, Sez and Survived Flag into factors so the algorith can make the prediction
dd$Pclass <- as.factor(dd$Pclass)
dd$Sex <- as.factor(dd$Sex)
dd$survivedFlag <- as.factor(dd$survivedFlag)

# Remove irrelevant variables from the dataset
de <- subset(dd, select=-c(PassengerId, Survived, Name, Ticket, Cabin, Embarked, count))

# how many NAs?
na_count <-sapply(de, function(y) sum(length(which(is.na(y)))))
na_count[na_count>0]

# Split the data into a model training and test set used to measure the performance of the algorithm
set.seed(975)
inTrain     = createDataPartition(de$survivedFlag, p = 0.7)[[1]]
training    = de[ inTrain,]      # 70% of records
testing     = de[-inTrain,]      # 30% of reocrds


##########################################################
## Build parallel model on Windows
##########################################################
library(randomForest)
library(foreach)
library(doSNOW)

cores <- 2
cl <- makeCluster(cores, type = "SOCK",outfile="")
registerDoSNOW(cl)


total.tree <- 2000
num.chunk <- cores
avg.tree <- ceiling(total.tree/num.chunk)

# Build a random forest model predicting the outcome that passengers survived
# model takes 1.20min to train
time <- system.time({
  rf_fit <- foreach(ntree = rep(avg.tree, num.chunk), .combine = combine, 
                    .packages = c("randomForest")) %dopar% {
                      randomForest(subset(training, select=-c(survivedFlag)), training$survivedFlag, ntree = ntree,importance=TRUE)
                    }
})

stopCluster(cl)

print("Time to build model was ")
time
# Time with 2 cores 
# user  system elapsed 
# 14.36    5.70  709.97 

save(rf_fit, file="rf_titanic.RData")
#load(file="rf_fit.RData")

# Use the model to make a prediction about whether passengers in the test set survived
pred <- predict(rf_fit, newdata=testing)

# compare the prediction of survival with the observation
results <- data.frame(observations=testing$survivedFlag, predictions=pred)
results$observations <- as.numeric(results$observations)  
results$predictions <- as.numeric(results$predictions)

# Print a confusion matrix to view the results
require(e1701)
confusionMatrix(results$predictions, results$observations)

# Look at the contribution of variables to the model
a <- data.frame(rf_fit$importance)
a <- a[order(a$X1,decreasing = TRUE),]
aa <- head(a, n=10L)
aa$var.name <- as.factor(row.names(aa))

# Plot on graph
ggplot(data=aa, aes(x=var.name, y=X1, fill=var.name)) +
  geom_bar(colour="black", stat="identity") +
  guides(fill=FALSE)  + xlim(rev(levels(aa$var.name)))


# top features predicting survival were:
#              X0           X1                  MeanDecreaseAccuracy
# Sex          9.411088e-02 0.1320663667         0.1084078070
# Fare         2.005345e-02 0.0347258948         0.0256671730
# Pclass       1.146529e-02 0.0314758024         0.0190736547
# Var.2        2.452483e-02 0.0263470997         0.0251675802
# Age          1.880284e-02 0.0066780367         0.0141398798
# 1601        -4.149712e-04 0.0036925307         0.0011526793
# Southampton  2.614621e-03 0.0022299192         0.0024586047
# Cherbourgh   2.156986e-03 0.0013554725         0.0018576520
# Queenstown  -1.659875e-04 0.0012982322         0.0003888703
# E25          1.328157e-04 0.0008473511         0.0004046531



