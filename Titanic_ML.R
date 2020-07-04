############################################################################
# KAGGLE MACHINE LEARNING PROJECT - TITANIC: MACHINE LEARNNG FROM DISASTER #
# Date - 30th May 2019                                                     #
# Skills utilised - data cleaning, feature engineering, ML (random forest) #
############################################################################

install.packages('dplyr')
install.packages('mice')
install.packages('VIM')
install.packages('randomForest')
install.packages('splitstackshape')
install.packages('tidyverse')
install.packages('ggplot2')
install.packages('ROCR')


# Import and Load Librarys
library('dplyr')              #Data manipulation tool
library('splitstackshape')    #Data manipulation tool
library('mice')               #Multivariate Imputation by Chained Equation - dealing with missing values
library('VIM')                #K-nearest neighbour - dealing with missing values
library('tidyverse')          #Data maniplulation tool
library('ggplot2')

## Read Files
train <- read.csv(file = "train.csv")
test <- read.csv(file = "test.csv")


###########################################
# DATA MANIPULATION - Feature Engineering #
###########################################


## Family size
#################

train$FamilySize <- train$SibSp + train$Parch

## Determining spouse survival
#################################

#Step 1 - splitting out the maiden names of those listed
train_spousesurvival <- cSplit(train, "Name", sep = "(", direction = "wide")
colnames(train_spousesurvival)[colnames(train_spousesurvival)=="Name_1"] <- "Name"
colnames(train_spousesurvival)[colnames(train_spousesurvival)=="Name_2"] <- "MaidenName"
train_spousesurvival$Name_3 <- NULL
train_spousesurvival$MaidenName <- gsub(")", "", train_spousesurvival$MaidenName)

#Step 2 - creating searchable full names 
train_spousesurvival <- cSplit(train_spousesurvival, "Name", sep = ",", direction = "wide")
colnames(train_spousesurvival)[colnames(train_spousesurvival)=="Name_1"] <- "Surname"
train_spousesurvival <- cSplit(train_spousesurvival, "Name_2", sep = ".", direction = "wide")
colnames(train_spousesurvival)[colnames(train_spousesurvival)=="Name_2_1"] <- "Title"
colnames(train_spousesurvival)[colnames(train_spousesurvival)=="Name_2_2"] <- "FirstNames"
train_spousesurvival$FullName <- paste(train_spousesurvival$FirstNames, train_spousesurvival$Surname, sep = " ", collapse = NULL)
train_spousesurvival <-
  transform(
    train_spousesurvival,
    TrueName =
      ifelse( is.na(MaidenName), train_spousesurvival$FullName,
              train_spousesurvival$MaidenName ))
  

#Step 3 - creating a list of male passengers and whether their wives survived
train_maidenname <- train_spousesurvival %>% select(MaidenName, FullName, Survived, PassengerId, TrueName)
train_maidenname <- na.omit(train_maidenname)
colnames(train_maidenname)[colnames(train_maidenname)=="Survived"] <- "SpouseSurvived"
train_maidenname$MaidenName <- NULL
train_HasWifeRecord <- merge(x=train_maidenname, y=train_spousesurvival, by="FullName")
train_HasWifeRecord <- train_HasWifeRecord[train_HasWifeRecord$Title!="Mrs"]
train_HasWifeRecord$PassengerId.x <- NULL
colnames(train_HasWifeRecord)[colnames(train_HasWifeRecord)=="PassengerId.y"] <- "PassengerId"
colnames(train_HasWifeRecord)[colnames(train_HasWifeRecord)=="TrueName.y"] <- "TrueName"
colnames(train_HasWifeRecord)[colnames(train_HasWifeRecord)=="TrueName.x"] <- "SpouseName"

#Step 4 - creating a list of female passengers and whether their husbands survived
train_maidenname_2 <- train_spousesurvival %>% select(MaidenName, FullName, Survived, Sex, PassengerId, TrueName)
train_maidenname_2 <- train_maidenname_2[train_maidenname_2$Sex=="male"]
train_maidenname_2$MaidenName <- NULL
train_maidenname_2$Sex <- NULL
colnames(train_maidenname_2)[colnames(train_maidenname_2)=="Survived"] <- "SpouseSurvived"
train_HasHusbandRecord <- merge(x=train_maidenname_2, y=train_spousesurvival, by="FullName")
train_HasHusbandRecord <- train_HasHusbandRecord[train_HasHusbandRecord$Title=="Mrs"]
train_HasHusbandRecord$PassengerId.x <- NULL
colnames(train_HasHusbandRecord)[colnames(train_HasHusbandRecord)=="PassengerId.y"] <- "PassengerId"
colnames(train_HasHusbandRecord)[colnames(train_HasHusbandRecord)=="TrueName.y"] <- "TrueName"
colnames(train_HasHusbandRecord)[colnames(train_HasHusbandRecord)=="TrueName.x"] <- "SpouseName"

#Step 5 - Appending the two spouse record data frames
train_HasSpouseRecord <- rbind(train_HasHusbandRecord, train_HasWifeRecord)
train_HasSpouseRecord <- train_HasSpouseRecord %>% select(SpouseSurvived, SpouseName, TrueName)

#Step 6 - Merging Spouse Survival data back into the main dataframe.
train_spousesurvivalAdded <- merge(x=train_spousesurvival, y=train_HasSpouseRecord, by = "TrueName", all=TRUE)
train_spousesurvivalAdded$SpouseSurvived <- replace_na(train_spousesurvivalAdded$SpouseSurvived, "No Spouse")
train_condensed_spsurAdded <- subset(train_spousesurvivalAdded, select = -c(MaidenName, FullName, TrueName, Surname, FirstNames, SpouseName))



#################################################
# DATA MANIPULATION - Dealing with Missing Data #
#################################################

## Identifying columns with missing values
#############################################
colnames(train_condensed_spsurAdded)[apply(train_condensed_spsurAdded, 2, anyNA)]

#Dealing with missing values 
###############################

#Method 1: Imputation by most frequent value
# Note - this method will alter both mean and reduce the variation of the data, causing bias.

getmode <- function(values) {
  uniquevalues <- unique(values[!is.na(values)])
  uniquevalues[which.max(tabulate(match(values, uniquevalues)))]
}

trainMODE <- train_condensed_spsurAdded
trainMODE$Age[is.na(trainMODE$Age)] <- getmode(trainMODE$Age)

  
#Method 2: Imputation by mean value
# Note - this method will keep the mean of Age the same, but will cause a reduction in the variation of the data, causing bias.

trainMEAN <- train_condensed_spsurAdded
averagevalue <- mean(trainMEAN$Age[!is.na(trainMEAN$Age)])
trainMEAN$Age[is.na(trainMEAN$Age)] <- averagevalue


#Method 3: Imputation by MICE using Predictive Mean Matching (PMM)
# MICE
# Multivariate Imputation via Chained Equations. MICE assumes that the missing data are missing at random (MAR), which means that the probability 
# that a value is missing depends only on observed value and can be predicted using them. It imputes data on a variable by variable basis by specifying
# an imputations model per variable. For further information please check the links below:
# https://datascienceplus.com/imputing-missing-data-with-r-mice-package/
# https://www.rdocumentation.org/packages/mice/versions/3.8.0/topics/mice 
#

# Step 1: Removing columns that have no impact on distribution.
trainMICE <- train_condensed_spsurAdded
trainMICE_condensed <- subset(trainMICE, select = -c(SibSp, Ticket, Cabin, Embarked, FamilySize, Title, SpouseSurvived))


# Step 2: Creating the Predictor Matrix to ignore PassengerId
allVars <- imputerVar <- c("Survived", "Pclass", "Sex", "Age", "Parch", "Fare")
missVars <- names(trainMICE)[colSums(is.na(trainMICE)) >0]
predictorMatrix <- matrix(1, ncol = length(allVars), nrow = length(allVars))
rownames(predictorMatrix) <- allVars
colnames(predictorMatrix) <- allVars
diag(predictorMatrix) <- 0
PassengerIdMatrix_col <- matrix(0, ncol = 1, nrow = length(allVars))
colnames(PassengerIdMatrix_col) <- "PassengerId"
PassengerIdMatrix_row <- matrix(0, ncol = length(allVars)+1, nrow = 1)
rownames(PassengerIdMatrix_row) <- "PassengerId"
predictorMatrix <- cbind(PassengerIdMatrix_col, predictorMatrix)
predictorMatrix <- rbind(PassengerIdMatrix_row, predictorMatrix)


# Step 3: Visualising the null values.
md.pattern(trainMICE_condensed)


# Step 4: Performing MICE
trainMICE_imputed <- mice(trainMICE_condensed, m=5, maxit = 50, method = "pmm", seed = 500, predictorMatrix = predictorMatrix)
summary(trainMICE_imputed)

# Step 4: Visualising the output of the MICE imputation.
trainMICE_imputed$imp$Age
xyplot(trainMICE_imputed, Age ~ Fare)
densityplot(trainMICE_imputed)
stripplot(trainMICE_imputed, pch = 20, cex = 1.2)
trainMICE_imputedComplete_1 <- complete(trainMICE_imputed, 1)
trainMICE_imputedComplete_2 <- complete(trainMICE_imputed, 2)
trainMICE_imputedComplete_3 <- complete(trainMICE_imputed, 3)
trainMICE_imputedComplete_4 <- complete(trainMICE_imputed, 4)
trainMICE_imputedComplete_5 <- complete(trainMICE_imputed, 5)


install.packages('ggplot2')
library('ggplot2')
ggplot(trainMICE_condensed, aes(x=Age)) + geom_density(colour = "darkblue") + geom_vline(aes(xintercept = mean(Age, na.rm = TRUE)), colour="darkblue", linetype = "dashed", size =1)  + geom_density(data = trainMICE_imputedComplete_2, colour = "#FFA07A") #+ geom_density(data = trainMICE_imputedComplete_1, colour = "#FA8072") #+ geom_density(data = trainMICE_imputedComplete_3, colour = "#CD5C5C") + geom_density(data = trainMICE_imputedComplete_4, colour = "#B22222") + geom_density(data = trainMICE_imputedComplete_5, colour = "#8B0000")


# Step 5: Merging values from  back into the main data frame
AgeValues <- subset(trainMICE_imputedComplete_2, select = c("PassengerId", "Age"))
trainMICE <- merge(x=trainMICE_before, y=AgeValues, by = "PassengerId", all=TRUE)
trainMICE$Age.x <- NULL
colnames(trainMICE)[colnames(trainMICE)=="Age.y"] <- "Age"


#Method 4: Imputation by K-nearest neighbour
# https://www.rdocumentation.org/packages/bnstruct/versions/1.0.6/topics/knn.impute
# Note - the above documentation states that the data should be in a dataframe. This is not correct.
# The data will only accept the data in a matrix format. 

trainKNN_input <-subset(train_condensed_spsurAdded, select = -c(SibSp, Ticket, Cabin, Embarked, FamilySize, Title, SpouseSurvived))
train_kNN_Output <- kNN(trainKNN_input, variable = c("Age"), k = 10)
AgeValues <- subset(train_kNN_Output, select = c("PassengerId", "Age"))
trainkNN <- merge(x=train_condensed_spsurAdded, y=AgeValues, by = "PassengerId", all=TRUE)
trainkNN$Age.x <- NULL
colnames(trainkNN)[colnames(trainkNN)=="Age.y"] <- "Age"


#Method 5: Removing any with NA
trainREMOVED <- train_condensed_spsurAdded[complete.cases(train_condensed_spsurAdded),]




##########################
# Random Forest ML Model #
##########################

# First pass through the Random Forest to look at which variables are key contributors to the prediction model.
library('randomForest')
trainkNN_RF <- subset(trainkNN, select = c("Survived", "Pclass", "Sex", "Parch", "SibSp", "Fare", "FamilySize", "Title", "SpouseSurvived", "Age"))
trainkNN_RF$Survived <- as.character(trainkNN_RF$Survived)
trainkNN_RF$Survived <- as.factor(trainkNN_RF$Survived)
trainkNN_RF$SpouseSurvived <- as.factor(trainkNN_RF$SpouseSurvived)

RF_kNN <- randomForest(Survived ~ ., data = trainkNN_RF, ntree = 100, mtry = 3, importance = TRUE)
RF_kNN
varImpPlot(RF_kNN)

# Based on the OOBE (~16.5%), mean decrease in accuracy (~1) & mean decrease in GINI (~7), Parch will be dropped and the RF repeated.
trainkNN_RF$Parch <- NULL
RF_kNN_ParchDrop <- randomForest(Survived ~ ., data = trainkNN_RF, ntree = 100, mtry = 3, importance = TRUE)
RF_kNN_ParchDrop
varImpPlot(RF_kNN_ParchDrop)

# Function to drop columns
ColumnDrop <- function(dataframe) {
  output <- subset(dataframe, select = c("Survived", "Pclass", "Sex", "SibSp", "Fare", "FamilySize", "Title", "SpouseSurvived", "Age"))
  output$Survived <- as.character(output$Survived)
  output$Survived <- as.factor(output$Survived)
  output$SpouseSurvived <- as.factor(output$SpouseSurvived)
  return(output)
}

# Having tested the ntree for this at 50, 100, 150, 200 & 500, seeing minimal improvements >100 so this will be used for each of the differently imputed datasets
# SPLIT DATA 80:20 for train and validate sets
trainMODE_RF <- ColumnDrop(trainMODE)
trainMEAN_RF <- ColumnDrop(trainMEAN)
trainMICE_RF <- ColumnDrop(trainMICE)
trainkNN_RF <- ColumnDrop(trainkNN)
trainRemoved_RF <- ColumnDrop(trainREMOVED)


RF_MODE <- randomForest(Survived ~ ., data = trainMODE_RF, ntree = 100, mtry = 3, importance = TRUE)
RF_MODE
RF_MEAN <- randomForest(Survived ~ ., data = trainMEAN_RF, ntree = 100, mtry = 3, importance = TRUE)
RF_MEAN
RF_MICE <- randomForest(Survived ~ ., data = trainMICE_RF, ntree = 100, mtry = 3, importance = TRUE)
RF_MICE
RF_kNN <- randomForest(Survived ~ ., data = trainkNN_RF, ntree = 100, mtry = 3, importance = TRUE)
RF_kNN
RF_REMOVED <- randomForest(Survived ~ ., data = trainRemoved_RF, ntree = 100, mtry = 3, importance = TRUE)
RF_REMOVED



# ROC & AUC
library('ROCR')
prediction_for_ROC <- predict(RF_kNN_ParchDrop, validation[,-5], type = "prob")
