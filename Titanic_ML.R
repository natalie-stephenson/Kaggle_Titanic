############################################################################
# KAGGLE MACHINE LEARNING PROJECT - TITANIC: MACHINE LEARNNG FROM DISASTER #
# Date - 30th May 2019                                                     #
# Skills utilised - data cleaning, feature engineering, ML (random forest) #
############################################################################

install.packages('dplyr')
install.packages('mice')
install.packages('vim')
install.packages('randomForest')
install.packages('splitstackshape')

#####################
# DATA MANIPULATION #
#####################

# Import and Load Librarys
library('dplyr') #Data manipulation tool
library('splitstackshape') #Data manipulation tool
library('mice')  #Multivariate Imputation by Chained Equation - dealing with missing values
library('vim')   #K-nearest neighbour - dealing with missing values


## Read Files
train <- read.csv(file = "train.csv")
gender <- read.csv(file = "gender_submission.csv")


## Feature Engineering

# Determining family size
train$FamilySize <- train$SibSp + train$Parch

# Determining spouse survival
train <- cSplit(train, "Name", sep = "(", direction = "wide")
colnames(train)[colnames(train)=="Name_1"] <- "Name"
colnames(train)[colnames(train)=="Name_2"] <- "MaidenName"
train$Name_3 <- NULL
train$MaidenName <- gsub(")", "", train$MaidenName)

train <- cSplit(train, "Name", sep = ",", direction = "wide")
colnames(train)[colnames(train)=="Name_1"] <- "Surname"
train <- cSplit(train, "Name_2", sep = ".", direction = "wide")
colnames(train)[colnames(train)=="Name_2_1"] <- "Title"
colnames(train)[colnames(train)=="Name_2_2"] <- "FirstNames"
train$FullName <- paste(train$FirstNames, train$Surname, sep = " ", collapse = NULL)

Spouse <- train[,c("MaidenName", "FullName", "Survived")]
colnames(Spouse)[colnames(Spouse)=="Survived"] <- "WifeSurvived"
Spouse <- na.omit(Spouse)

############### need to combine with husband to determine husband survival, then join back to original table???
############### or if statement???


#Dealing with missing values - Imputation by most frequent value
trainMODE <- train

  
#Dealing with missing values - Imputation by MICE
trainMICE <-

#Dealing with missing values - Imputation by K-nearest neighbour
trainKNN <-


##########################
# Random Forest ML Model #
##########################

library('randomForest')

