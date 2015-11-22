## Loading packages
suppressPackageStartupMessages(library('BBmisc'))
pkgs <- c('jsonlite','plyr','stringr','ff','ffbase')
suppressAll(lib(pkgs)); rm(pkgs)

library(e1071)
library(vcd)
library(caret)
library(klaR)
library(reshape2)
library(qdapTools)


##PreProcessing Data

##Setting Working directory
setwd("~/Coursera DS/Capstone2/yelp_dataset_challenge_academic_dataset/yelp_dataset_challenge_academic_dataset")
name_file <- paste0(getwd(),'/yelp_academic_dataset_',c('business','checkin','review','tip','user'),'.json')
YelpDat <- llply(as.list(name_file), function(x) stream_in(file(x),pagesize = 10000))
names(YelpDat) <- c('business','checkin','review','tip','user')
YelpBus<-flatten(YelpDat[['business']],recursive=TRUE)
YelpBus_Restaurant<-YelpBus[grep("Restaurants",YelpBus$categories),]
YelpRev<-flatten(YelpDat[['review']],recursive=TRUE)
YelpRat_flat = YelpRev[,c("user_id","business_id","stars")]
YelpRat_Rest = YelpRat_flat[YelpRat_flat$business_id %in% YelpBus_Restaurant$business_id,]
YelpRating_unique = YelpRat_Rest[!duplicated(YelpRat_Rest[1:2]),]
UserRev_frequency = data.frame(table(YelpRating_unique$user_id))
MostRev_review_users = UserRev_frequency[order(-UserRev_frequency$Freq),][1:10,]
head(MostRev_review_users)
Select_User_reviews = subset(YelpRating_unique, user_id == "kGgAARL2UmvCcTRfiscjug")
Reviewer_Model = join(YelpBus_Restaurant, Select_User_reviews, by = "business_id", type = "inner", match = "first")
Reviewer_Model$like[Reviewer_Model$stars >= 4] = "True"
Reviewer_Model$like[Reviewer_Model$stars < 4] = "False"
Categories_tab<-!!(mtabulate(Reviewer_Model$categories))
Reviewer_Model_tab<-cbind(Reviewer_Model,Categories_tab)
drop_cols = c("user_id","stars", "business_id","review_count")
Reviewer_Model_drop = Reviewer_Model_tab[,!(names(Reviewer_Model_tab) %in% drop_cols)]
Reviewer_Model_logical<-Reviewer_Model_drop[,c("like",names(Reviewer_Model_drop[,sapply(Reviewer_Model_drop,is.logical)]))]
Reviewer_Model_logical[is.na(Reviewer_Model_logical)]<-"FALSE"
trainYelpDataLV <- nearZeroVar(Reviewer_Model_logical, saveMetrics=TRUE)
trainYelpDataLV[trainYelpDataLV$nzv=="TRUE",]
L<-c(rownames(trainYelpDataLV[trainYelpDataLV$nzv=="TRUE",]))
SampleTestYelpData <- Reviewer_Model_logical[,!(names(Reviewer_Model_logical) %in% L)] 
SampleTestYelpData$like = as.factor(SampleTestYelpData$like)

# Training model
set.seed(123456)
in_train <- createDataPartition(SampleTestYelpData$like, p = .6, list = FALSE, times = 1)
Reviewer_Model_train <- SampleTestYelpData[ in_train,]
Reviewer_Model_test  <- SampleTestYelpData[-in_train,]
ModControl <- trainControl(method='cv', number=5, repeats=1, classProbs=TRUE, summaryFunction=twoClassSummary)
model <- train(like ~ ., data=Reviewer_Model_train, method='rf', trControl=ModControl)

#Results
PredTest <- predict(model, Reviewer_Model_test)
confusionMatrix(PredTest, Reviewer_Model_test$like)
suppressMessages(InSampleError<-1 - as.numeric(confusionMatrix(Reviewer_Model_train$like, predict(model, Reviewer_Model_train))$overall[1]))
OutOfSampleError <- 1 - as.numeric(confusionMatrix(Reviewer_Model_test$like, PredTest)$overall[1])
cat("In Sample Error Rate: ", InSampleError, "\n")
cat("Out of Sample Error Rate: ", OutOfSampleError, "\n")



