### Project of Tatanic Survival Rate Pridiction ###

# 0. Loading required R packages ----------------------------------------------- 
library(tidyverse)
library(data.table)
library(ROCR)
library(caret)
library(kernlab)
library(e1071)
library(broom)
library(gbm)
library(Rfit)
library(MASS)
library(caret)
library(rattle)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ggplot2)

# function for visualization of the confusion matrix
draw_confusion_matrix <- function(cm,mod) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title(paste0('Confusion Matrix for ',mod), cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Class1', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Class2', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Class1', cex=1.2, srt=90)
  text(140, 335, 'Class2', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
} 


# 1. Import & Clean Data -------------------------------------------------------

# Load the data
mydir <- 'C:/Users/oneda/Desktop/ZZ/graud/course/Data Mining/Fproject/titanic data/'
train <- read.csv(paste0(mydir,'train_complete_age&survive.csv'), na.strings=c("",".","NA")) %>% {.[,-1]}
test <- read.csv(paste0(mydir,'test_complete_age&survive.csv'),na.strings=c("",".","NA"))%>% {.[,-1]}

# merge the two datasets
passenger <- rbind(train,test) %>%
  {transform(., Pclass = as.factor(Pclass), # reformat categorical var to factor
             Sex = as.factor(Sex),
             Embarked = as.factor(Embarked))}%>%
  {mutate(., FamSize = SibSp + Parch)} %>% # feature engineering
  {na.omit(.[,c('Survived','Name',"Pclass","Sex", "Age", "SibSp", "Parch", 
                "FamSize", "Fare", "Embarked")])}

# transform Sex and Embarked to categorical variable for KNN classification
passenger$Sex.dum <- 0
passenger$Sex.dum[passenger$Sex=='male'] <- 1
passenger$Sex.dum <- as.factor(passenger$Sex.dum)

passenger$Embarked.dum <- 0
passenger$Embarked.dum[passenger$Embarked.dum=='Q'] <- 1
passenger$Embarked.dum[passenger$Embarked.dum=='S']<- 2
passenger$Embarked.dum <- as.factor(passenger$Sex.dum)

# split the data into train and test
set.seed(1)
tr.size <- floor(0.7*nrow(passenger))
tr.ind <- sample(seq_len(nrow(passenger)),tr.size)
train <- passenger[tr.ind,]
test <- passenger[-tr.ind,]


# Specify variables used for modeling #
yVar <- "Survived"

xVar <- c("Pclass","Sex", "Age", 
          "SibSp", "Parch", "Fare", "Embarked")# use SibSp and Parch

xVar_selected <- c("Pclass","Sex", "Age", 
                   "FamSize", "Fare", "Embarked")# use FamSize

# 2. plot relation s~variables -------------------------------------------------

# # Survivial Rate by Passenger Class

Pclass.plotdata <- passenger %>%
  group_by(Pclass, Survived) %>%
  summarize(n = n()) %>% 
  mutate(pct = n/sum(n),
         lbl = scales::percent(pct))      
Pclass.plotdata

Pclassplot<-ggplot(Pclass.plotdata, 
                   aes(x = factor(Pclass,
                                  levels = c("1", "2", "3")),     
                       y = pct,                                                
                       fill = factor(Survived, 
                                     levels = c("0", "1"),             
                                     labels = c("victim",
                                                "survivor")))) + 
  geom_bar(stat = "identity",                                     
           position = "fill") +                                   
  scale_y_continuous(breaks = seq(0, 1, .2), 
                     label = scales::percent) +
  geom_text(aes(label = lbl),                                     
            size = 3, 
            position = position_stack(vjust = 0.5)) +
  scale_fill_brewer(palette = "Set2") +
  labs(y = "Percent", 
       fill = "Status",
       x = "passenger Class",
       title = "Survivial Rate by Passenger Class") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Survivial Rate by Passenger Class.png", Pclassplot , width = 4, height = 3, dpi = 800)

# outdir <- 'C:/Users/oneda/Desktop/ZZ/graud/course/Data Mining/Fproject/plot/'
# tiff(paste0(outdir,'Pclassplot.png'), units="in", width=5, height=5, res=300)
# draw_confusion_matrix(Pclassplot,'Logistic Regression with SibSp + Parch')
# dev.off()

# Survivial Rate by Gender

Gender.plotdata <- passenger %>%
  group_by(Sex, Survived) %>%
  summarize(n = n()) %>% 
  mutate(pct = n/sum(n),
         lbl = scales::percent(pct))      
Gender.plotdata

Genderplot <- ggplot(Gender.plotdata, 
                     aes(x = factor(Sex,
                                    levels = c("female", "male")),     
                         y = pct,                                                
                         fill = factor(Survived, 
                                       levels = c("0", "1"),             
                                       labels = c("victim",
                                                  "survivor")))) + 
  geom_bar(stat = "identity",                                     
           position = "fill") +                                   
  scale_y_continuous(breaks = seq(0, 1, .2), 
                     label = scales::percent) +
  geom_text(aes(label = lbl),                                     
            size = 3, 
            position = position_stack(vjust = 0.5)) +
  scale_fill_brewer(palette = "Set2") +
  labs(y = "Percent", 
       fill = "Status",
       x = "Gender",
       title = "Survivial Rate by Gender") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Survivial Rate by Gender.png", Genderplot , width = 4, height = 3, dpi = 800)

# Survivial Rate by Port of Embarkation

Embarked.plotdata <- passenger %>%
  group_by(Embarked, Survived) %>%
  summarize(n = n()) %>% 
  mutate(pct = n/sum(n),
         lbl = scales::percent(pct))      
Embarked.plotdata

Enbarkedplot<-ggplot(Embarked.plotdata, 
                     aes(x = factor(Embarked,
                                    levels = c("C", "S", "Q"),
                                    labels = c("Cherbourg","Queenstown","Southampton")),     
                         y = pct,                                                
                         fill = factor(Survived, 
                                       levels = c("0", "1"),             
                                       labels = c("victim",
                                                  "survivor")))) + 
  geom_bar(stat = "identity",                                     
           position = "fill") +                                   
  scale_y_continuous(breaks = seq(0, 1, .2), 
                     label = scales::percent) +
  geom_text(aes(label = lbl),                                     
            size = 3, 
            position = position_stack(vjust = 0.5)) +
  scale_fill_brewer(palette = "Set2") +
  labs(y = "Percent", 
       fill = "Status",
       x = "Port of Embarkation",
       title = "Survivial Rate by Port of Embarkation") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Survivial Rate by Port of Embarkation.png", Enbarkedplot , width = 4, height = 3, dpi = 800)

# Survivial by Age

## survivor count by Age
survived.passenger <- passenger[which(passenger$Survived==1),]

Agecountplot<-ggplot(survived.passenger,aes(x = Age))+ 
  geom_histogram(bins = 80,
                 fill="cornflowerblue",
                 color="black")+
  labs(x="Age",
       y="count",
       title = "Survivor by Age")

ggsave("Survivor by Age.png", Agecountplot , width = 8, height = 3, dpi = 800)

## survivor rate by Age

Age.plotdata <- passenger %>%
  group_by(Age, Survived) %>%
  summarize(n = n()) %>% 
  mutate(pct = n/sum(n),
         lbl = scales::percent(pct))      
Age.plotdata

Ageplot<-ggplot(Age.plotdata, 
                aes(x = factor(Age,
                               levels = c("0.17","0.33","0.42","0.67","0.75","0.83","0.92",
                                          "0","1","2","3","4","5","6","7","8","9","10",
                                          "11","11.5","12","13","14","14.5","15","16","17","18","18.5","19",
                                          "20","20.5","21","22","22.5","23","23.5","24","24.5","25","26","26.5","27","28",
                                          "28.5","29","30","30.5","31","32","32.5","33","34","34.5","35","36","36.5","37",
                                          "38","38.5","39","40","40.5","41","42","43","44","45","45.5","46",
                                          "47","48","49","50","51","52","53","54","55","55.5","60.5",
                                          "56","57","58","59","60","61","62","63","64",
                                          "65","66","70","70.5","71","74","80")),
                    y = pct,                                                
                    fill = factor(Survived, 
                                  levels = c("0", "1"),             
                                  labels = c("victim",
                                             "survivor")))) + 
  geom_bar(stat = "identity",                                     
           position = "fill") +                                   
  scale_y_continuous(breaks = seq(0, 1, .2), 
                     label = scales::percent) +
  scale_fill_brewer(palette = "Set2") +
  labs(y = "Percent", 
       fill = "Status",
       x = "Age",
       title = "Survivial Rate by Age") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Survivial Rate by Age.png", Ageplot , width = 8, height = 3, dpi = 800)

# Survivial by Fare
## survivor count by Fare

ggplot(survived.passenger,aes(x = Fare))+ 
  geom_histogram(bins = 80,
                 fill="cornflowerblue",
                 color="black")+
  labs(x="Fare",
       y="count",
       title = "Survivor by Fare")

# Survivial by SibSp
## survivor count by SibSp

ggplot(survived.passenger,aes(x = SibSp))+ 
  geom_histogram(bins = 80,
                 fill="cornflowerblue",
                 color="black")+
  labs(x="SibSp",
       y="count",
       title = "Survivor by SibSp")
## survivor rate by SibSp
SibSp.plotdata <- passenger %>%
  group_by(SibSp, Survived) %>%
  summarize(n = n()) %>% 
  mutate(pct = n/sum(n),
         lbl = scales::percent(pct))      
SibSp.plotdata

sibspplot<-ggplot(SibSp.plotdata, 
                  aes(x = factor(SibSp,
                                 levels = c("0", "1", "2","3","5","8")),     
                      y = pct,                                                
                      fill = factor(Survived, 
                                    levels = c("0", "1"),             
                                    labels = c("victim",
                                               "survivor")))) + 
  geom_bar(stat = "identity",                                     
           position = "fill") +                                   
  scale_y_continuous(breaks = seq(0, 1, .2), 
                     label = scales::percent) +
  geom_text(aes(label = lbl),                                     
            size = 3, 
            position = position_stack(vjust = 0.5)) +
  scale_fill_brewer(palette = "Set2") +
  labs(y = "Percent", 
       fill = "Status",
       x = "Number of SibSp",
       title = "Survivial Rate by SibSp") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Survivial Rate by SibSp.png", sibspplot , width = 5, height = 3, dpi = 800)

# Survivial by Parch
## survivor count by Parch

ggplot(survived.passenger,aes(x = Parch))+ 
  geom_histogram(bins = 80,
                 fill="cornflowerblue",
                 color="black")+
  labs(x="Parch",
       y="count",
       title = "Survivor by Parch")

## survivor rate by Parch
Parch.plotdata <- passenger %>%
  group_by(Parch, Survived) %>%
  summarize(n = n()) %>% 
  mutate(pct = n/sum(n),
         lbl = scales::percent(pct))      
Parch.plotdata

parchplot<-ggplot(Parch.plotdata, 
                  aes(x = factor(Parch,
                                 levels = c("0", "1", "2","3","4","5","6","9")),     
                      y = pct,                                                
                      fill = factor(Survived, 
                                    levels = c("0", "1"),             
                                    labels = c("victim",
                                               "survivor")))) + 
  geom_bar(stat = "identity",                                     
           position = "fill") +                                   
  scale_y_continuous(breaks = seq(0, 1, .2), 
                     label = scales::percent) +
  geom_text(aes(label = lbl),                                     
            size = 3, 
            position = position_stack(vjust = 0.5)) +
  scale_fill_brewer(palette = "Set2") +
  labs(y = "Percent", 
       fill = "Status",
       x = "Number of Parch",
       title = "Survivial Rate by Parch") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Survivial Rate by Parch.png", parchplot , width = 5, height = 3, dpi = 800)

# Survivial by FamSize
## survivor count by FamSize

ggplot(survived.passenger,aes(x = FamSize))+ 
  geom_histogram(bins = 80,
                 fill="cornflowerblue",
                 color="black")+
  labs(x="FamSize",
       y="count",
       title = "Survivor by FamSize")

## survivor rate by FamSize
FamSize.plotdata <- passenger %>%
  group_by(FamSize, Survived) %>%
  summarize(n = n()) %>% 
  mutate(pct = n/sum(n),
         lbl = scales::percent(pct))      
FamSize.plotdata

famsizeplot<-ggplot(FamSize.plotdata, 
                    aes(x = factor(FamSize,
                                   levels = c("0", "1", "2","3","4","5","6","7","10")),     
                        y = pct,                                                
                        fill = factor(Survived, 
                                      levels = c("0", "1"),             
                                      labels = c("victim",
                                                 "survivor")))) + 
  geom_bar(stat = "identity",                                     
           position = "fill") +                                   
  scale_y_continuous(breaks = seq(0, 1, .2), 
                     label = scales::percent) +
  geom_text(aes(label = lbl),                                     
            size = 3, 
            position = position_stack(vjust = 0.5)) +
  scale_fill_brewer(palette = "Set2") +
  labs(y = "Percent", 
       fill = "Status",
       x = "Number of FamSize",
       title = "Survivial Rate by FamSize") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("Survivial Rate by FamSize.png", famsizeplot , width = 6, height = 3, dpi = 800)

# 3. Logistic regression  ------------------------------------

#### Build logistic models ----

# Model 1
log.fit1 <- glm(Survived~., data = train[,c(yVar, xVar)],family = "binomial")
summary(log.fit1)
log.prob1 <- predict(log.fit1, test[,xVar],type = "response")
log.pred1 <- rep(0,length(log.prob1))
log.pred1[log.prob1 > 0.5] = 1


# Model 2
log.fit <- glm(Survived~., data = train[,c(yVar, xVar_selected)],family = "binomial")
summary(log.fit)
log.prob <- predict(log.fit, test[,xVar_selected],type = "response")
log.pred <- rep(0,length(log.prob))
log.pred[log.prob > 0.5] = 1


#### Check multicollinearity ----

# Model 1
car::vif(log.fit1)
# Model 2
car::vif(log.fit) 


#### Confusion matrix -----

# Model 1
cm.log1 <- confusionMatrix(as.factor(log.pred1), as.factor(test$Survived),
                           mode = "everything", positive = "1")
draw_confusion_matrix(cm.log1,'Logistic Regression with SibSp + Parch')

# Model 2
cm.log <-  confusionMatrix(as.factor(log.pred), as.factor(test$Survived),
                           mode = "everything", positive = "1")
draw_confusion_matrix(cm.log,'Logistic Regression with FamSize')


#### ROC curve ----

# Model 1
log.roc = prediction(log.prob1, test$Survived)
perf.roc = performance(log.roc, "tpr", "fpr")
pred.auc = performance(log.roc,"auc")
plot(perf.roc,main = "ROC",col = "firebrick", lwd =2)
abline(0,1,col = "steelblue",lty = 2,lwd =2)
text(0.18,0.86,paste0("AUC=",round(pred.auc@y.values[[1]],digits=4)), srt=0.2, pos=3)
dev.off()


# Model 2

log.roc = prediction(log.prob, test$Survived)
perf.roc = performance(log.roc, "tpr", "fpr")
pred.auc = performance(log.roc,"auc")
plot(perf.roc,main = "ROC",col = "firebrick", lwd =2)
abline(0,1,col = "steelblue",lty = 2,lwd =2)
text(0.18,0.86,paste0("AUC=",round(pred.auc@y.values[[1]],digits=4)), srt=0.2, pos=3)
dev.off()


# 4. LDA ------------------------------------------------

#### build LDA models ----

# Model 1
lda.tr1 = lda(Survived ~ ., data = passenger[,c(yVar,xVar)], subset = tr.ind)
lda.pred.tr1 = predict(lda.tr1, train)
lda.pred.te1 = predict(lda.tr1, test)

# Model 2
lda.tr = lda(Survived ~ ., data = passenger[,c(yVar,xVar_selected)], subset = tr.ind)
lda.pred.tr = predict(lda.tr, train)
lda.pred.te = predict(lda.tr, test)


#### confusion matrix ----

# Model 1
cm.lda1 <- confusionMatrix(as.factor(lda.pred.te1$class), as.factor(test$Survived))
draw_confusion_matrix(cm.lda1,'LDA with SibSP + Parch')

# Model 2
cm.lda <- confusionMatrix(as.factor(lda.pred.te$class), as.factor(test$Survived))
draw_confusion_matrix(cm.lda,'LDA with FamSize')


#### Response classification plots -----

# Model 1
ldahist(lda.pred.te1$x[,1], g= lda.pred.te1$class)
dev.off()

plot(lda.pred.te1$x[,1], lda.pred.te1$class, col=test$Survived)
dev.off()

# Model 2
par(mar=c(1,1,1,1))
ldahist(lda.pred.te$x[,1], g= lda.pred.te$class)
dev.off()

plot(lda.pred.te$x[,1], lda.pred.te$class, col=test$Survived)
dev.off()


#### ROC -----

# Model 1
log.roc1 = prediction(lda.pred.te1$posterior[,2], test$Survived)
perf.roc1 = performance(log.roc1, "tpr", "fpr")
pred.auc1 = performance(log.roc1,"auc")
plot(perf.roc1,main = "ROC",col = "firebrick", lwd =2)
abline(0,1,col = "steelblue",lty = 2,lwd =2)
text(0.18,0.86,paste0("AUC=",round(pred.auc1@y.values[[1]],digits=4)), srt=0.2, pos=3)
dev.off()

# Model 2
log.roc = prediction(lda.pred.te$posterior[,2], test$Survived)
perf.roc = performance(log.roc, "tpr", "fpr")
pred.auc = performance(log.roc,"auc")
plot(perf.roc,main = "ROC",col = "firebrick", lwd =2)
abline(0,1,col = "steelblue",lty = 2,lwd =2)
text(0.18,0.86,paste0("AUC=",round(pred.auc@y.values[[1]],digits=4)), srt=0.2, pos=3)
dev.off()


# 5. QDA Analysis ------------------------------------------------

#### Build QDA models -----

# Model 1
qda.tr1 = qda(Survived ~ ., data = passenger[,c(yVar,xVar)], subset = tr.ind)
qda.pred.tr1 = predict(qda.tr1, train)
qda.pred.te1 = predict(qda.tr1, test)

# Model 2
qda.tr = qda(Survived ~ ., data = passenger[,c(yVar,xVar_selected)], subset = tr.ind)
qda.pred.tr = predict(qda.tr, train)
qda.pred.te = predict(qda.tr, test)


#### Confusion matrix ----

# Model 1
cm.qda1 <- confusionMatrix(as.factor(qda.pred.te1$class), as.factor(test$Survived))
draw_confusion_matrix(cm.qda1,'QDA with SibSP + Parch')

# Model 2
cm.qda <- confusionMatrix(as.factor(qda.pred.te$class), as.factor(test$Survived))
draw_confusion_matrix(cm.qda,'QDA with FamSize')


#### Response classification plot for test data ----

# Model 1
plot(qda.pred.te1$posterior[,2], qda.pred.te1$class, col=test$Survived)
dev.off()

# Model 2
plot(qda.pred.te$posterior[,2], qda.pred.te$class, col=test$Survived)
dev.off()


#### ROC -----

# Model 1
log.roc1 = prediction(qda.pred.te1$posterior[,2], test$Survived)
perf.roc1 = performance(log.roc1, "tpr", "fpr")
pred.auc1 = performance(log.roc1,"auc")
plot(perf.roc1,main = "ROC",col = "firebrick", lwd =2)
abline(0,1,col = "steelblue",lty = 2,lwd =2)
text(0.18,0.86,paste0("AUC=",round(pred.auc1@y.values[[1]],digits=4)), srt=0.2, pos=3)
dev.off()

# Model 2
log.roc = prediction(qda.pred.te$posterior[,2], test$Survived)
perf.roc = performance(log.roc, "tpr", "fpr")
pred.auc = performance(log.roc,"auc")
plot(perf.roc,main = "ROC",col = "firebrick", lwd =2)
abline(0,1,col = "steelblue",lty = 2,lwd =2)
text(0.18,0.86,paste0("AUC=",round(pred.auc@y.values[[1]],digits=4)), srt=0.2, pos=3)
dev.off()


# 6. KNN model ----------------------------------------------------

#### Build KNN models ----

# use dummy variables created in section 1 for categorical vars
knn.xVar <-c("Pclass","Sex.dum", "Age", 
             "SibSp", "Parch", "Fare", "Embarked.dum") # use SibSp and Parch
knn.xVar_selected <- c("Pclass","Sex.dum", "Age", 
                       "FamSize", "Fare", "Embarked.dum")# use FamSize
# Model 1
knn_caret1 <- train(train[,knn.xVar], as.factor(train$Survived), 
                    method = "knn", preProcess = c("center","scale"))
knnPredict1 <- predict(knn_caret1, newdata = test[,knn.xVar]) 
knnProb1 <- predict(knn_caret1, newdata = test[,knn.xVar],type = "prob") 

# Model 2
knn_caret <- train(train[,knn.xVar_selected], as.factor(train$Survived), 
                   method = "knn", preProcess = c("center","scale"))
knnPredict <- predict(knn_caret, newdata = test[,knn.xVar_selected]) 
knnProb <- predict(knn_caret, newdata = test[,knn.xVar_selected],type = "prob") 


#### confusion matrix ----

# Model 1
cm.knn1 <- confusionMatrix(knnPredict1, as.factor(test$Survived))
draw_confusion_matrix(cm.knn1,'KNN with SibSP + Parch')

# Model 2
cm.knn <- confusionMatrix(knnPredict, as.factor(test$Survived))
draw_confusion_matrix(cm.knn,'KNN with FamSize')


#### ROC ----

# Model 1
log.roc = prediction(knnProb1[,'1'], test$Survived)
perf.roc = performance(log.roc, "tpr", "fpr")
pred.auc = performance(log.roc,"auc")
plot(perf.roc,main = "ROC",col = "firebrick", lwd =2)
abline(0,1,col = "steelblue",lty = 2,lwd =2)
text(0.18,0.86,paste0("AUC=",round(pred.auc@y.values[[1]],digits=4)), srt=0.2, pos=3)
dev.off()

# Model 2
log.roc = prediction(knnProb[,'1'], test$Survived)
perf.roc = performance(log.roc, "tpr", "fpr")
pred.auc = performance(log.roc,"auc")
plot(perf.roc,main = "ROC",col = "firebrick", lwd =2)
abline(0,1,col = "steelblue",lty = 2,lwd =2)
text(0.18,0.86,paste0("AUC=",round(pred.auc@y.values[[1]],digits=4)), srt=0.2, pos=3)
dev.off()

# 7.Decision tree---------------------------------------------------------------
set.seed(1)

train.survive <- train
train.survive = na.omit(train.survive)

test.survive <- test
test.survive = na.omit(test.survive)

titanic = rbind(train.survive,test.survive)
titanic[sapply(titanic, is.character)] <- lapply(titanic[sapply(titanic, is.character)], as.factor)

#Creating additional features Family Size and a classification for Age
titanic$FamSize = titanic$SibSp + titanic$Parch
titanic$Embarked = ifelse(titanic$Embarked=="",NA,titanic$Embarked)
titanic = na.omit(titanic)
titanic$AgeClass <- NA
titanic$AgeClass[titanic$Age <= 12] <- "Child"
titanic$AgeClass[titanic$Age > 12 & titanic$Age <= 18] <- "Adolescent"
titanic$AgeClass[titanic$Age > 18 & titanic$Age <= 59] <- "Adult"
titanic$AgeClass[titanic$Age >= 60] <- "Elderly Adult"
titanic$AgeClass = as.factor(titanic$AgeClass)
titanic$Survived = as.factor(titanic$Survived)
titanic$Pclass = as.factor(titanic$Pclass)

#Stacked Barplots for Sex/Survived and AgeClass/Survived
ggplot(titanic, aes(fill=Survived, x=Sex)) + geom_bar(position="stack", stat="count", color="black") + scale_fill_manual(values = c("#DADAEB", "#9E9AC8")) + ggtitle("Survival grouped by Sex") + xlab("") + ylab("Number of Passengers")

ggplot(titanic, aes(fill=Survived, x=AgeClass)) + geom_bar(position="stack", stat="count", color="black") + scale_fill_manual(values = c("#DADAEB", "#9E9AC8")) + xlab("") + ylab("Number of Passengers")


#Splitting the data in a 70/30 split
trainIndex = createDataPartition(titanic$Survived, p=.70, list=FALSE)

train = titanic[trainIndex,]
test = titanic[-trainIndex,]

#Classification tree that uses Fare for a node
survive.tree = train(Survived ~Pclass+Sex+Age+SibSp+Parch+Fare, data=train, method="rpart", trControl = trainControl(method = "cv"))
#Classification tree that uses FamSize for a node
survive.tree2 = train(Survived ~ Pclass+Sex+Age+FamSize+Fare, data=train, method="rpart", trControl = trainControl(method = "cv"))

#Plotting both of the trees for interpretation

rpart.plot(survive.tree2$finalModel, extra=4)

survive.pred = predict(survive.tree, newdata=test)
cm.tree = confusionMatrix(survive.pred, test$Survived)
cm.tree

survive.pred2 = predict(survive.tree2, test)
cm.tree2 = confusionMatrix(survive.pred2, test$Survived)
cm.tree2

cv.ctrl = trainControl(method="repeatedcv", repeats=3)
rf = train(Survived~Sex+Pclass+Age+SibSp+Parch+Fare+Embarked, data=train, method="rf", ntree=1000, trControl=cv.ctrl)
rf

rf.pred = predict(rf,test)
cm.rf = confusionMatrix(rf.pred, test$Survived)
cm.rf

par(mfrow=c(1,2))
fourfoldplot(cm.tree2$table, std="all.max", main="Classification Tree")
fourfoldplot(cm.rf$table, std="all.max", main="Random Forest")