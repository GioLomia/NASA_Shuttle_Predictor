library(caret)
library(xgboost)
library(mlbench)
library(tidyverse)
library(recipes)
library(DataExplorer)
library(recipes)
library(MASS)
library(glmnet)
library(tidyverse)
library(caret)
library(keras)
library(rsample)
library(tidyverse)
library(ISLR)
library(tidyverse)
library(polycor)
library(esquisse)
library(recipes)

raw_data<-shuttle_new
glimpse(raw_data)
plot_histogram(raw_data)

train_test_split<-initial_split(raw_data,0.8)

train_in <- training(train_test_split)
test_in <-testing(train_test_split)

rec_obj <- recipe(Column10 ~ ., data = train_in) %>%
  #step_center(all_numeric(), -all_outcomes()) %>%
  #step_scale(all_numeric(), -all_outcomes()) %>%
  #insert normalization step here %>%
  #step_BoxCox(all_numeric(),-all_outcomes())%>%
  #step_dummy(all_nominal(),-all_outcomes())%>%
  step_nzv(all_predictors(), -all_outcomes()) %>%
  prep(data = train_in)

train_clean <- bake(rec_obj, new_data = train_in)
test_clean <- bake(rec_obj, new_data = test_in)

#Step 5 - Prepare data for lasso model
x_train <- model.matrix(Column10 ~ ., train_clean)[,-1]
y_train <- as.factor(train_clean$Column10)
y_train<-make.names(y_train)
x_test <- model.matrix(Column10 ~ ., test_clean)[,-1]
y_test <- as.factor(test_clean$Column10)
y_test<-make.names(y_test)
modelLookup("xgbTree")
xgb_grid <- expand.grid( nrounds=500,
                         max_depth=c(1,2), 
                         eta=c(0.0025,0.05),
                         gamma=c(2), 
                         colsample_bytree=c(1),
                         min_child_weight=c(1,2),
                         subsample=c(0,1,2)
                         
)
ctrl<-trainControl(method = "repeatedcv",
                   number=2,
                   repeats = 2,
                   verboseIter = TRUE,
                   returnResamp = "all",
                   returnData=TRUE,
                   allowParallel = TRUE,
                   classProbs = TRUE)

BC_xgb <- train(x_train,
                y_train,
                trControl = ctrl,
                tuneGrid = xgb_grid,
                method = "xgbTree",
                metric="ROC")

BC_xgb
ggplot(varImp(BC_xgb,scale = F))
ggplot(BC_xgb)

BC.xgbpred=predict(BC_xgb,newdata=x_test)

table(BC.xgbpred,y_test)
mean(BC.xgbpred==y_test)




