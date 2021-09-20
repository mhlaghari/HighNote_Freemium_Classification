library("readxl") # used to read excel files
library("dplyr") # used for data munging 
library("FNN") # used for knn regression (knn.reg function)
library("caret") # used for various predictive models
library("class") # for using confusion matrix function
library("rpart.plot") # used to plot decision tree
library("rpart")  # used for Regression tree
library("glmnet") # used for Lasso and Ridge regression
library('NeuralNetTools') # used to plot Neural Networks
library("PRROC") # top plot ROC curve
library("ROCR") # top plot lift curve
library("tidyverse")
library(mice)

data = read_csv('/Users/mhlaghari/Downloads/HN_data_PostModule.csv')

# create Y and X data frames
y = data %>% pull("adopter") %>% as.factor()
# exclude X1 since its a row number
x = data %>% select(-c("net_user", "male", "adopter"))

#Dealing with missing values
impute <- mice(x, m=3, method = 'pmm')
x <- complete(impute, 2)

#Feature Scaling
x_scaled <- as.data.frame(lapply(x, scale))

#Creating Train/test split
smp_size <- floor(0.75 * nrow(x_scaled))

# randomly select row numbers for training data set
train_ind <- sample(seq_len(nrow(x_scaled)), size = smp_size)

# creating test and training sets for x
x_train <- x_scaled[train_ind, ]
x_test <- x_scaled[-train_ind, ]

# creating test and training sets for y
y_train <- y[train_ind]
y_test <- y[-train_ind]

# Create an empty data frame to store results from different models
clf_results <- data.frame(matrix(ncol = 5, nrow = 0))
names(clf_results) <- c("Model", "Accuracy", "Precision", "Recall", "F1")

# Create an empty data frame to store TP, TN, FP and FN values
cost_benefit_df <- data.frame(matrix(ncol = 5, nrow = 0))
names(cost_benefit_df) <- c("Model", "TP", "FN", "FP", "TN")

# Cross validation 
cross_validation <- trainControl(## 10-fold CV
                                method = "repeatedcv",
                                number = 10,
                                ## repeated three times
                                repeats = 2)
# Hyperparamter tuning
# k = number of nrearest neighbours
Param_Grid <-  expand.grid( k = 1:4)

# fit the model to training data
knn_clf_fit <- train(x_train,
                     y_train, 
                     method = "knn",
                     tuneGrid = Param_Grid,
                     trControl = cross_validation )

# check the accuracy for different models
knn_clf_fit

# Plot accuracies for different k values
plot(knn_clf_fit)

# print the best model
print(knn_clf_fit$finalModel)

# Predict on test data
knnPredict <- predict(knn_clf_fit, newdata = x_test) 

# Print Confusion matrix, Accuracy, Sensitivity etc 
confusionMatrix(knnPredict, y_test)

# Add results into clf_results dataframe
x1 <- confusionMatrix(knnPredict, y_test, positive = '1')[["overall"]]
y1 <- confusionMatrix(knnPredict, y_test, positive = '1')[["byClass"]]

clf_results[nrow(clf_results) + 1,] <-  list(Model = "KNN Entire Dataset", 
                                             Accuracy = round (x1[["Accuracy"]],3), 
                                            Precision = round (y1[["Precision"]],3), 
                                            Recall = round (y1[["Recall"]],3), 
                                            F1 = round (y1[["F1"]],3))
# Print Accuracy and F1 score

cat("Accuarcy is ", round(x1[["Accuracy"]],3), "and F1 is ", round (y1[["F1"]],3)  )

# Add results into cost_benefit_df dataframe for cost benefit analysis 
a1 <- confusionMatrix(knnPredict, y_test)

cost_benefit_df[nrow(cost_benefit_df) + 1,] <-  list(Model = "KNN Entire Dataset", 
                                             TP = a1[["table"]][1], 
                                             FN = a1[["table"]][2], 
                                             FP = a1[["table"]][3], 
                                             TN = a1[["table"]][4])




XG_clf_fit <- train(x_train, 
                    y_train,
                    method = "xgbTree",
                    preProc = c("center", "scale"))

# print the final model
XG_clf_fit$finalModel

# Predict on test data
XG_clf_predict <- predict(XG_clf_fit,x_test)

# Print Confusion matrix, Accuracy, Sensitivity etc 
confusionMatrix(XG_clf_predict,  y_test )

# Add results into clf_results dataframe
x4 <- confusionMatrix(XG_clf_predict,  y_test, positive = '1' )[["overall"]]
y4 <- confusionMatrix(XG_clf_predict,  y_test, positive = '1' )[["byClass"]]

clf_results[nrow(clf_results) + 1,] <-  list(Model = "XG Boost Entire Dataset", 
                                             Accuracy = round (x4[["Accuracy"]],3), 
                                            Precision = round (y4[["Precision"]],3), 
                                            Recall = round (y4[["Recall"]],3), 
                                            F1 = round (y4[["F1"]],3))

# Print Accuracy and F1 score
cat("Accuarcy is ", round(x4[["Accuracy"]],3), "and F1 is ", round (y4[["F1"]],3)  )

# Add results into cost_benefit_df dataframe for cost benefit analysis 
a4 <- confusionMatrix(XG_clf_predict, y_test)

cost_benefit_df[nrow(cost_benefit_df) + 1,] <-  list(Model = "XGBoost Entire Dataset", 
                                             TP = a4[["table"]][1], 
                                             FN = a4[["table"]][2], 
                                             FP = a4[["table"]][3], 
                                             TN = a4[["table"]][4])

nn_clf_fit <- train(x_train,
                    y_train,
                    method = "nnet",
                    trace = F,
                    #tuneGrid = my.grid,
                    linout = 0,
                    stepmax = 100,
                    threshold = 0.01 )
print(nn_clf_fit)

# Predict on test data
nn_clf_predict <- predict(nn_clf_fit,x_test)

# Print Confusion matrix, Accuarcy, Sensitivity etc 
confusionMatrix(nn_clf_predict,  y_test)

# Add results into clf_results dataframe
x5 <- confusionMatrix(nn_clf_predict,  y_test, positive = '1')[["overall"]]
y5 <- confusionMatrix(nn_clf_predict,  y_test, positive = '1')[["byClass"]]

clf_results[nrow(clf_results) + 1,] <-  list(Model = "Neural Network Entire Dataset", 
                                             Accuracy = round (x5[["Accuracy"]],3), 
                                            Precision = round (y5[["Precision"]],3), 
                                            Recall = round (y5[["Recall"]],3), 
                                            F1 = round (y5[["F1"]],3))

# Print Accuracy and F1 score
cat("Accuarcy is ", round(x5[["Accuracy"]],3), "and F1 is ", round (y5[["F1"]],3)  )

# Add results into cost_benefit_df dataframe for cost benefit analysis 
a5 <- confusionMatrix(nn_clf_predict, y_test)

cost_benefit_df[nrow(cost_benefit_df) + 1,] <-  list(Model = "Neural Network Entire Dataset", 
                                             TP = a5[["table"]][1], 
                                             FN = a5[["table"]][2], 
                                             FP = a5[["table"]][3], 
                                             TN = a5[["table"]][4])




# Predict probabilities of each model to plot ROC curve
knnPredict_prob <- predict(knn_clf_fit, newdata = x_test, type = "prob") 
XG_boost_prob <- predict(XG_clf_fit, newdata = x_test, type = "prob")
nn_clf_prob <- predict(nn_clf_fit, newdata = x_test, type = "prob")

# List of predictions
preds_list <- list(knnPredict_prob[,1],  
                    XG_boost_prob[,1], nn_clf_prob[,1] )

# List of actual values (same for all)
m <- length(preds_list)
actuals_list <- rep(list(y_test), m)

# Plot the ROC curves
pred <- prediction(preds_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")

# calculate AUC for all models
AUC_models <- performance(pred, "auc")
auc_knn = round(AUC_models@y.values[[1]], 3)
auc_xg = round(AUC_models@y.values[[2]], 3)
auc_nn = round(AUC_models@y.values[[3]], 3)

# Plot the ROC curves
plot(rocs, col = as.list(1:m), main = "ROC Curves of different models")
legend(x = "bottomright", 
       legend = c( paste0("KNN - ", auc_knn), 
                   paste0("XG Boost - ", auc_xg), 
                   paste0("Neural Net - ", auc_nn)), fill = 1:m)

#Creating delta subset
z <- x %>% select(-c("friend_cnt","avg_friend_age","avg_friend_male",
                             "friend_country_cnt"          
,"subscriber_friend_cnt","songsListened","lovedTracks","posts",
"playlists","shouts","good_country"))

#Feature Scaling
z_scaled <- as.data.frame(lapply(z, scale))

#Creating Train/test split
smp_size <- floor(0.75 * nrow(z_scaled))

# randomly select row numbers for training data set
train_ind <- sample(seq_len(nrow(z_scaled)), size = smp_size)

# creating test and training sets for x
z_train <- z_scaled[train_ind, ]
z_test <- z_scaled[-train_ind, ]

# creating test and training sets for y
y1_train <- y[train_ind]
y1_test <- y[-train_ind]

# Create an empty data frame to store results from different models
clf_results1 <- data.frame(matrix(ncol = 5, nrow = 0))
names(clf_results1) <- c("Model", "Accuracy", "Precision", "Recall", "F1")

# Create an empty data frame to store TP, TN, FP and FN values
cost_benefit_df1 <- data.frame(matrix(ncol = 5, nrow = 0))
names(cost_benefit_df1) <- c("Model", "TP", "FN", "FP", "TN")

# Cross validation 
cross_validation <- trainControl(## 10-fold CV
                                method = "repeatedcv",
                                number = 10,
                                ## repeated three times
                                repeats = 2)
# Hyperparamter tuning
# k = number of nrearest neighbours
Param_Grid <-  expand.grid( k = 1:4)

# fit the model to training data
knn_clf_fit1 <- train(z_train,
                     y1_train, 
                     method = "knn",
                     tuneGrid = Param_Grid,
                     trControl = cross_validation )

# check the accuracy for different models
knn_clf_fit1

# Plot accuracies for different k values
plot(knn_clf_fit1)

# print the best model
print(knn_clf_fit1$finalModel)

# Predict on test data
knnPredict1 <- predict(knn_clf_fit1, newdata = z_test) 

# Print Confusion matrix, Accuracy, Sensitivity etc 
confusionMatrix(knnPredict1, y1_test)

# Add results into clf_results dataframe
x6 <- confusionMatrix(knnPredict1, y1_test, positive = '1')[["overall"]]
y6 <- confusionMatrix(knnPredict1, y1_test, positive = '1')[["byClass"]]

clf_results1[nrow(clf_results1) + 1,] <-  list(Model = "KNN Delta", 
                                             Accuracy = round (x6[["Accuracy"]],3), 
                                            Precision = round (y6[["Precision"]],3), 
                                            Recall = round (y6[["Recall"]],3), 
                                            F1 = round (y6[["F1"]],3))
# Print Accuracy and F1 score

cat("Accuarcy is ", round(x6[["Accuracy"]],3), "and F1 is ", round (y6[["F1"]],3)  )

# Add results into cost_benefit_df dataframe for cost benefit analysis 
a6 <- confusionMatrix(knnPredict1, y_test)

cost_benefit_df1[nrow(cost_benefit_df1) + 1,] <-  list(Model = "KNN Delta", 
                                             TP = a6[["table"]][1], 
                                             FN = a6[["table"]][2], 
                                             FP = a6[["table"]][3], 
                                             TN = a6[["table"]][4])

XG_clf_fit1 <- train(z_train, 
                    y1_train,
                    method = "xgbTree",
                    preProc = c("center", "scale"))

# print the final model
XG_clf_fit1$finalModel

# Predict on test data
XG_clf_predict1 <- predict(XG_clf_fit1,z_test)

# Print Confusion matrix, Accuracy, Sensitivity etc 
confusionMatrix(XG_clf_predict1,  y1_test )

# Add results into clf_results dataframe
x7 <- confusionMatrix(XG_clf_predict1,  y1_test, positive = '1' )[["overall"]]
y7 <- confusionMatrix(XG_clf_predict1,  y1_test, positive = '1' )[["byClass"]]

clf_results1[nrow(clf_results1) + 1,] <-  list(Model = "XG Boost Delta", 
                                             Accuracy = round (x7[["Accuracy"]],3), 
                                            Precision = round (y7[["Precision"]],3), 
                                            Recall = round (y7[["Recall"]],3), 
                                            F1 = round (y7[["F1"]],3))

# Print Accuracy and F1 score
cat("Accuarcy is ", round(x7[["Accuracy"]],3), "and F1 is ", round (y7[["F1"]],3)  )

# Add results into cost_benefit_df dataframe for cost benefit analysis 
a7 <- confusionMatrix(knnPredict1, y_test)

cost_benefit_df1[nrow(cost_benefit_df1) + 1,] <-  list(Model = "XG Boost Delta", 
                                             TP = a7[["table"]][1], 
                                             FN = a7[["table"]][2], 
                                             FP = a7[["table"]][3], 
                                             TN = a7[["table"]][4])



nn_clf_fit1 <- train(z_train,
                    y1_train,
                    method = "nnet",
                    trace = F,
                    #tuneGrid = my.grid,
                    linout = 0,
                    stepmax = 100,
                    threshold = 0.01 )
print(nn_clf_fit1)

# Predict on test data
nn_clf_predict1 <- predict(nn_clf_fit1,z_test)

# Print Confusion matrix, Accuarcy, Sensitivity etc 
confusionMatrix(nn_clf_predict1,  y1_test)

# Add results into clf_results dataframe
x8 <- confusionMatrix(nn_clf_predict1,  y1_test, positive = '1')[["overall"]]
y8 <- confusionMatrix(nn_clf_predict1,  y1_test, positive = '1')[["byClass"]]

clf_results1[nrow(clf_results1) + 1,] <-  list(Model = "Neural Network", 
                                             Accuracy = round (x8[["Accuracy"]],3), 
                                            Precision = round (y8[["Precision"]],3), 
                                            Recall = round (y8[["Recall"]],3), 
                                            F1 = round (y8[["F1"]],3))

# Print Accuracy and F1 score
cat("Accuarcy is ", round(x8[["Accuracy"]],3), "and F1 is ", round (y8[["F1"]],3)  )

# Add results into cost_benefit_df dataframe for cost benefit analysis 
a8 <- confusionMatrix(knnPredict1, y_test)

cost_benefit_df1[nrow(cost_benefit_df1) + 1,] <-  list(Model = "NN Delta", 
                                             TP = a8[["table"]][1], 
                                             FN = a8[["table"]][2], 
                                             FP = a8[["table"]][3], 
                                             TN = a8[["table"]][4])





print(clf_results)

# Plot accuracy for all the Classification Models

ggplot(clf_results %>% arrange(desc(Accuracy)) %>%
       mutate(Model=factor(Model, levels=Model) ), 
       aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity" , width=0.3, fill="steelblue") + 
  coord_cartesian(ylim = c(0.88, 1)) +
  geom_hline(aes(yintercept = mean(Accuracy)),
             colour = "green",linetype="dashed") +
  ggtitle("Compare Accuracy for all Models - Entire Dataset") +
  theme(plot.title = element_text(color="black", size=10, hjust = 0.5))

print(clf_results1)

# Plot accuracy for all the Classification Models

ggplot(clf_results1 %>% arrange(desc(Accuracy)) %>%
       mutate(Model=factor(Model, levels=Model) ), 
       aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity" , width=0.3, fill="steelblue") + 
  coord_cartesian(ylim = c(0.88, 1)) +
  geom_hline(aes(yintercept = mean(Accuracy)),
             colour = "green",linetype="dashed") +
  ggtitle("Compare Accuracy for all Models - Delta") +
  theme(plot.title = element_text(color="black", size=10, hjust = 0.5))



# Predict probabilities of each model to plot ROC curve
knnPredict_prob <- predict(knn_clf_fit, newdata = x_test, type = "prob") 
XG_boost_prob <- predict(XG_clf_fit, newdata = x_test, type = "prob")
nn_clf_prob <- predict(nn_clf_fit, newdata = x_test, type = "prob")

# List of predictions
preds_list <- list(knnPredict_prob[,1],  
                    XG_boost_prob[,1], nn_clf_prob[,1] )

# List of actual values (same for all)
m <- length(preds_list)
actuals_list <- rep(list(y_test), m)

# Plot the ROC curves
pred <- prediction(preds_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")

# calculate AUC for all models
AUC_models <- performance(pred, "auc")
auc_knn = round(AUC_models@y.values[[1]], 3)
auc_xg = round(AUC_models@y.values[[2]], 3)
auc_nn = round(AUC_models@y.values[[3]], 3)

# Plot the ROC curves
plot(rocs, col = as.list(1:m), main = "ROC Curves of different models - entire dataset")
legend(x = "bottomright", 
       legend = c( paste0("KNN - ", auc_knn), 
                   paste0("XG Boost - ", auc_xg), 
                   paste0("Neural Net - ", auc_nn)), fill = 1:m)

# Predict probabilities of each model to plot ROC curve
knnPredict_prob1 <- predict(knn_clf_fit1, newdata = z_test, type = "prob") 
XG_boost_prob1 <- predict(XG_clf_fit1, newdata = z_test, type = "prob")
nn_clf_prob1 <- predict(nn_clf_fit1, newdata = z_test, type = "prob")

# List of predictions
preds_list1 <- list(knnPredict_prob1[,1],  
                    XG_boost_prob1[,1], nn_clf_prob1[,1] )

# List of actual values (same for all)
m1 <- length(preds_list1)
actuals_list1 <- rep(list(y1_test), m)

# Plot the ROC curves
pred1 <- prediction(preds_list1, actuals_list1)
rocs1 <- performance(pred1, "tpr", "fpr")

# calculate AUC for all models
AUC_models1 <- performance(pred1, "auc")
auc_knn1 = round(AUC_models@y.values[[1]], 3)
auc_xg1 = round(AUC_models@y.values[[2]], 3)
auc_nn1 = round(AUC_models@y.values[[3]], 3)

# Plot the ROC curves
plot(rocs1, col = as.list(1:m), main = "ROC Curves of different models - delta")
legend(x = "bottomright", 
       legend = c( paste0("KNN - ", auc_knn1), 
                   paste0("XG Boost - ", auc_xg1), 
                   paste0("Neural Net - ", auc_nn1)), fill = 1:m)





benefit_TP = 10000
benefit_TN = 10
cost_FN = -8000
cost_FP = -100

cost_benefit_df <- cost_benefit_df %>% 
                    mutate(Profit = (benefit_TP * TP) + (benefit_TN * TN) + 
                                    (cost_FP * FP) + (cost_FN * FN))

cost_benefit_df1 <- cost_benefit_df1 %>% 
                    mutate(Profit = (benefit_TP * TP) + (benefit_TN * TN) + 
                                    (cost_FP * FP) + (cost_FN * FN))

print(cost_benefit_df)



print(cost_benefit_df1)



new = read_csv('/Users/mhlaghari/Downloads/HN_data_PostModule2.csv')

# create Y and X data frames
new.y = data %>% pull("adopter") %>% as.factor()
# exclude X1 since its a row number
new.x = data %>% select(-c("net_user", "male", "adopter"))

#Dealing with missing values
impute <- mice(new.x, m=3, method = 'pmm')
new.x <- complete(impute, 2)

#Feature Scaling
new.x_scaled <- as.data.frame(lapply(new.x, scale))

#Trying the NN predict on the unseen dataset
nn_clf_prob.new <- predict(nn_clf_fit, newdata = new.x_scaled, type = "prob")

new$premium <- nn_clf_prob.new[,1]

new <- new %>% arrange(desc(premium))

successNum <-  new[1:1000,] %>% filter(premium < 0.5) %>% tally()

write.csv(x=new, file = "HNtargetList.csv", row.names = TRUE)

HNTarget <- read.csv('HNtargetList.csv')


#List of top 1000 customers marketing should free trial to.
HNTarget


