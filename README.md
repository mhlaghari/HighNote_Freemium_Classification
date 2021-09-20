# HighNote_Freemium_Classification

Situation
You are all familiar with the HighNote freemium monetization case. HighNote is an online community similar to Spotify (and LinkedIn and OkCupid) that allows both free users and premium users to co-exist. You covered this case at a high level using summary statistics in Module 1 with Prof Anindya Ghose. Now you have the chance to build on your knowledge using additional micro data at the individual level. This could potentially be useful to figure out who specifically to target to convert from free to fee (premium).
Complication
A general challenge in Freemium communities is getting people to pay premium when they can also use the service for free (at the expense of seeing ads say or not getting some features). However, free users are not as profitable as premium users because the fight for ad dollars from brands is intense. HighNote would like to use a data-driven approach to try to get more free users to become premium users. They are willing to offer a two month free trial, but they do not know who to give this promotion to. They have hired you as a consultant to help them design a targeting strategy.
Key Question
Who are the top 1000 most likely free users to convert to premium?
Data
See the data dictionary first and then see the CSV file that you will use. The dataset has a variable called adopter which is 1 if the user adopts the premium level and is 0 if the user is a free user. It has 2 categories of X variables about these users, how their behaviors and engagement changed in the previous time period and what their levels are in the current period.
Your task
1. Make sure you split your data into 75% for training and validation and 25% as a test set. You have to identify the top 1000 from the test set. Do not use any part of the test set for model building, hyper parameter tuning or model selection.
2. You will use the predictive modeling process we learnt in the class to land on a final binary classification model to use.
a. Start by building a model using only the previous time period’s change variables b. See if you get better classification performance by adding the current period
variables.
3. Use your final best predictive model to determine a list of top 1000 currently free (adopter = 0) users from your (entire) data set and their probability of conversion. Explain, your rationale for choosing the model you use in the end.
Extra credit (worth up to 3 extra % points)
How can we make sure that if we target the people we can compute the correct ROI from the proposed model? Describe your strategy of actually deploying the targeting strategy from the predictive model.


# HighNote Assignment

#### My approach was to first run the models on the entire dataset and then only on the delta.  I did not take 'male' into consideration as the 57% of the values were unknown, hence not being a good determinant.

#### This project was completed on R

# Content
1. Data Preprocessing of the entire dataset
1.1 KNN
1.2 XGBoost
1.3 Neural Network
2. Data Preprocessing of the Delta subset
2.1 KNN
2.2 XGBoost
2.3 Neural Network
3. Comparing Results and Accuracies 
4. Cost-Benefit Analysis
5. Predicint on New Dataset
6. Deployment strategy


```R
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
```

    
    Attaching package: ‘dplyr’
    
    
    The following objects are masked from ‘package:stats’:
    
        filter, lag
    
    
    The following objects are masked from ‘package:base’:
    
        intersect, setdiff, setequal, union
    
    
    Loading required package: lattice
    
    Loading required package: ggplot2
    
    
    Attaching package: ‘class’
    
    
    The following objects are masked from ‘package:FNN’:
    
        knn, knn.cv
    
    
    Loading required package: rpart
    
    Loading required package: Matrix
    
    Loaded glmnet 4.1-2
    
    ── [1mAttaching packages[22m ─────────────────────────────────────── tidyverse 1.3.1 ──
    
    [32m✔[39m [34mtibble [39m 3.1.4     [32m✔[39m [34mpurrr  [39m 0.3.4
    [32m✔[39m [34mtidyr  [39m 1.1.3     [32m✔[39m [34mstringr[39m 1.4.0
    [32m✔[39m [34mreadr  [39m 2.0.1     [32m✔[39m [34mforcats[39m 0.5.1
    
    ── [1mConflicts[22m ────────────────────────────────────────── tidyverse_conflicts() ──
    [31m✖[39m [34mtidyr[39m::[32mexpand()[39m masks [34mMatrix[39m::expand()
    [31m✖[39m [34mdplyr[39m::[32mfilter()[39m masks [34mstats[39m::filter()
    [31m✖[39m [34mdplyr[39m::[32mlag()[39m    masks [34mstats[39m::lag()
    [31m✖[39m [34mpurrr[39m::[32mlift()[39m   masks [34mcaret[39m::lift()
    [31m✖[39m [34mtidyr[39m::[32mpack()[39m   masks [34mMatrix[39m::pack()
    [31m✖[39m [34mtidyr[39m::[32munpack()[39m masks [34mMatrix[39m::unpack()
    
    
    Attaching package: ‘mice’
    
    
    The following object is masked from ‘package:stats’:
    
        filter
    
    
    The following objects are masked from ‘package:base’:
    
        cbind, rbind
    
    


# 1. Data Preprocessing on the entire dataset


```R
data = read_csv('/Users/mhlaghari/Downloads/HN_data_PostModule.csv')

# create Y and X data frames
y = data %>% pull("adopter") %>% as.factor()
# exclude X1 since its a row number
x = data %>% select(-c("net_user", "male", "adopter"))
```

    [1m[1mRows: [1m[22m[34m[34m107213[34m[39m [1m[1mColumns: [1m[22m[34m[34m27[34m[39m
    
    [36m──[39m [1m[1mColumn specification[1m[22m [36m────────────────────────────────────────────────────────[39m
    [1mDelimiter:[22m ","
    [31mchr[39m  (1): net_user
    [32mdbl[39m (26): age, male, friend_cnt, avg_friend_age, avg_friend_male, friend_cou...
    
    
    [36mℹ[39m Use [30m[47m[30m[47m`spec()`[47m[30m[49m[39m to retrieve the full column specification for this data.
    [36mℹ[39m Specify the column types or set [30m[47m[30m[47m`show_col_types = FALSE`[47m[30m[49m[39m to quiet this message.
    



```R
#Dealing with missing values
impute <- mice(x, m=3, method = 'pmm')
x <- complete(impute, 2)
```

    
     iter imp variable
      1   1  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      1   2  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      1   3  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      2   1  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      2   2  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      2   3  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      3   1  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      3   2  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      3   3  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      4   1  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      4   2  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      4   3  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      5   1  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      5   2  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      5   3  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country



```R
#Feature Scaling
x_scaled <- as.data.frame(lapply(x, scale))
```


```R
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
```

# 1.1 KNN


```R
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
```


    k-Nearest Neighbors 
    
    80409 samples
       24 predictor
        2 classes: '0', '1' 
    
    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 2 times) 
    Summary of sample sizes: 72368, 72369, 72368, 72369, 72368, 72368, ... 
    Resampling results across tuning parameters:
    
      k  Accuracy   Kappa     
      1  0.8951672  0.09687286
      2  0.8945391  0.09215514
      3  0.9235160  0.08912072
      4  0.9241254  0.08998894
    
    Accuracy was used to select the optimal model using the largest value.
    The final value used for the model was k = 4.



```R
# Plot accuracies for different k values
plot(knn_clf_fit)

# print the best model
print(knn_clf_fit$finalModel)
```

    4-nearest neighbor model
    Training set outcome distribution:
    
        0     1 
    74997  5412 
    



    
![png](output_10_1.png)
    



```R
# Predict on test data
knnPredict <- predict(knn_clf_fit, newdata = x_test) 
```


```R
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

```


    Confusion Matrix and Statistics
    
              Reference
    Prediction     0     1
             0 24624  1674
             1   379   127
                                              
                   Accuracy : 0.9234          
                     95% CI : (0.9202, 0.9266)
        No Information Rate : 0.9328          
        P-Value [Acc > NIR] : 1               
                                              
                      Kappa : 0.0831          
                                              
     Mcnemar's Test P-Value : <2e-16          
                                              
                Sensitivity : 0.98484         
                Specificity : 0.07052         
             Pos Pred Value : 0.93634         
             Neg Pred Value : 0.25099         
                 Prevalence : 0.93281         
             Detection Rate : 0.91867         
       Detection Prevalence : 0.98112         
          Balanced Accuracy : 0.52768         
                                              
           'Positive' Class : 0               
                                              


    Accuarcy is  0.923 and F1 is  0.11


```R

```

# 1.2 XGBoost


```R
XG_clf_fit <- train(x_train, 
                    y_train,
                    method = "xgbTree",
                    preProc = c("center", "scale"))
```


```R
# print the final model
XG_clf_fit$finalModel
```


    ##### xgb.Booster
    raw: 56.5 Kb 
    call:
      xgboost::xgb.train(params = list(eta = param$eta, max_depth = param$max_depth, 
        gamma = param$gamma, colsample_bytree = param$colsample_bytree, 
        min_child_weight = param$min_child_weight, subsample = param$subsample), 
        data = x, nrounds = param$nrounds, objective = "binary:logistic")
    params (as set within xgb.train):
      eta = "0.3", max_depth = "1", gamma = "0", colsample_bytree = "0.8", min_child_weight = "1", subsample = "1", objective = "binary:logistic", validate_parameters = "TRUE"
    xgb.attributes:
      niter
    callbacks:
      cb.print.evaluation(period = print_every_n)
    # of features: 24 
    niter: 100
    nfeatures : 24 
    xNames : age friend_cnt avg_friend_age avg_friend_male friend_country_cnt subscriber_friend_cnt songsListened lovedTracks posts playlists shouts tenure good_country delta1_friend_cnt delta1_avg_friend_age delta1_avg_friend_male delta1_friend_country_cnt delta1_subscriber_friend_cnt delta1_songsListened delta1_lovedTracks delta1_posts delta1_playlists delta1_shouts delta1_good_country 
    problemType : Classification 
    tuneValue :
    	   nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
    17     100         1 0.3     0              0.8                1         1
    obsLevels : 0 1 
    param :
    	list()



```R
# Predict on test data
XG_clf_predict <- predict(XG_clf_fit,x_test)
```


```R
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
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction     0     1
             0 24954  1764
             1    49    37
                                              
                   Accuracy : 0.9324          
                     95% CI : (0.9293, 0.9353)
        No Information Rate : 0.9328          
        P-Value [Acc > NIR] : 0.621           
                                              
                      Kappa : 0.0333          
                                              
     Mcnemar's Test P-Value : <2e-16          
                                              
                Sensitivity : 0.99804         
                Specificity : 0.02054         
             Pos Pred Value : 0.93398         
             Neg Pred Value : 0.43023         
                 Prevalence : 0.93281         
             Detection Rate : 0.93098         
       Detection Prevalence : 0.99679         
          Balanced Accuracy : 0.50929         
                                              
           'Positive' Class : 0               
                                              


    Accuarcy is  0.932 and F1 is  0.039

# 1.3 Neural Network


```R
nn_clf_fit <- train(x_train,
                    y_train,
                    method = "nnet",
                    trace = F,
                    #tuneGrid = my.grid,
                    linout = 0,
                    stepmax = 100,
                    threshold = 0.01 )
print(nn_clf_fit)
```

    Neural Network 
    
    80409 samples
       24 predictor
        2 classes: '0', '1' 
    
    No pre-processing
    Resampling: Bootstrapped (25 reps) 
    Summary of sample sizes: 80409, 80409, 80409, 80409, 80409, 80409, ... 
    Resampling results across tuning parameters:
    
      size  decay  Accuracy   Kappa       
      1     0e+00  0.9328202  0.000000e+00
      1     1e-04  0.9328202  0.000000e+00
      1     1e-01  0.9328202  0.000000e+00
      3     0e+00  0.9327731  2.808841e-03
      3     1e-04  0.9327959  9.081037e-05
      3     1e-01  0.9327946  1.505942e-03
      5     0e+00  0.9326848  6.641195e-03
      5     1e-04  0.9326224  1.548542e-02
      5     1e-01  0.9326759  7.810493e-03
    
    Accuracy was used to select the optimal model using the largest value.
    The final values used for the model were size = 1 and decay = 0.1.



```R
# Predict on test data
nn_clf_predict <- predict(nn_clf_fit,x_test)
```


```R
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

```


    Confusion Matrix and Statistics
    
              Reference
    Prediction     0     1
             0 25003  1801
             1     0     0
                                              
                   Accuracy : 0.9328          
                     95% CI : (0.9297, 0.9358)
        No Information Rate : 0.9328          
        P-Value [Acc > NIR] : 0.5063          
                                              
                      Kappa : 0               
                                              
     Mcnemar's Test P-Value : <2e-16          
                                              
                Sensitivity : 1.0000          
                Specificity : 0.0000          
             Pos Pred Value : 0.9328          
             Neg Pred Value :    NaN          
                 Prevalence : 0.9328          
             Detection Rate : 0.9328          
       Detection Prevalence : 1.0000          
          Balanced Accuracy : 0.5000          
                                              
           'Positive' Class : 0               
                                              


    Accuarcy is  0.933 and F1 is  NA


```R

```


```R
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
```


    
![png](output_24_0.png)
    


# 2. Data Preprocessing - Delta Subset


```R
#Creating delta subset
z <- x %>% select(-c("friend_cnt","avg_friend_age","avg_friend_male",
                             "friend_country_cnt"          
,"subscriber_friend_cnt","songsListened","lovedTracks","posts",
"playlists","shouts","good_country"))
```


```R
#Feature Scaling
z_scaled <- as.data.frame(lapply(z, scale))
```


```R
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
```

# 2.1 KNN


```R
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
```


    k-Nearest Neighbors 
    
    80409 samples
       13 predictor
        2 classes: '0', '1' 
    
    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 2 times) 
    Summary of sample sizes: 72368, 72368, 72368, 72368, 72368, 72368, ... 
    Resampling results across tuning parameters:
    
      k  Accuracy   Kappa     
      1  0.8965663  0.09226030
      2  0.8988235  0.08759888
      3  0.9226579  0.06262724
      4  0.9226455  0.05933087
    
    Accuracy was used to select the optimal model using the largest value.
    The final value used for the model was k = 3.



```R
# Plot accuracies for different k values
plot(knn_clf_fit1)

# print the best model
print(knn_clf_fit1$finalModel)
```

    3-nearest neighbor model
    Training set outcome distribution:
    
        0     1 
    75050  5359 
    



    
![png](output_31_1.png)
    



```R
# Predict on test data
knnPredict1 <- predict(knn_clf_fit1, newdata = z_test) 
```


```R
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
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction     0     1
             0 24499  1721
             1   451   133
                                              
                   Accuracy : 0.919           
                     95% CI : (0.9156, 0.9222)
        No Information Rate : 0.9308          
        P-Value [Acc > NIR] : 1               
                                              
                      Kappa : 0.0786          
                                              
     Mcnemar's Test P-Value : <2e-16          
                                              
                Sensitivity : 0.98192         
                Specificity : 0.07174         
             Pos Pred Value : 0.93436         
             Neg Pred Value : 0.22774         
                 Prevalence : 0.93083         
             Detection Rate : 0.91401         
       Detection Prevalence : 0.97821         
          Balanced Accuracy : 0.52683         
                                              
           'Positive' Class : 0               
                                              


    Accuarcy is  0.919 and F1 is  0.109

# 2.2 XGBoost


```R
XG_clf_fit1 <- train(z_train, 
                    y1_train,
                    method = "xgbTree",
                    preProc = c("center", "scale"))
```


```R
# print the final model
XG_clf_fit1$finalModel
```


    ##### xgb.Booster
    raw: 29.5 Kb 
    call:
      xgboost::xgb.train(params = list(eta = param$eta, max_depth = param$max_depth, 
        gamma = param$gamma, colsample_bytree = param$colsample_bytree, 
        min_child_weight = param$min_child_weight, subsample = param$subsample), 
        data = x, nrounds = param$nrounds, objective = "binary:logistic")
    params (as set within xgb.train):
      eta = "0.3", max_depth = "1", gamma = "0", colsample_bytree = "0.6", min_child_weight = "1", subsample = "1", objective = "binary:logistic", validate_parameters = "TRUE"
    xgb.attributes:
      niter
    callbacks:
      cb.print.evaluation(period = print_every_n)
    # of features: 13 
    niter: 50
    nfeatures : 13 
    xNames : age tenure delta1_friend_cnt delta1_avg_friend_age delta1_avg_friend_male delta1_friend_country_cnt delta1_subscriber_friend_cnt delta1_songsListened delta1_lovedTracks delta1_posts delta1_playlists delta1_shouts delta1_good_country 
    problemType : Classification 
    tuneValue :
    	  nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
    7      50         1 0.3     0              0.6                1         1
    obsLevels : 0 1 
    param :
    	list()



```R
# Predict on test data
XG_clf_predict1 <- predict(XG_clf_fit1,z_test)
```


```R
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
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction     0     1
             0 24942  1846
             1     8     8
                                              
                   Accuracy : 0.9308          
                     95% CI : (0.9277, 0.9338)
        No Information Rate : 0.9308          
        P-Value [Acc > NIR] : 0.5062          
                                              
                      Kappa : 0.0074          
                                              
     Mcnemar's Test P-Value : <2e-16          
                                              
                Sensitivity : 0.999679        
                Specificity : 0.004315        
             Pos Pred Value : 0.931089        
             Neg Pred Value : 0.500000        
                 Prevalence : 0.930831        
             Detection Rate : 0.930533        
       Detection Prevalence : 0.999403        
          Balanced Accuracy : 0.501997        
                                              
           'Positive' Class : 0               
                                              


    Accuarcy is  0.931 and F1 is  0.009


```R

```

# 2.3 Neural Network


```R
nn_clf_fit1 <- train(z_train,
                    y1_train,
                    method = "nnet",
                    trace = F,
                    #tuneGrid = my.grid,
                    linout = 0,
                    stepmax = 100,
                    threshold = 0.01 )
print(nn_clf_fit1)
```

    Neural Network 
    
    80409 samples
       13 predictor
        2 classes: '0', '1' 
    
    No pre-processing
    Resampling: Bootstrapped (25 reps) 
    Summary of sample sizes: 80409, 80409, 80409, 80409, 80409, 80409, ... 
    Resampling results across tuning parameters:
    
      size  decay  Accuracy   Kappa      
      1     0e+00  0.9337343  0.000000000
      1     1e-04  0.9337343  0.000000000
      1     1e-01  0.9337343  0.000000000
      3     0e+00  0.9336600  0.001625514
      3     1e-04  0.9335963  0.003796600
      3     1e-01  0.9334404  0.008715229
      5     0e+00  0.9334882  0.007101212
      5     1e-04  0.9334621  0.006626140
      5     1e-01  0.9334273  0.011800504
    
    Accuracy was used to select the optimal model using the largest value.
    The final values used for the model were size = 1 and decay = 0.1.



```R
# Predict on test data
nn_clf_predict1 <- predict(nn_clf_fit1,z_test)
```


```R
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
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction     0     1
             0 24950  1854
             1     0     0
                                              
                   Accuracy : 0.9308          
                     95% CI : (0.9277, 0.9338)
        No Information Rate : 0.9308          
        P-Value [Acc > NIR] : 0.5062          
                                              
                      Kappa : 0               
                                              
     Mcnemar's Test P-Value : <2e-16          
                                              
                Sensitivity : 1.0000          
                Specificity : 0.0000          
             Pos Pred Value : 0.9308          
             Neg Pred Value :    NaN          
                 Prevalence : 0.9308          
             Detection Rate : 0.9308          
       Detection Prevalence : 1.0000          
          Balanced Accuracy : 0.5000          
                                              
           'Positive' Class : 0               
                                              


    Accuarcy is  0.931 and F1 is  NA


```R

```


```R

```

# 3. Compairing model accuracies


```R
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
```

                              Model Accuracy Precision Recall    F1
    1            KNN Entire Dataset    0.923     0.251  0.071 0.110
    2       XG Boost Entire Dataset    0.932     0.430  0.021 0.039
    3 Neural Network Entire Dataset    0.933        NA  0.000    NA



    
![png](output_47_1.png)
    



```R
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
```

               Model Accuracy Precision Recall    F1
    1      KNN Delta    0.919     0.228  0.072 0.109
    2 XG Boost Delta    0.931     0.500  0.004 0.009
    3 Neural Network    0.931        NA  0.000    NA



    
![png](output_48_1.png)
    



```R

```


```R
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
```


    
![png](output_50_0.png)
    



```R
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
```


    
![png](output_51_0.png)
    



```R

```


```R

```

# 4. Cost Benefit Analysis


```R
benefit_TP = 10000
benefit_TN = 10
cost_FN = -8000
cost_FP = -100

cost_benefit_df <- cost_benefit_df %>% 
                    mutate(Profit = (benefit_TP * TP) + (benefit_TN * TN) + 
                                    (cost_FP * FP) + (cost_FN * FN))
```


```R
cost_benefit_df1 <- cost_benefit_df1 %>% 
                    mutate(Profit = (benefit_TP * TP) + (benefit_TN * TN) + 
                                    (cost_FP * FP) + (cost_FN * FN))
```


```R
print(cost_benefit_df)


```

                              Model    TP  FN   FP  TN    Profit
    1            KNN Entire Dataset 24624 379 1674 127 243041870
    2        XGBoost Entire Dataset 24954  49 1764  37 248971970
    3 Neural Network Entire Dataset 25003   0 1801   0 249849900



```R
print(cost_benefit_df1)


```

               Model    TP  FN   FP  TN    Profit
    1      KNN Delta 24548 455 1672 129 241674090
    2 XG Boost Delta 24548 455 1672 129 241674090
    3      KNN Delta 24548 455 1672 129 241674090
    4       NN Delta 24548 455 1672 129 241674090


# 5. Predicting on New Data


```R
new = read_csv('/Users/mhlaghari/Downloads/HN_data_PostModule2.csv')

# create Y and X data frames
new.y = data %>% pull("adopter") %>% as.factor()
# exclude X1 since its a row number
new.x = data %>% select(-c("net_user", "male", "adopter"))
```

    [1m[1mRows: [1m[22m[34m[34m107213[34m[39m [1m[1mColumns: [1m[22m[34m[34m27[34m[39m
    
    [36m──[39m [1m[1mColumn specification[1m[22m [36m────────────────────────────────────────────────────────[39m
    [1mDelimiter:[22m ","
    [31mchr[39m  (1): net_user
    [32mdbl[39m (25): age, male, friend_cnt, avg_friend_age, avg_friend_male, friend_cou...
    [33mlgl[39m  (1): adopter
    
    
    [36mℹ[39m Use [30m[47m[30m[47m`spec()`[47m[30m[49m[39m to retrieve the full column specification for this data.
    [36mℹ[39m Specify the column types or set [30m[47m[30m[47m`show_col_types = FALSE`[47m[30m[49m[39m to quiet this message.
    



```R
#Dealing with missing values
impute <- mice(new.x, m=3, method = 'pmm')
new.x <- complete(impute, 2)
```

    
     iter imp variable
      1   1  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      1   2  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      1   3  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      2   1  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      2   2  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      2   3  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      3   1  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      3   2  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      3   3  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      4   1  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      4   2  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      4   3  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      5   1  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      5   2  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country
      5   3  age  friend_cnt  avg_friend_age  avg_friend_male  friend_country_cnt  subscriber_friend_cnt  shouts  tenure  good_country  delta1_friend_cnt  delta1_avg_friend_age  delta1_avg_friend_male  delta1_friend_country_cnt  delta1_subscriber_friend_cnt  delta1_songsListened  delta1_lovedTracks  delta1_posts  delta1_playlists  delta1_shouts  delta1_good_country



```R
#Feature Scaling
new.x_scaled <- as.data.frame(lapply(new.x, scale))
```


```R
#Trying the NN predict on the unseen dataset
nn_clf_prob.new <- predict(nn_clf_fit, newdata = new.x_scaled, type = "prob")
```


```R
new$premium <- nn_clf_prob.new[,1]
```


```R
new <- new %>% arrange(desc(premium))
```


```R
successNum <-  new[1:1000,] %>% filter(premium < 0.5) %>% tally()
```


```R
write.csv(x=new, file = "HNtargetList.csv", row.names = TRUE)
```


```R
HNTarget <- read.csv('HNtargetList.csv')

```


```R
#List of top 1000 customers marketing should free trial to.
HNTarget
```


<table class="dataframe">
<caption>A data.frame: 107213 × 29</caption>
<thead>
	<tr><th scope=col>X</th><th scope=col>net_user</th><th scope=col>age</th><th scope=col>male</th><th scope=col>friend_cnt</th><th scope=col>avg_friend_age</th><th scope=col>avg_friend_male</th><th scope=col>friend_country_cnt</th><th scope=col>subscriber_friend_cnt</th><th scope=col>songsListened</th><th scope=col>⋯</th><th scope=col>delta1_friend_country_cnt</th><th scope=col>delta1_subscriber_friend_cnt</th><th scope=col>delta1_songsListened</th><th scope=col>delta1_lovedTracks</th><th scope=col>delta1_posts</th><th scope=col>delta1_playlists</th><th scope=col>delta1_shouts</th><th scope=col>delta1_good_country</th><th scope=col>adopter</th><th scope=col>premium</th></tr>
	<tr><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>⋯</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;lgl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td> 1</td><td>nexus6868     </td><td>41</td><td> 1</td><td> 2</td><td>50.00000</td><td>1.0000000</td><td>1</td><td>0</td><td> 6677</td><td>⋯</td><td>0</td><td>-1</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td> 2</td><td>jibrohni      </td><td>NA</td><td>NA</td><td> 0</td><td>      NA</td><td>       NA</td><td>0</td><td>0</td><td>  483</td><td>⋯</td><td>0</td><td> 0</td><td> 25</td><td>15</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td> 3</td><td>chillingraven </td><td>26</td><td> 1</td><td>29</td><td>24.78261</td><td>0.6666667</td><td>8</td><td>0</td><td>23743</td><td>⋯</td><td>0</td><td>-1</td><td>959</td><td> 0</td><td>0</td><td>0</td><td> 2</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td> 4</td><td>euseo         </td><td>29</td><td> 0</td><td> 5</td><td>25.60000</td><td>0.8000000</td><td>2</td><td>0</td><td>    4</td><td>⋯</td><td>1</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td> 5</td><td>fuxer         </td><td>NA</td><td> 1</td><td> 1</td><td>25.00000</td><td>1.0000000</td><td>1</td><td>0</td><td>  641</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td> 6</td><td>h4z30420      </td><td>NA</td><td>NA</td><td> 0</td><td>      NA</td><td>       NA</td><td>0</td><td>0</td><td>  734</td><td>⋯</td><td>0</td><td> 0</td><td>100</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td> 7</td><td>whitedynamite2</td><td>NA</td><td>NA</td><td> 0</td><td>      NA</td><td>       NA</td><td>0</td><td>0</td><td>  141</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td> 8</td><td>ourjen        </td><td>27</td><td> 0</td><td> 6</td><td>29.40000</td><td>1.0000000</td><td>2</td><td>0</td><td> 3543</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td> 9</td><td>maddin_87     </td><td>NA</td><td>NA</td><td> 1</td><td>      NA</td><td>       NA</td><td>0</td><td>0</td><td>   24</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>10</td><td>joecardone    </td><td>NA</td><td>NA</td><td> 1</td><td>27.00000</td><td>0.0000000</td><td>1</td><td>0</td><td>    4</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>11</td><td>marcdisorder  </td><td>NA</td><td> 1</td><td> 1</td><td>28.00000</td><td>0.0000000</td><td>1</td><td>0</td><td> 1660</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>12</td><td>drosenqvist   </td><td>34</td><td>NA</td><td> 2</td><td>25.00000</td><td>0.5000000</td><td>2</td><td>0</td><td>19528</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>13</td><td>thatajf       </td><td>21</td><td> 0</td><td> 1</td><td>22.00000</td><td>1.0000000</td><td>1</td><td>0</td><td> 1266</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>14</td><td>vesnoowechka  </td><td>NA</td><td>NA</td><td> 1</td><td>22.00000</td><td>1.0000000</td><td>1</td><td>0</td><td>   19</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>15</td><td>psylocibee    </td><td>21</td><td> 0</td><td> 7</td><td>25.80000</td><td>0.5714286</td><td>1</td><td>0</td><td>  971</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>16</td><td>brunitds      </td><td>16</td><td> 0</td><td> 3</td><td>18.00000</td><td>0.3333333</td><td>1</td><td>0</td><td> 3346</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>17</td><td>trajano_band  </td><td>NA</td><td>NA</td><td> 2</td><td>22.00000</td><td>0.5000000</td><td>1</td><td>0</td><td>    0</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>18</td><td>luna_suia     </td><td>NA</td><td>NA</td><td> 2</td><td>16.00000</td><td>0.0000000</td><td>1</td><td>0</td><td>   49</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>19</td><td>bond_bloodshed</td><td>NA</td><td> 1</td><td> 1</td><td>      NA</td><td>0.0000000</td><td>1</td><td>0</td><td> 5027</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>20</td><td>pilimi        </td><td>NA</td><td>NA</td><td> 1</td><td>19.00000</td><td>1.0000000</td><td>1</td><td>0</td><td>  629</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>21</td><td>lsd-effect    </td><td>NA</td><td>NA</td><td> 1</td><td>23.00000</td><td>0.0000000</td><td>1</td><td>0</td><td>   10</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>22</td><td>suckerlover   </td><td>NA</td><td>NA</td><td> 1</td><td>      NA</td><td>0.0000000</td><td>1</td><td>0</td><td>    0</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>23</td><td>cuomokid10    </td><td>21</td><td> 1</td><td> 2</td><td>23.50000</td><td>0.0000000</td><td>1</td><td>0</td><td>   55</td><td>⋯</td><td>0</td><td> 0</td><td> 50</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>24</td><td>ellieeatworld </td><td>17</td><td> 0</td><td> 5</td><td>20.80000</td><td>0.8000000</td><td>3</td><td>0</td><td> 7368</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>25</td><td>jayder        </td><td>NA</td><td>NA</td><td> 2</td><td>26.00000</td><td>0.5000000</td><td>1</td><td>0</td><td> 2359</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>26</td><td>pstitt        </td><td>NA</td><td>NA</td><td> 1</td><td>      NA</td><td>0.0000000</td><td>1</td><td>0</td><td>  105</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>27</td><td>neugen        </td><td>25</td><td> 1</td><td> 8</td><td>25.71429</td><td>0.7500000</td><td>1</td><td>0</td><td> 3380</td><td>⋯</td><td>0</td><td> 0</td><td> 10</td><td> 0</td><td>0</td><td>0</td><td>NA</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>28</td><td>madamemunni   </td><td>NA</td><td> 0</td><td>22</td><td>18.50000</td><td>0.5000000</td><td>1</td><td>0</td><td>  361</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td>NA</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>29</td><td>benj-amin     </td><td>22</td><td> 1</td><td> 0</td><td>      NA</td><td>       NA</td><td>0</td><td>0</td><td> 4268</td><td>⋯</td><td>0</td><td> 0</td><td>414</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>30</td><td>aline95       </td><td>16</td><td> 0</td><td> 1</td><td>17.00000</td><td>1.0000000</td><td>1</td><td>0</td><td>    0</td><td>⋯</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>0</td><td>0</td><td> 0</td><td> 0</td><td>NA</td><td>0.9972757</td></tr>
	<tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋱</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><td>107184</td><td>-saosin-       </td><td>21</td><td> 1</td><td> 10</td><td>21.88889</td><td>0.6666667</td><td> 2</td><td> 0</td><td>   64</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107185</td><td>dramaqueen69   </td><td>NA</td><td>NA</td><td>  1</td><td>      NA</td><td>0.0000000</td><td> 1</td><td> 0</td><td>    0</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td>NA</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107186</td><td>aspaceoddity   </td><td>NA</td><td>NA</td><td>  1</td><td>      NA</td><td>0.0000000</td><td> 0</td><td> 0</td><td>  107</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td>NA</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107187</td><td>melonhead20    </td><td>NA</td><td> 0</td><td>  2</td><td>      NA</td><td>0.0000000</td><td> 2</td><td> 0</td><td>  168</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107188</td><td>aliisanalien   </td><td>21</td><td> 0</td><td>  1</td><td>21.00000</td><td>0.0000000</td><td> 1</td><td> 0</td><td>  660</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107189</td><td>natedawg199    </td><td>19</td><td> 1</td><td>  7</td><td>21.80000</td><td>0.1666667</td><td> 1</td><td> 0</td><td>    0</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107190</td><td>halvarn        </td><td>26</td><td> 1</td><td>  8</td><td>26.50000</td><td>1.0000000</td><td> 1</td><td> 0</td><td>14219</td><td>⋯</td><td> 0</td><td> 0</td><td>   560</td><td> -9</td><td>0</td><td>-1</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107191</td><td>juliadanielski </td><td>16</td><td> 0</td><td> 26</td><td>17.20000</td><td>0.2400000</td><td> 5</td><td> 0</td><td>  406</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td> -1</td><td>NA</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107192</td><td>goldensaturday </td><td>NA</td><td> 0</td><td> 23</td><td>19.53846</td><td>0.5000000</td><td>11</td><td> 0</td><td> 2731</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107193</td><td>crazyfrenchy91 </td><td>20</td><td> 0</td><td>  2</td><td>19.50000</td><td>0.0000000</td><td> 1</td><td> 0</td><td>    0</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td> NA</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107194</td><td>im_haley       </td><td>20</td><td> 0</td><td> 13</td><td>21.60000</td><td>0.2727273</td><td> 2</td><td> 0</td><td> 2311</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107195</td><td>dawarmalhaar   </td><td>14</td><td> 1</td><td>  6</td><td>14.60000</td><td>0.2000000</td><td> 1</td><td> 0</td><td> 1068</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107196</td><td>samstevenson   </td><td>13</td><td>NA</td><td>  3</td><td>23.00000</td><td>1.0000000</td><td> 2</td><td> 0</td><td>13273</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107197</td><td>lil88          </td><td>23</td><td> 0</td><td>  2</td><td>      NA</td><td>0.0000000</td><td> 1</td><td> 0</td><td>   29</td><td>⋯</td><td> 0</td><td> 0</td><td> -2015</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107198</td><td>niggs-de       </td><td>25</td><td> 1</td><td>  3</td><td>28.00000</td><td>1.0000000</td><td> 1</td><td> 0</td><td> 1155</td><td>⋯</td><td> 0</td><td> 0</td><td>    11</td><td> -1</td><td>0</td><td>-1</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107199</td><td>generatorevil  </td><td>NA</td><td>NA</td><td>  1</td><td>20.00000</td><td>0.0000000</td><td> 1</td><td> 0</td><td>    0</td><td>⋯</td><td>-1</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td>NA</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107200</td><td>red_bullet     </td><td>21</td><td> 0</td><td> 63</td><td>23.94231</td><td>0.8333333</td><td>23</td><td> 0</td><td>28218</td><td>⋯</td><td>-1</td><td> 0</td><td>   204</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107201</td><td>the_person     </td><td>20</td><td> 0</td><td>  1</td><td>19.00000</td><td>0.0000000</td><td> 1</td><td> 0</td><td>    0</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107202</td><td>yoake.no.senshi</td><td>25</td><td>NA</td><td> 28</td><td>26.04348</td><td>0.7083333</td><td>11</td><td> 0</td><td>  379</td><td>⋯</td><td> 0</td><td> 0</td><td>     0</td><td>  0</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107203</td><td>xenia52        </td><td>20</td><td> 0</td><td> 20</td><td>20.20000</td><td>0.5000000</td><td> 9</td><td> 0</td><td>42786</td><td>⋯</td><td> 0</td><td> 0</td><td>   714</td><td>-28</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107204</td><td>aurarian       </td><td>22</td><td> 1</td><td> 16</td><td>22.53846</td><td>0.5333333</td><td> 3</td><td> 0</td><td>  796</td><td>⋯</td><td> 0</td><td> 0</td><td> -3505</td><td>  1</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107205</td><td>awsomnes       </td><td>NA</td><td>NA</td><td>  0</td><td>      NA</td><td>       NA</td><td> 0</td><td> 0</td><td>    1</td><td>⋯</td><td>-2</td><td> 0</td><td> -8302</td><td> -2</td><td>0</td><td> 0</td><td>  0</td><td>NA</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107206</td><td>gothicaddams   </td><td>NA</td><td> 0</td><td> 13</td><td>22.44444</td><td>0.4166667</td><td> 5</td><td> 0</td><td> 6173</td><td>⋯</td><td>-1</td><td> 0</td><td>  1139</td><td>-57</td><td>3</td><td> 0</td><td>-47</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107207</td><td>coccomaroco    </td><td>21</td><td> 0</td><td> 24</td><td>20.66667</td><td>0.3636364</td><td> 2</td><td> 0</td><td>11933</td><td>⋯</td><td> 0</td><td> 0</td><td>  3859</td><td>-77</td><td>0</td><td> 0</td><td>  1</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107208</td><td>jrclark9       </td><td>27</td><td> 1</td><td>  1</td><td>29.00000</td><td>1.0000000</td><td> 0</td><td> 0</td><td>    1</td><td>⋯</td><td> 0</td><td> 0</td><td> -1515</td><td>-65</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107209</td><td>gothman1       </td><td>21</td><td> 1</td><td> 20</td><td>19.92308</td><td>0.4705882</td><td> 5</td><td> 0</td><td>65404</td><td>⋯</td><td> 0</td><td> 0</td><td>-27434</td><td>  0</td><td>0</td><td> 0</td><td> -1</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107210</td><td>morkedcram     </td><td>21</td><td> 1</td><td>  3</td><td>19.00000</td><td>0.0000000</td><td> 2</td><td> 0</td><td>  381</td><td>⋯</td><td> 1</td><td> 0</td><td>-31718</td><td>  8</td><td>0</td><td>-1</td><td> -1</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107211</td><td>muzgus         </td><td>24</td><td> 1</td><td> 38</td><td>23.44828</td><td>0.6969697</td><td> 1</td><td> 1</td><td>  722</td><td>⋯</td><td> 0</td><td>-1</td><td>-48440</td><td> -2</td><td>0</td><td> 0</td><td>  0</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107212</td><td>remco05        </td><td>25</td><td> 1</td><td> 20</td><td>23.43750</td><td>0.9333333</td><td> 6</td><td> 1</td><td> 9996</td><td>⋯</td><td> 0</td><td> 1</td><td>-63968</td><td>  0</td><td>0</td><td>-1</td><td>-99</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
	<tr><td>107213</td><td>tevo-          </td><td>20</td><td> 1</td><td>641</td><td>20.01143</td><td>0.8302207</td><td>48</td><td>13</td><td>12119</td><td>⋯</td><td> 3</td><td> 3</td><td>-93584</td><td>  0</td><td>0</td><td> 1</td><td> 90</td><td> 0</td><td>NA</td><td>0.7514964</td></tr>
</tbody>
</table>



# 6. Deployment

#### To deploy this solution, I would encourage my software developers to either make this a part of an existing marketing solution or create a new program. 
