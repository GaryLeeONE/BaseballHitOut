require(xgboost)
require(dplyr)
require(Matrix)
require(caret)
require(doParallel)
require(mltest)

simplify = FALSE
only_spray_angle = FALSE

if ("data" %in% objects()) {
    rm(data)
}

if (!"data" %in% objects()) {
    data <- as_tibble(read.csv("data_hitout.csv"))
    data <- data %>% mutate_if( is.character, as.factor ) %>%
        mutate_if( is.integer, as.double )
    
    if (only_spray_angle) {
        data <- data %>% select(-hc_x, -hc_y, -hc_x_new, -hc_y_new)
    }
    
    data$stand <- model.matrix( ~ stand - 1, data = data )
    data$home_park <- model.matrix( ~ home_park - 1, data = data )
    data$if_fielding_alignment <- model.matrix( ~ if_fielding_alignment - 1,
                                                data = data )
    data$of_fielding_alignment <- model.matrix( ~ of_fielding_alignment - 1,
                                                data = data )
    
    if (simplify) {
        data <- data %>% select(-events) %>%
            rename(events = events_simple)
    } else {
        data <- data %>% select(-events_simple)
    }
    
    seed <- 2
    set.seed(seed)
    
    train_ind <- sample( 1:nrow(data), floor(0.8*nrow(data)))
    train <- data[train_ind,]
    test <- data[-train_ind,]
    
    trainXSMatrix <- sparse.model.matrix( events~.-1, data = train )
    testXSMatrix <- sparse.model.matrix( events~.-1, data = test )
    
    train.y <- train$events # extract response from training set
    test.y  <- test$events  # extract response from test set
    
    trainYvec <- as.integer(train.y) -1    # extract response from training set; class label starts from 0
    testYvec  <- as.integer(test.y) -1     # extract response from test set; class label starts from 0
    numberOfClasses <- max(trainYvec) + 1
    
}

cl <- makePSOCKcluster(5)
registerDoParallel(cl)

ctrl <- trainControl(method="cv") #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(events ~ ., data = train,
                method = "knn", trControl = ctrl,
                preProcess = c("center","scale"),
                tuneGrid = expand.grid(k = seq(from = 3, to = 12, by = 1)))

## When you are done:
stopCluster(cl)

knnpred <- predict(knnFit, test)
knneval <- ml_test(knnpred, test.y)

