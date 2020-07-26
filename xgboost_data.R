require(xgboost)
require(dplyr)
require(Matrix)
require(caret)

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

eta <- 0.02
nround <- 10000 # number of rounds/trees

searchGrid <- expand.grid(subsample = c(0.8),          # Already set
                          colsample_bytree = c(0.65),  # Already set
                          gamma = c(0.9),   # Already set
                          max_depth = c(5), #    Already set
                          min_child_weight = c(3),   # Already set
                          lambda = 10) # Already set

ptm <- proc.time()

hyperparams <- apply(searchGrid, 1, function(st){
    current_subsample <- st[["subsample"]]
    current_colsample_bytree <- st[["colsample_bytree"]]
    current_gamma <- st[["gamma"]]
    current_max_depth <- st[["max_depth"]]
    current_min_child_weight <- st[["min_child_weight"]]
    current_lambda <- st[["lambda"]]
    
    if (simplify) {
        param <- list("objective" = "binary:logistic",
                      "eval_metric" = "error",
                      "eta" = eta,
                      "gamma" = current_gamma,
                      "max_depth" = current_max_depth,
                      "min_child_weight" = current_min_child_weight,
                      "subsample" = current_subsample,
                      "colsample_bytree" = current_colsample_bytree,
                      "lambda" = current_lambda,
                      "tree_method" = 'gpu_hist')
    } else {
        param <- list("objective" = "multi:softmax",
                      "eval_metric" = "mlogloss",
                      "num_class" = numberOfClasses,
                      "eta" = eta,
                      "gamma" = current_gamma,
                      "max_depth" = current_max_depth,
                      "min_child_weight" = current_min_child_weight,
                      "subsample" = current_subsample,
                      "colsample_bytree" = current_colsample_bytree,
                      "lambda" = current_lambda,
                      "tree_method" = 'gpu_hist')
    }
    
    
    xgbcv <- xgb.cv( params = param, data = trainXSMatrix, label = trainYvec,
                     nrounds = nround, nfold = 5, showsd = TRUE,
                     early_stopping_rounds = 50L, 
                     stratified = TRUE, print_every_n = 25L )
    
    xvalidationScores <- as.data.frame(xgbcv$evaluation_log)
    best_ntree <- xgbcv$best_iteration
    train_error <- xvalidationScores[best_ntree,2]
    test_error <- xvalidationScores[best_ntree,4]
    output <- return(c(train_error, test_error,
                       eta, best_ntree,
                       current_gamma,
                       current_max_depth,
                       current_min_child_weight,
                       current_subsample,
                       current_colsample_bytree,
                       current_lambda))
    
})

print(proc.time() - ptm)

output <- as.data.frame(t(hyperparams))
varnames <- c("TrainMetric", "TestMetric", "eta", "stopping_iter",
              "gamma", "max_depth", "min_child_weight",
              "subsample", "colsample_bytree", "lambda")
names(output) <- varnames




searchGrid <- expand.grid(subsample = c(0.8),          # Already set
                          colsample_bytree = c(0.65),  # Already set
                          gamma = c(0.9),   # Already set
                          max_depth = c(5), #    Already set
                          min_child_weight = c(3),   # Already set
                          lambda = 10) # Already set

param <- list("objective" = "multi:softmax",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses,
              "eta" = 0.02,
              "gamma" = 0.9,
              "max_depth" = 5,
              "min_child_weight" = 3,
              "subsample" = 0.8,
              "colsample_bytree" = 0.65,
              "lambda" = 10,
              "tree_method" = 'gpu_hist')


# print( 1 - min(xgbcv$evaluation_log[,4]) )
# nround <- which.min(unlist(xgbcv$evaluation_log[,4]))
#
# cv_eval <- as.data.frame(xgbcv$evaluation_log)
# train_metric <- cv_eval[,2]
# test_metric <- cv_eval[,4]
#
# ggplot() +
#     geom_path(aes(x = 1:length(train_metric), y = train_metric, color = "r"),
#               size = 1) +
#     geom_path(aes(x = 1:length(test_metric), y = test_metric, color = "b"),
#               size = 1)



ptm <- proc.time()

xgbstree <- xgboost(params = param,
                    data = trainXSMatrix,
                    label = trainYvec,
                    nrounds = 9114, # same number of trees as before
                    print_every_n = 50L)

print(proc.time() - ptm)

if (simplify) {
    xgbstree.pred <- as.integer(predict(xgbstree, testXSMatrix) >= 0.5)   # this is a long vector
} else {
    xgbstree.pred <- predict(xgbstree, testXSMatrix)   # this is a long vector
}

print(xgbstree.acc <- mean(xgbstree.pred == testYvec))   # classification accuracy on test set

imp_mat <- xgb.importance(model = xgbstree)
xgb.plot.importance(importance_matrix = imp_mat[1:10,])
