qsub -I -A open -l nodes=1:ppn=8 -l walltime=6:00:00 -N sessionName=rf -l pmem=40gb
module load gcc
# setwd("D:/DMC 2019")

library(car)
library(readr)
library(lattice)
library(nlme)
library(ggplot2)
library(GGally)
library(nnet)
library(foreign)
library(biotools)
library(glmmML)
library(MASS)
library(lme4)
library(multcomp)
library(dplyr)
library(qwraps2)
library(knitr)
library(xtable)
library(kableExtra)
library(DT)
library(glmnet)
library(h2o)
library(gtools)

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_ncol462.tsv", sep = '\t', header = TRUE)
table(dat$fraud,dat$trustLevel)

table(dat$fraud)

dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$col_weight <- ifelse(dat$fraud==0,25,10)


set.seed(1)
f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
table(dat$fold)

h2o.init(nthreads=-1, max_mem_size="30G")

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)



## Tuning 

best_seednumber = 1234
best_test_profit = 0
best_param = list()

for (iter in 1:20) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)

  hyper_params <- list(
    activation=sample(c("Tanh","TanhWithDropout"),1),
    hidden = sample(list(c(100,100,100,100,100),c(100,100,100),c(100,100,100,100)),1)[[1]],
    input_dropout_ratio=runif(1,0,0.05),
    l1=sample(seq(0,1e-4,1e-6),1),
    l2=sample(seq(0,1e-4,1e-6),1),
    weight_option = sample(list(c(25,10),c(50,20)),1)[[1]])

  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",hyper_params$weight_option[1],
                                   hyper_params$weight_option[2])
  
  h2o.dl.10 <- h2o.deeplearning(x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                y= "fraud", training_frame = x_train.hex,
                                ignore_const_cols =T, seed =  seed.number, reproducible = T,
                                loss = "CrossEntropy",weights_column = "col_weight",
                               fold_column = "fold",
                               initial_weight_distribution = "UniformAdaptive",
                               hidden = hyper_params$hidden, 
                               distribution = "bernoulli",activation = hyper_params$activation,
                                input_dropout_ratio = hyper_params$input_dropout_ratio,
                                l1=hyper_params$l1, l2=hyper_params$l2,
                               keep_cross_validation_predictions = T,
                                missing_values_handling = "MeanImputation")
  
  (t <- table(as.vector(h2o.getFrame(h2o.dl.10@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
             as.vector(x_train.hex$fraud)))
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param


## Good models (Don't touch until you find better one) Full data and initial_weight_distribution = "UniformAdaptive"

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",25,10)
# x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)

h2o.dl.10 <- h2o.deeplearning(seed= 1,
                              x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                              y= "fraud", training_frame = x_train.hex,
                              ignore_const_cols =T, reproducible = T,
                              fold_column = "fold", 
                              weights_column = "col_weight",
                              hidden = c(100,100,100,100,100),loss = "CrossEntropy",
                              distribution = "bernoulli", 
                              initial_weight_distribution = "UniformAdaptive",
                              activation = "Tanh", epochs = 10,
                              input_dropout_ratio = 0.006604091,
                              l1= 9.2e-05, l2= 6e-05,
                              keep_cross_validation_predictions = T,
                              missing_values_handling = "MeanImputation")

t <- table(as.vector(h2o.getFrame(h2o.dl.10@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
           as.vector(x_train.hex$fraud))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  ## 110

cvpreds_id <- h2o.dl.10@model$cross_validation_holdout_predictions_frame_id$name
cvpreds <- h2o.getFrame(cvpreds_id)
h2o_dl_110 <- as.data.frame(cvpreds)
write.csv(h2o_dl_110, "/gpfs/group/asb17/default/Vishal/h2o_dl_110_preds_on_train_largedata.csv")


(as.data.frame(h2o.varimp(h2o.dl.10)))[1:10,]

### Model with profit 110 : going over all training splits

library(h2o)
h2o.init(nthreads=-1, max_mem_size="20G")

Profit = 0;
N= 30;

for (i in 1:N){
  
  dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_ncol462.tsv", sep = '\t', header = TRUE)
  dat$MRN  <- as.character(seq(1,dim(dat)[1]))
  
  f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
  nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))
  df <- floor(length(f)/10)
  dnf <- floor(length(nf)/10)    
  dat$fold <- rep(NA_integer_,dim(dat)[1])
  dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
  dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
  dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
  dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
  dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
  dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
  dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
  dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
  dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
  dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
  
  x_train_deep = as.data.frame(data.matrix(dat))
  x_train.hex <- as.h2o(x_train_deep)
  x_train.hex$fraud <- as.factor(x_train.hex$fraud)
  
  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",25,10)
  
  h2o.dl.10.110 <- h2o.deeplearning(seed= 1,
                                x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                y= "fraud", training_frame = x_train.hex,
                                ignore_const_cols =T, reproducible = T,
                                fold_column = "fold", 
                                weights_column = "col_weight",
                                hidden = c(100,100,100,100,100),loss = "CrossEntropy",
                                distribution = "bernoulli", 
                                initial_weight_distribution = "UniformAdaptive",
                                activation = "Tanh", epochs = 10,
                                input_dropout_ratio = 0.006604091,
                                l1= 9.2e-05, l2= 6e-05,
                                keep_cross_validation_predictions = T,
                                missing_values_handling = "MeanImputation")
  
  t <- table(as.vector(h2o.getFrame(h2o.dl.10.110@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
             as.vector(x_train.hex$fraud))
  
  (Profit <- Profit + -5 * t[1,2]-25*t[2,1]+ 5*t[2,2]  )
  
}

Profit/N;    # -54.54166


############## Reduced (Initial) Data set ##################

dat <- read.table(file = "/gpfs/group/asb17/default/DMC2019/task/train.csv", sep = '|', header = TRUE)
table(dat$fraud,dat$trustLevel)

table(dat$fraud)

dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$col_weight <- ifelse(dat$fraud==0,25,10)

set.seed(1)
f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
table(dat$fold)

h2o.init(nthreads=-1, max_mem_size="30G")

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)

x_train.hex$fraud <- as.factor(x_train.hex$fraud)

x_train.hex$trustLevel <- as.factor(x_train.hex$trustLevel)
# x_train.hex$scansWithoutRegistration <- as.factor(x_train.hex$scansWithoutRegistration)
# x_train.hex$lineItemVoids <- as.factor(x_train.hex$lineItemVoids)
# x_train.hex$quantityModifications <- as.factor(x_train.hex$quantityModifications)


## Tuning

best_seednumber = 1234
best_test_profit = -100
best_param = list()

for (iter in 1:50) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    activation=sample(c("Tanh","TanhWithDropout"),1),
    hidden = sample(list(c(100,100,100,100),c(200,200,200),c(100,100,100,100,100)),1)[[1]],
    input_dropout_ratio=runif(1,0,0.05),
    l1=sample(seq(0,1e-4,1e-6),1),
    l2=sample(seq(0,1e-4,1e-6),1),
    weight_option = sample(list(c(25,10),c(50,20)),1)[[1]])
  
  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",hyper_params$weight_option[1],
                                   hyper_params$weight_option[2])
  
  h2o.dl.10 <- h2o.deeplearning(x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                y= "fraud", training_frame = x_train.hex,
                                ignore_const_cols =T, seed =  seed.number, reproducible = T,
                                loss = "CrossEntropy",weights_column = "col_weight",
                                fold_column = "fold", epochs = 25,
                                initial_weight_distribution = "UniformAdaptive",
                                hidden = hyper_params$hidden, 
                                distribution = "bernoulli",activation = hyper_params$activation,
                                input_dropout_ratio = hyper_params$input_dropout_ratio,
                                l1=hyper_params$l1, l2=hyper_params$l2,
                                keep_cross_validation_predictions = T,
                                missing_values_handling = "MeanImputation")
  
  (t <- table(as.vector(h2o.getFrame(h2o.dl.10@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
              as.vector(x_train.hex$fraud)))
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param


### Good Model with reduced variables (10 fold cv) : Profit 155 and initial_weight_distribution = "UniformAdaptive"

# x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",25,10)
x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
h2o.dl.10 <- h2o.deeplearning(seed= 8770,
                              x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                              y= "fraud", training_frame = x_train.hex,
                              ignore_const_cols =T, reproducible = T,
                              fold_column = "fold", 
                              weights_column = "col_weight",
                              hidden = c(100,100,100,100,100),loss = "CrossEntropy",
                              distribution = "bernoulli", 
                              initial_weight_distribution = "UniformAdaptive",
                              activation = "Tanh", epochs = 25,
                              input_dropout_ratio = 0.03217263,
                              l1= 1.8e-05, l2= 5.9e-05,
                              keep_cross_validation_predictions = T,
                              missing_values_handling = "MeanImputation")

t <- table(as.vector(h2o.getFrame(h2o.dl.10@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
           as.vector(x_train.hex$fraud))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  ## 155

cvpreds_id <- h2o.dl.10@model$cross_validation_holdout_predictions_frame_id$name
cvpreds <- h2o.getFrame(cvpreds_id)
h2o_dl_155 <- as.data.frame(cvpreds)
write.csv(h2o_dl_155, "/gpfs/group/asb17/default/Vishal/h2o_dl_155_preds_on_train.csv")

# p <- predict(h2o.dl.10,
#              as.h2o(dat[,!(colnames(dat) %in% c("fraud","MRN","col_weight"))]))
# 

## Better Model with reduced variables (10 fold cv) : Profit 145 and seed = 1 for splitting the data

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
# x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",25,10)
h2o.dl.10 <- h2o.deeplearning(seed= 1166,
                              x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                              y= "fraud", training_frame = x_train.hex,
                              ignore_const_cols =T, reproducible = T,
                              fold_column = "fold", epochs = 20,
                              weights_column = "col_weight",
                              hidden = c(100,100,100,100,100),loss = "CrossEntropy",
                              distribution = "bernoulli", 
                              initial_weight_distribution = "UniformAdaptive",
                              activation = "Tanh",
                              input_dropout_ratio = 0.01475202,
                              l1= 9.5e-05, l2= 7.4e-05,
                              keep_cross_validation_predictions = T,
                              missing_values_handling = "MeanImputation")

t <- table(as.vector(h2o.getFrame(h2o.dl.10@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
           as.vector(x_train.hex$fraud))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  ## 145

cvpreds_id <- h2o.dl.10@model$cross_validation_holdout_predictions_frame_id$name
cvpreds <- h2o.getFrame(cvpreds_id)
h2o_dl_145 <- as.data.frame(cvpreds)
write.csv(h2o_dl_145, "/gpfs/group/asb17/default/Vishal/h2o_dl_145_preds_on_train.csv")


################### Comparison between models based on different seeds on training data

## Model with profit 155

library(h2o)
h2o.init(nthreads=-1, max_mem_size="30G")

Profit = 0;
N= 50;

for (i in 1:N){

  dat <- read.table(file = "/gpfs/group/asb17/default/DMC2019/task/train.csv", sep = '|', header = TRUE)
  dat$MRN  <- as.character(seq(1,dim(dat)[1]))
  
  f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
  nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))
  df <- floor(length(f)/10)
  dnf <- floor(length(nf)/10)    
  dat$fold <- rep(NA_integer_,dim(dat)[1])
  dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
  dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
  dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
  dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
  dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
  dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
  dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
  dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
  dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
  dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
  
  x_train_deep = as.data.frame(data.matrix(dat))
  x_train.hex <- as.h2o(x_train_deep)
  x_train.hex$fraud <- as.factor(x_train.hex$fraud)
  x_train.hex$trustLevel <- as.factor(x_train.hex$trustLevel)
  
  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
  h2o.dl.10.155 <- h2o.deeplearning(seed= 8770,
                                x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                y= "fraud", training_frame = x_train.hex,
                                ignore_const_cols =T, reproducible = T,
                                fold_column = "fold", 
                                weights_column = "col_weight",
                                hidden = c(100,100,100,100,100),loss = "CrossEntropy",
                                distribution = "bernoulli", 
                                initial_weight_distribution = "UniformAdaptive",
                                activation = "Tanh", epochs = 25,
                                input_dropout_ratio = 0.03217263,
                                l1= 1.8e-05, l2= 5.9e-05,
                                keep_cross_validation_predictions = T,
                                missing_values_handling = "MeanImputation")
  
  t <- table(as.vector(h2o.getFrame(h2o.dl.10.155@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
             as.vector(x_train.hex$fraud))
  
    Profit <- Profit + -5 * t[1,2]-25*t[2,1]+ 5*t[2,2]  

    }

Profit/N;    # -76.4

## Model with profit 145

library(h2o)
h2o.init(nthreads=-1, max_mem_size="30G")

Profit = 0;
N= 50;

for (i in 1:N){
  
  dat <- read.table(file = "/gpfs/group/asb17/default/DMC2019/task/train.csv", sep = '|', header = TRUE)
  dat$MRN  <- as.character(seq(1,dim(dat)[1]))
  
  f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
  nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))
  df <- floor(length(f)/10)
  dnf <- floor(length(nf)/10)    
  dat$fold <- rep(NA_integer_,dim(dat)[1])
  dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
  dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
  dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
  dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
  dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
  dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
  dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
  dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
  dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
  dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
  
  x_train_deep = as.data.frame(data.matrix(dat))
  x_train.hex <- as.h2o(x_train_deep)
  x_train.hex$fraud <- as.factor(x_train.hex$fraud)
  x_train.hex$trustLevel <- as.factor(x_train.hex$trustLevel)
  
  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
  # x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",25,10)
  h2o.dl.10.145 <- h2o.deeplearning(seed= 1166,
                                x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                y= "fraud", training_frame = x_train.hex,
                                ignore_const_cols =T, reproducible = T,
                                fold_column = "fold", epochs = 20,
                                weights_column = "col_weight",
                                hidden = c(100,100,100,100,100),loss = "CrossEntropy",
                                distribution = "bernoulli", 
                                initial_weight_distribution = "UniformAdaptive",
                                activation = "Tanh",
                                input_dropout_ratio = 0.01475202,
                                l1= 9.5e-05, l2= 7.4e-05,
                                keep_cross_validation_predictions = T,
                                missing_values_handling = "MeanImputation")
  
  t <- table(as.vector(h2o.getFrame(h2o.dl.10.145@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
             as.vector(x_train.hex$fraud))
  
  Profit <- Profit + -5 * t[1,2]-25*t[2,1]+ 5*t[2,2]  
  
}

Profit/N;  -55.6


#######################################################

# BART Machine : Reduced Dataset

options(java.parameters = "-Xmx50000m")
library("bartMachine")

dat <- read.table(file = "/gpfs/group/asb17/default/DMC2019/task/train.csv", sep = '|', header = TRUE)
table(dat$fraud,dat$trustLevel)

table(dat$fraud)

dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$fraud <- factor(dat$fraud) 

set.seed(1)

f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
table(dat$fold)


best_seednumber = 1234
best_test_profit = -10
best_param = list()

for (iter in 1:50) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    num_trees = sample(c(100,200,300),1),
    k = sample(c(1,2),1))
  
  
  bart.5 <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                          y = dat$fraud, verbose = F,use_missing_data=F,
                      folds_vec = as.integer(dat$fold), num_trees = hyper_params$num_trees, 
                      k = hyper_params$k, prob_rule_class = 0.5, seed = seed.number)
  
  
  t <- t(bart.5$confusion_matrix[1:2,1:2])
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param

## Best BART model yet : train seed = 1 : Profit 80 (Reduced Data)

bart.10 <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                    y = dat$fraud, verbose = T,use_missing_data=F,
                    folds_vec = as.integer(dat$fold), num_trees = 300, 
                    k = 2, prob_rule_class = 0.5 , seed = 4751)

t <- t(bart.10$confusion_matrix[1:2,1:2])
(max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))

# bart.10.final <- bartMachine(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
#                              y = dat$fraud, verbose = T,use_missing_data=F,
#                              num_trees = 300, 
#                              k = 2, prob_rule_class = 0.5 , seed = 4751)
# 
# p <- predict(bart.10.final, as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))])
#              ,prob_rule_class = 0.5, type = "class")


## Repeating over random seeds for the above model (Reduced data)

options(java.parameters = "-Xmx50000m")
library("bartMachine")

Profit = 0;
N= 10;

for (i in 1:N){
  
  dat <- read.table(file = "/gpfs/group/asb17/default/DMC2019/task/train.csv", sep = '|', header = TRUE)
  
  dat$MRN  <- as.character(seq(1,dim(dat)[1]))
  dat$fraud <- factor(dat$fraud) 
  
  f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
  nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))
  
  df <- floor(length(f)/10)
  dnf <- floor(length(nf)/10)    
  
  dat$fold <- rep(NA_integer_,dim(dat)[1])
  dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
  dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
  dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
  dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
  dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
  dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
  dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
  dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
  dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
  dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
  
  bart.10 <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                       y = dat$fraud,use_missing_data=F,verbose = F,
                       folds_vec = as.integer(dat$fold), num_trees = 300, 
                       k = 1, prob_rule_class = 0.5 , seed = 4751)
  
  t <- t(bart.10$confusion_matrix[1:2,1:2])
  
    Profit <- Profit + as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2])
  
}

Profit/N;  # -35 (tree = 300)  # - 85.3 (tree = 100)


# BART Machine: Fulldata

options(java.parameters = "-Xmx50000m")
library("bartMachine")

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_ncol462.tsv", sep = '\t', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$fraud <- factor(dat$fraud) 

# set.seed(1)
set.seed(10)
f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10

best_seednumber = 1234
best_test_profit = -10
best_param = list()

for (iter in 1:25) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    num_trees = sample(c(100,200,300),1),
    k = sample(c(1,2),1))
  
  
  bart.10 <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                      y = dat$fraud, verbose = T,use_missing_data=F,
                      folds_vec = as.integer(dat$fold), num_trees = hyper_params$num_trees, 
                      k = hyper_params$k, prob_rule_class = 0.5, seed = seed.number)
  
  
  t <- t(bart.10$confusion_matrix[1:2,1:2])
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param



## Best BART model yet : train seed = 1 : Profit 70 (Large Dataset)

bart.10 <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                     y = dat$fraud, verbose = F,use_missing_data=F,
                     folds_vec = as.integer(dat$fold), num_trees = 100, 
                     k = 2, prob_rule_class = 0.5 , seed = 1629)

t <- t(bart.10$confusion_matrix[1:2,1:2])
(max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))

#####################################

#####  Train v4 87

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v4_87.tsv", sep = '\t', header = TRUE)
table(dat$fraud,dat$trustLevel)

table(dat$fraud)

dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$col_weight <- ifelse(dat$fraud==0,25,10)

# set.seed(1) # main
# set.seed(20)
# set.seed(3)
# set.seed(450)
set.seed(123)

f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
table(dat$fold)

library(h2o)
h2o.init(nthreads=-1, max_mem_size="20G")

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)


## Tuning 

best_seednumber = 1234
best_test_profit = 0
best_param = list()

for (iter in 1:25) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    activation=sample(c("Tanh"),1),
    hidden = sample(list(c(200,200,200),c(150,150,150,150),c(100,100,100),c(100,100,100,100)),1)[[1]],
    input_dropout_ratio=runif(1,0,0.05),
    l1=sample(seq(0,1e-4,1e-6),1),
    l2=sample(seq(0,1e-4,1e-6),1),
    epochs = sample(c(50,25),1),
    weight_option = sample(list(c(25,10),c(50,20)),1)[[1]])
  
  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",hyper_params$weight_option[1],
                                   hyper_params$weight_option[2])
  
  h2o.dl.v4_87 <- h2o.deeplearning(x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                   y= "fraud", training_frame = x_train.hex,
                                   ignore_const_cols =T, seed =  seed.number, reproducible = T,
                                   loss = "CrossEntropy",weights_column = "col_weight",
                                   fold_column = "fold", epochs = hyper_params$epochs,
                                   initial_weight_distribution = "UniformAdaptive",
                                   hidden = hyper_params$hidden, 
                                   distribution = "bernoulli",activation = hyper_params$activation,
                                   input_dropout_ratio = hyper_params$input_dropout_ratio,
                                   l1=hyper_params$l1, l2=hyper_params$l2,
                                   keep_cross_validation_predictions = T,
                                   missing_values_handling = "MeanImputation")
  
  (t <- table((as.vector(h2o.getFrame(h2o.dl.v4_87@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
              as.vector(x_train.hex$fraud)))
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param


##### BEST MODEL YET : Profit 355 , fold seed = 3 

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
h2o.dl.355.v4_87_m1 <- h2o.deeplearning(seed= 3382,
                                       x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                       y= "fraud", training_frame = x_train.hex,
                                       ignore_const_cols =T, reproducible = T,
                                       fold_column = "fold", 
                                       weights_column = "col_weight",
                                       hidden = c(200,200,200),loss = "CrossEntropy",
                                       distribution = "bernoulli", 
                                       initial_weight_distribution = "UniformAdaptive",
                                       activation = "Tanh", epochs = 50,
                                       input_dropout_ratio = 0.008745386,
                                       l1= 1.8e-05, l2= 7.8e-05,
                                       keep_cross_validation_predictions = T,
                                       missing_values_handling = "MeanImputation")

(t <- table((as.vector(h2o.getFrame(h2o.dl.355.v4_87_m1@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  ## 355

# (as.data.frame(h2o.varimp(h2o.dl.355.v4_87_m1)))[1:10,]


##### BEST MODEL YET : Profit 350 , fold seed = 1 

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
h2o.dl.350.v4_87_m2 <- h2o.deeplearning(seed= 9719,
                                     x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                     y= "fraud", training_frame = x_train.hex,
                                     ignore_const_cols =T, reproducible = T,
                                     fold_column = "fold", 
                                     weights_column = "col_weight",
                                     hidden = c(100,100,100),loss = "CrossEntropy",
                                     distribution = "bernoulli", 
                                     initial_weight_distribution = "UniformAdaptive",
                                     activation = "Tanh", epochs = 50,
                                     input_dropout_ratio = 0.04782543,
                                     l1= 3.5e-05, l2= 1.5e-05,
                                     keep_cross_validation_predictions = T,
                                     missing_values_handling = "MeanImputation")

(t <- table((as.vector(h2o.getFrame(h2o.dl.350.v4_87_m2@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  ## 350


# (as.data.frame(h2o.varimp(h2o.dl.410.v4_87_m2)))[1:20,]


################

Conformity between m1 and m2

table(as.vector(h2o.getFrame(h2o.dl.410.v4_87_m1@model[["cross_validation_holdout_predictions_frame_id"]]
[["name"]])[,"predict"]),as.vector(h2o.getFrame(h2o.dl.410.v4_87_m2@model[["cross_validation_holdout_predictions_frame_id"]]
[["name"]])[,"predict"]))

###############

# Average Profit Story (NN 440)

Profit = 0;
N= 10;

for (i in 1:N){

  dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v4_87.tsv", sep = '\t', header = TRUE)
  dat$MRN  <- as.character(seq(1,dim(dat)[1]))
  
  f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
  nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))
  df <- floor(length(f)/10)
  dnf <- floor(length(nf)/10)    
  dat$fold <- rep(NA_integer_,dim(dat)[1])
  dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
  dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
  dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
  dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
  dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
  dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
  dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
  dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
  dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
  dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
  
  x_train_deep = as.data.frame(data.matrix(dat))
  x_train.hex <- as.h2o(x_train_deep)
  x_train.hex$fraud <- as.factor(x_train.hex$fraud)
  x_train.hex$trustLevel <- as.factor(x_train.hex$trustLevel)
  
  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
  h2o.dl.440.v4_87_m1 <- h2o.deeplearning(seed= 667,
                                          x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                          y= "fraud", training_frame = x_train.hex,
                                          ignore_const_cols =T, reproducible = T,
                                          fold_column = "fold", 
                                          weights_column = "col_weight",
                                          hidden = c(300,300,100,100),loss = "CrossEntropy",
                                          distribution = "bernoulli", 
                                          initial_weight_distribution = "UniformAdaptive",
                                          activation = "Tanh", epochs = 20,
                                          input_dropout_ratio = 0.03897825,
                                          l1= 7.1e-05, l2= 9.5e-05,
                                          keep_cross_validation_predictions = T,
                                          missing_values_handling = "MeanImputation")
  
  t <- table(as.vector(h2o.getFrame(h2o.dl.440.v4_87_m1@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
             as.vector(x_train.hex$fraud))
  
  Profit <- Profit + -5 * t[1,2]-25*t[2,1]+ 5*t[2,2]  
  
}

Profit/N ## 440 MODEL  # 343 - profit

######################

# Average Profit Story (NN 410)

Profit = 0;
N= 10;

for (i in 1:N){
  
  dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v4_87.tsv", sep = '\t', header = TRUE)
  dat$MRN  <- as.character(seq(1,dim(dat)[1]))
  
  f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
  nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))
  df <- floor(length(f)/10)
  dnf <- floor(length(nf)/10)    
  dat$fold <- rep(NA_integer_,dim(dat)[1])
  dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
  dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
  dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
  dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
  dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
  dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
  dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
  dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
  dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
  dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
  
  x_train_deep = as.data.frame(data.matrix(dat))
  x_train.hex <- as.h2o(x_train_deep)
  x_train.hex$fraud <- as.factor(x_train.hex$fraud)
  x_train.hex$trustLevel <- as.factor(x_train.hex$trustLevel)
  
  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
  h2o.dl.410.v4_87_m2 <- h2o.deeplearning(seed= 4514,
                                          x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                          y= "fraud", training_frame = x_train.hex,
                                          ignore_const_cols =T, reproducible = T,
                                          fold_column = "fold", 
                                          weights_column = "col_weight",
                                          hidden = c(100,100,100,100),loss = "CrossEntropy",
                                          distribution = "bernoulli", 
                                          initial_weight_distribution = "UniformAdaptive",
                                          activation = "Tanh", epochs = 20,
                                          input_dropout_ratio = 0.01178966,
                                          l1= 1.2e-05, l2= 6.1e-05,
                                          keep_cross_validation_predictions = T,
                                          missing_values_handling = "MeanImputation")
  
  t <- table(as.vector(h2o.getFrame(h2o.dl.410.v4_87_m2@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
             as.vector(x_train.hex$fraud))
  
  Profit <- Profit + -5 * t[1,2]-25*t[2,1]+ 5*t[2,2]  
  
}

Profit/N ## 410 MODEL  # 333

###################################### train_cs0.5

library(h2o)
h2o.init(nthreads=-1, max_mem_size="20G")


dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_cs0.5.tsv", sep = '\t', header = TRUE)
table(dat$fraud,dat$trustLevel)

table(dat$fraud)

dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$col_weight <- ifelse(dat$fraud==0,25,10)

# set.seed(100) 
# set.seed(4459) 
set.seed(2)


f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
table(dat$fold)

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)


## Tuning 

best_seednumber = 1234
best_test_profit = 0
best_param = list()

for (iter in 1:20) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    activation=sample(c("Tanh"),1),
    hidden = sample(list(c(200,200,200),c(150,150,150),c(100,100,100),c(100,100,100,100)),1)[[1]],
    input_dropout_ratio=runif(1,0,0.05),
    l1=sample(seq(0,1e-4,1e-6),1),
    l2=sample(seq(0,1e-4,1e-6),1),
    epochs = sample(c(15,20,25),1),
    weight_option = sample(list(c(25,10),c(50,20)),1)[[1]])
  
  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",hyper_params$weight_option[1],
                                   hyper_params$weight_option[2])
  
  h2o.dl.train_cs0.5 <- h2o.deeplearning(x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                   y= "fraud", training_frame = x_train.hex,
                                   ignore_const_cols =T, seed =  seed.number, reproducible = T,
                                   loss = "CrossEntropy",weights_column = "col_weight",
                                   fold_column = "fold", epochs = hyper_params$epochs,
                                   initial_weight_distribution = "UniformAdaptive",
                                   hidden = hyper_params$hidden, 
                                   distribution = "bernoulli",activation = hyper_params$activation,
                                   input_dropout_ratio = hyper_params$input_dropout_ratio,
                                   l1=hyper_params$l1, l2=hyper_params$l2,
                                   keep_cross_validation_predictions = T,
                                   missing_values_handling = "MeanImputation")
  
  (t <- table(as.vector(h2o.getFrame(h2o.dl.train_cs0.5@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
              as.vector(x_train.hex$fraud)))
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param

#### Good model: profit 410 ; train seed: 2

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
h2o.dl.410.train_cs0.5 <- h2o.deeplearning(seed= 2815,
                                           x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                           y= "fraud", training_frame = x_train.hex,
                                           ignore_const_cols =T, reproducible = T,
                                           fold_column = "fold", 
                                           weights_column = "col_weight",
                                           hidden = c(200,200,250),loss = "CrossEntropy",
                                           distribution = "bernoulli", 
                                           initial_weight_distribution = "UniformAdaptive",
                                           activation = "Tanh", epochs = 30,
                                           input_dropout_ratio = 0.04453855,
                                           l1= 6e-06, l2= 8.5e-05,
                                           keep_cross_validation_predictions = T,
                                           missing_values_handling = "MeanImputation")

t <- table(as.vector(h2o.getFrame(h2o.dl.410.train_cs0.5@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
           as.vector(x_train.hex$fraud))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 410


#### Good model: profit 420 ; train seed: 100

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
h2o.dl.420.train_cs0.5 <- h2o.deeplearning(seed= 3003,
                                           x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                           y= "fraud", training_frame = x_train.hex,
                                           ignore_const_cols =T, reproducible = T,
                                           fold_column = "fold", 
                                           weights_column = "col_weight",
                                           hidden = c(200,200,200),loss = "CrossEntropy",
                                           distribution = "bernoulli", 
                                           initial_weight_distribution = "UniformAdaptive",
                                           activation = "Tanh", epochs = 25,
                                           input_dropout_ratio = 0.03322926,
                                           l1= 7.9e-05, l2= 7.2e-05,
                                           keep_cross_validation_predictions = T,
                                           missing_values_handling = "MeanImputation")

t <- table(as.vector(h2o.getFrame(h2o.dl.420.train_cs0.5@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
           as.vector(x_train.hex$fraud))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 420


#### Good model: profit 420 ; train seed: 4459

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",25,10)
h2o.dl.420.train_cs0.5_m2 <- h2o.deeplearning(seed= 3598,
                                           x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                           y= "fraud", training_frame = x_train.hex,
                                           ignore_const_cols =T, reproducible = T,
                                           fold_column = "fold", 
                                           weights_column = "col_weight",
                                           hidden = c(200,200,200),loss = "CrossEntropy",
                                           distribution = "bernoulli", 
                                           initial_weight_distribution = "UniformAdaptive",
                                           activation = "Tanh", epochs = 25,
                                           input_dropout_ratio = 0.01370762,
                                           l1= 2.3e-05, l2= 4e-05,
                                           keep_cross_validation_predictions = T,
                                           missing_values_handling = "MeanImputation")

t <- table(as.vector(h2o.getFrame(h2o.dl.420.train_cs0.5_m2@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
           as.vector(x_train.hex$fraud))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 420

###################################### train_v4.2_53.tsv

library(h2o)
h2o.init(nthreads=-1, max_mem_size="10G")


dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v4.2_53.tsv", sep = '\t', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))

# set.seed(1)
# set.seed(100)
# set.seed(4459)
# set.seed(100)
set.seed(1492)
# set.seed(142)

f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
table(dat$fold)

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)


## Tuning 

best_seednumber = 1234
best_test_profit = 0
best_param = list()

for (iter in 1:15) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    activation=sample(c("Tanh"),1),
    hidden = sample(list(c(200,200,200),c(150,150,150,150),c(100,100,100),c(100,100,100,100)),1)[[1]],
    input_dropout_ratio=runif(1,0,0.05),
    l1=sample(seq(0,1e-4,1e-6),1),
    l2=sample(seq(0,1e-4,1e-6),1),
    epochs = sample(c(50,25),1),
    weight_option = sample(list(c(25,10),c(50,20)),1)[[1]])
  
  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",hyper_params$weight_option[1],
                                   hyper_params$weight_option[2])
  
  h2o.dl.train_v4.2_53 <- h2o.deeplearning(x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                           y= "fraud", training_frame = x_train.hex,
                                           ignore_const_cols =T, seed =  seed.number, reproducible = T,
                                           loss = "CrossEntropy",weights_column = "col_weight",
                                           fold_column = "fold", epochs = hyper_params$epochs,
                                           initial_weight_distribution = "UniformAdaptive",
                                           hidden = hyper_params$hidden, 
                                           distribution = "bernoulli",activation = hyper_params$activation,
                                           input_dropout_ratio = hyper_params$input_dropout_ratio,
                                           l1=hyper_params$l1, l2=hyper_params$l2,
                                           keep_cross_validation_predictions = T,
                                           missing_values_handling = "MeanImputation")
  
  (t <- table((as.vector(h2o.getFrame(h2o.dl.train_v4.2_53@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
              as.vector(x_train.hex$fraud)))
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param


# #### Good model: profit 425 ; train seed: 1
# 
# x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",25,10)
# h2o.dl.425.train_v4.2_53 <- h2o.deeplearning(seed= 572,
#                                               x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
#                                               y= "fraud", training_frame = x_train.hex,
#                                               ignore_const_cols =T, reproducible = T,
#                                               fold_column = "fold", 
#                                               weights_column = "col_weight",
#                                               hidden = c(200,200,200),loss = "CrossEntropy",
#                                               distribution = "bernoulli", 
#                                               initial_weight_distribution = "UniformAdaptive",
#                                               activation = "Tanh", epochs = 50,
#                                               input_dropout_ratio = 0.03005215,
#                                               l1= 5.3e-05, l2= 3.8e-05,
#                                               keep_cross_validation_predictions = T,
#                                               missing_values_handling = "MeanImputation")
# 
# (t <- table((as.vector(h2o.getFrame(h2o.dl.425.train_v4.2_53@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
#             as.vector(x_train.hex$fraud)))
# 
# (Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 425
# 

#### Good model: profit 355 ; train seed: 1492 (better model wrt p1)
c(100,98,100,98)

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
h2o.dl.355.train_v4.2_53 <- h2o.deeplearning(seed= 9735,
                                             x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                             y= "fraud", training_frame = x_train.hex,
                                             ignore_const_cols =T, reproducible = T,
                                             fold_column = "fold", 
                                             weights_column = "col_weight",
                                             hidden = c(100,98,100,98),loss = "CrossEntropy",
                                             distribution = "bernoulli", 
                                             initial_weight_distribution = "UniformAdaptive",
                                             activation = "Tanh", epochs = 50,
                                             input_dropout_ratio = 0.01044282,
                                             l1= 4.1e-05, l2= 1.9e-05,
                                             keep_cross_validation_predictions = T,
                                             missing_values_handling = "MeanImputation")

(t <- table((as.vector(h2o.getFrame(h2o.dl.355.train_v4.2_53@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 355


# #### Good model: profit 440 ; train seed: 142
# 
# c(100,100,100,99)
# 
# x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
# h2o.dl.440.train_v4.2_53 <- h2o.deeplearning(seed= 2725,
#                                              x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
#                                              y= "fraud", training_frame = x_train.hex,
#                                              ignore_const_cols =T, reproducible = T,
#                                              fold_column = "fold", 
#                                              weights_column = "col_weight",
#                                              hidden = c(100,100,100,99),
#                                              loss = "CrossEntropy",
#                                              distribution = "bernoulli", 
#                                              initial_weight_distribution = "UniformAdaptive",
#                                              activation = "Tanh", epochs = 25,
#                                              input_dropout_ratio = 0.03031014,
#                                              l1= 4.5e-05, l2= 8.4e-05,
#                                              keep_cross_validation_predictions = T,
#                                              missing_values_handling = "MeanImputation")
# 
# (t <- table((as.vector(h2o.getFrame(h2o.dl.440.train_v4.2_53@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
#             as.vector(x_train.hex$fraud)))
# 
# (Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 440
# 

###################################### train_v5.2_128.tsv

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v5.2_128.tsv", sep = '\t', header = TRUE)
table(dat$fraud,dat$trustLevel)

table(dat$fraud)

dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$col_weight <- ifelse(dat$fraud==0,25,10)

# set.seed(1)
# set.seed(100)
# set.seed(4459)
# set.seed(10)
# set.seed(1492)

# set.seed(123)
set.seed(12345)

f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
table(dat$fold)

library(h2o)
h2o.init(nthreads=-1, max_mem_size="20G")

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)


## Tuning 

best_seednumber = 1234
best_test_profit = 0
best_param = list()

for (iter in 1:15) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    activation=sample(c("Tanh"),1),
    hidden = sample(list(c(200,200,200),c(150,150,150,150),c(100,100,100),c(100,100,100,100)),1)[[1]],
    input_dropout_ratio=runif(1,0,0.05),
    l1=sample(seq(0,1e-4,1e-6),1),
    l2=sample(seq(0,1e-4,1e-6),1),
    epochs = sample(c(50,25),1),
    weight_option = sample(list(c(25,10),c(50,20)),1)[[1]])
  
  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",hyper_params$weight_option[1],
                                   hyper_params$weight_option[2])
  
  h2o.dl.train_v5.2_128 <- h2o.deeplearning(x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                           y= "fraud", training_frame = x_train.hex,
                                           ignore_const_cols =T, seed =  seed.number, reproducible = T,
                                           loss = "CrossEntropy",weights_column = "col_weight",
                                           fold_column = "fold", epochs = hyper_params$epochs,
                                           initial_weight_distribution = "UniformAdaptive",
                                           hidden = hyper_params$hidden, 
                                           distribution = "bernoulli",activation = hyper_params$activation,
                                           input_dropout_ratio = hyper_params$input_dropout_ratio,
                                           l1=hyper_params$l1, l2=hyper_params$l2,
                                           keep_cross_validation_predictions = T,
                                           missing_values_handling = "MeanImputation")
  
  (t <- table((as.vector(h2o.getFrame(h2o.dl.train_v5.2_128@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
              as.vector(x_train.hex$fraud)))
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param


#### Good model: profit 400 ; train seed: 1

c(100,100,100,99)

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
h2o.dl.400.train_v5.2_128 <- h2o.deeplearning(seed= 7933,
                                             x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                             y= "fraud", training_frame = x_train.hex,
                                             ignore_const_cols =T, reproducible = T,
                                             fold_column = "fold", 
                                             weights_column = "col_weight",
                                             hidden = c(100,100,100,99),loss = "CrossEntropy",
                                             distribution = "bernoulli", 
                                             initial_weight_distribution = "UniformAdaptive",
                                             activation = "Tanh", epochs = 50,
                                             input_dropout_ratio = 0.03583845,
                                             l1= 8.8e-05, l2= 8.7e-05,
                                             keep_cross_validation_predictions = T,
                                             missing_values_handling = "MeanImputation")

t <- table(as.vector(h2o.getFrame(h2o.dl.400.train_v5.2_128@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
           as.vector(x_train.hex$fraud))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 400

###################################### train_v5.2_35.tsv

library(h2o)
h2o.init(nthreads=-1, max_mem_size="20G")

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v5.2_35.tsv", sep = '\t', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$col_weight <- ifelse(dat$fraud==0,25,10)

set.seed(1)
# set.seed(100)
# set.seed(4459)
# set.seed(10)
# set.seed(1492)

f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)


## Tuning 

best_seednumber = 1234
best_test_profit = 0
best_param = list()

for (iter in 1:20) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    activation=sample(c("Tanh"),1),
    hidden = sample(list(c(200,200,200),c(150,150,150,150),c(100,100,100),c(100,100,100,100)),1)[[1]],
    input_dropout_ratio=runif(1,0,0.05),
    l1=sample(seq(0,1e-4,1e-6),1),
    l2=sample(seq(0,1e-4,1e-6),1),
    epochs = sample(c(50,25),1),
    weight_option = sample(list(c(25,10),c(50,20)),1)[[1]])
  
  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",hyper_params$weight_option[1],
                                   hyper_params$weight_option[2])
  
  h2o.dl.train_v5.2_35 <- h2o.deeplearning(x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                            y= "fraud", training_frame = x_train.hex,
                                            ignore_const_cols =T, seed =  seed.number, reproducible = T,
                                            loss = "CrossEntropy",weights_column = "col_weight",
                                            fold_column = "fold", epochs = hyper_params$epochs,
                                            initial_weight_distribution = "UniformAdaptive",
                                            hidden = hyper_params$hidden, 
                                            distribution = "bernoulli",activation = hyper_params$activation,
                                            input_dropout_ratio = hyper_params$input_dropout_ratio,
                                            l1=hyper_params$l1, l2=hyper_params$l2,
                                            keep_cross_validation_predictions = T,
                                            missing_values_handling = "MeanImputation")
  
  (t <- table((as.vector(h2o.getFrame(h2o.dl.train_v5.2_35@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
              as.vector(x_train.hex$fraud)))
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param


#### Good model: profit 340 ; train seed: 1

c(200,198,200)

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
h2o.dl.340.train_v5.2_35 <- h2o.deeplearning(seed= 5730,
                                              x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                              y= "fraud", training_frame = x_train.hex,
                                              ignore_const_cols =T, reproducible = T,
                                              fold_column = "fold", 
                                              weights_column = "col_weight",
                                              hidden = c(200,198,202),loss = "CrossEntropy",
                                              distribution = "bernoulli", 
                                              initial_weight_distribution = "UniformAdaptive",
                                              activation = "Tanh", epochs = 50,
                                              input_dropout_ratio = 0.009377728,
                                              l1= 5.3e-05, l2= 3.3e-05,
                                              keep_cross_validation_predictions = T,
                                              missing_values_handling = "MeanImputation")

(t <- table((as.vector(h2o.getFrame(h2o.dl.340.train_v5.2_35@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 340


#################################

# BART

options(java.parameters = "-Xmx20000m")
library("bartMachine")

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v4.2_53.tsv", sep = '\t', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$fraud <- factor(dat$fraud) 

set.seed(1)
# set.seed(10)

f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
table(dat$fold)

best_seednumber = 1234
best_test_profit = 0
best_param = list()

for (iter in 1:10) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    num_trees = sample(c(100,200,300,500),1),
    k = sample(c(1,2),1))
  
  
  bart_v4.2_53 <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                      y = dat$fraud, verbose = F,use_missing_data=F,
                      folds_vec = as.integer(dat$fold), num_trees = hyper_params$num_trees, 
                      k = hyper_params$k, prob_rule_class = 0.5, seed = seed.number)
  
  t <- t(bart_v4.2_53$confusion_matrix[1:2,1:2])
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param

###############################

# BART v4_87

options(java.parameters = "-Xmx22000m")
library("bartMachine")

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v4_87.tsv", sep = '\t', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$fraud <- factor(dat$fraud) 

set.seed(1)
# set.seed(10)

f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
table(dat$fold)

best_seednumber = 1234
best_test_profit = 0
best_param = list()

for (iter in 1:35) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    num_trees = sample(c(50,100,200,300),1),
    k = sample(c(1,2),1))
  
  
  bart_v4_87 <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                            y = dat$fraud, verbose = F,use_missing_data=F,
                            folds_vec = as.integer(dat$fold), num_trees = hyper_params$num_trees, 
                            k = hyper_params$k, prob_rule_class = 0.5, seed = seed.number)
  
  t <- t(bart_v4_87$confusion_matrix[1:2,1:2])
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param

###################################### train initial

library(h2o)
h2o.init(nthreads=-1, max_mem_size="20G")


dat <- read.table(file = "/gpfs/group/asb17/default/DMC2019/task/train.csv", sep = '|', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$col_weight <- ifelse(dat$fraud==0,25,10)

# set.seed(1)
set.seed(100)
# set.seed(1234)
# set.seed(10)
# set.seed(123)

f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)


## Tuning 

best_seednumber = 1234
best_test_profit = -200
best_param = list()

for (iter in 1:60) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    activation=sample(c("Tanh"),1),
    hidden = sample(list(c(200,200,200),c(150,150,150,150),c(100,100,100),c(100,100,100,100)),1)[[1]],
    input_dropout_ratio=runif(1,0,0.05),
    l1=sample(seq(0,1e-4,1e-6),1),
    l2=sample(seq(0,1e-4,1e-6),1),
    epochs = sample(c(50,25),1),
    weight_option = sample(list(c(25,10),c(50,20)),1)[[1]])
  
  x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",hyper_params$weight_option[1],
                                   hyper_params$weight_option[2])
  
  h2o.dl.train_initial <- h2o.deeplearning(x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                           y= "fraud", training_frame = x_train.hex,
                                           ignore_const_cols =T, seed =  seed.number, reproducible = T,
                                           loss = "CrossEntropy",weights_column = "col_weight",
                                           fold_column = "fold", epochs = hyper_params$epochs,
                                           initial_weight_distribution = "UniformAdaptive",
                                           hidden = hyper_params$hidden, 
                                           distribution = "bernoulli",activation = hyper_params$activation,
                                           input_dropout_ratio = hyper_params$input_dropout_ratio,
                                           l1=hyper_params$l1, l2=hyper_params$l2,
                                           keep_cross_validation_predictions = T,
                                           missing_values_handling = "MeanImputation")
  
  (t <- table((as.vector(h2o.getFrame(h2o.dl.train_initial@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
              as.vector(x_train.hex$fraud)))
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param


#### Good model: profit  ; train seed: 100

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
h2o.dl.train_initial <- h2o.deeplearning(seed= 1937,
                                             x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                             y= "fraud", training_frame = x_train.hex,
                                             ignore_const_cols =T, reproducible = T,
                                             fold_column = "fold", 
                                             weights_column = "col_weight",
                                             hidden = c(200,200,200),loss = "CrossEntropy",
                                             distribution = "bernoulli", 
                                             initial_weight_distribution = "UniformAdaptive",
                                             activation = "Tanh", epochs = 50,
                                             input_dropout_ratio = 0.001562837,
                                             l1= 2e-06, l2= 9.8e-05,
                                             keep_cross_validation_predictions = T,
                                             missing_values_handling = "MeanImputation")

(t <- table((as.vector(h2o.getFrame(h2o.dl.train_initial@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  



#### Good model: profit  ; train seed: 123

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
h2o.dl.train_initial <- h2o.deeplearning(seed= 3339,
                                         x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                         y= "fraud", training_frame = x_train.hex,
                                         ignore_const_cols =T, reproducible = T,
                                         fold_column = "fold", 
                                         weights_column = "col_weight",
                                         hidden = c(100,100,100),loss = "CrossEntropy",
                                         distribution = "bernoulli", 
                                         initial_weight_distribution = "UniformAdaptive",
                                         activation = "Tanh", epochs = 50,
                                         input_dropout_ratio = 0.003407305,
                                         l1= 3.4e-06, l2= 5.3e-05,
                                         keep_cross_validation_predictions = T,
                                         missing_values_handling = "MeanImputation")

(t <- table((as.vector(h2o.getFrame(h2o.dl.train_initial@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  

# BART v5.2_35

options(java.parameters = "-Xmx22000m")
library("bartMachine")

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v5.2_35.tsv", sep = '\t', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$fraud <- factor(dat$fraud) 

# set.seed(1)
set.seed(123)

f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
table(dat$fold)


best_seednumber = 1234
best_test_profit = 0
best_param = list()

for (iter in 1:40) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    num_trees = sample(c(50,100,200),1),
    k = sample(c(1,2),1))
  
  
  bart_v5.2_35 <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                          y = dat$fraud, verbose = F,use_missing_data=F,
                          folds_vec = as.integer(dat$fold), num_trees = hyper_params$num_trees, 
                          k = hyper_params$k, prob_rule_class = 0.5, seed = seed.number)
  
  t <- t(bart_v5.2_35$confusion_matrix[1:2,1:2])
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param

# BART v5.2_128

options(java.parameters = "-Xmx22000m")
library("bartMachine")

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v5.2_128.tsv", sep = '\t', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$fraud <- factor(dat$fraud) 

set.seed(1)
# set.seed(123)

f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
table(dat$fold)


best_seednumber = 1234
best_test_profit = 0
best_param = list()

for (iter in 1:40) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    num_trees = sample(c(50,100,200),1),
    k = sample(c(1,2),1))
  
  
  bart_v5.2_128 <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                            y = dat$fraud, verbose = F,use_missing_data=F,
                            folds_vec = as.integer(dat$fold), num_trees = hyper_params$num_trees, 
                            k = hyper_params$k, prob_rule_class = 0.5, seed = seed.number)
  
  t <- t(bart_v5.2_128$confusion_matrix[1:2,1:2])
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param


# BART v0

options(java.parameters = "-Xmx22000m")
library("bartMachine")

dat <- read.table(file = "/gpfs/group/asb17/default/DMC2019/task/train.csv", sep = '|', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$fraud <- factor(dat$fraud) 

# set.seed(1)
set.seed(123)

f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10
table(dat$fold)


best_seednumber = 1234
best_test_profit = 0
best_param = list()

for (iter in 1:35) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    num_trees = sample(c(50,100,200),1),
    k = sample(c(1,2),1))
  
  
  bart_v0 <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                             y = dat$fraud, verbose = F,use_missing_data=F,
                             folds_vec = as.integer(dat$fold), num_trees = hyper_params$num_trees, 
                             k = hyper_params$k, prob_rule_class = 0.5, seed = seed.number)
  
  t <- t(bart_v0$confusion_matrix[1:2,1:2])
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param

## NN cs0.5


library(h2o)
h2o.init(nthreads=-1, max_mem_size="20G")

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_cs0.5.tsv", sep = '\t', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))

# set.seed(1)
# set.seed(100)
set.seed(1234)
# set.seed(10)
# set.seed(123)

f <- sample(dat$MRN[dat$fraud == 1],length(dat$MRN[dat$fraud == 1]))
nf <- sample(dat$MRN[dat$fraud == 0],length(dat$MRN[dat$fraud == 0]))

df <- floor(length(f)/10)
dnf <- floor(length(nf)/10)    

dat$fold <- rep(NA_integer_,dim(dat)[1])
dat$fold[dat$MRN %in% c(f[1:df],nf[1:dnf])] <- 1
dat$fold[dat$MRN %in% c(f[(df+1):(2*df)],nf[(dnf+1):(2*dnf)])] <- 2
dat$fold[dat$MRN %in% c(f[(2*df+1):(3*df)],nf[(2*dnf+1):(3*dnf)])] <- 3
dat$fold[dat$MRN %in% c(f[(3*df+1):(4*df)],nf[(3*dnf+1):(4*dnf)])] <- 4
dat$fold[dat$MRN %in% c(f[(4*df+1):(5*df)],nf[(4*dnf+1):(5*dnf)])] <- 5
dat$fold[dat$MRN %in% c(f[(5*df+1):(6*df)],nf[(5*dnf+1):(6*dnf)])] <- 6
dat$fold[dat$MRN %in% c(f[(6*df+1):(7*df)],nf[(6*dnf+1):(7*dnf)])] <- 7
dat$fold[dat$MRN %in% c(f[(7*df+1):(8*df)],nf[(7*dnf+1):(8*dnf)])] <- 8
dat$fold[dat$MRN %in% c(f[(8*df+1):(9*df)],nf[(8*dnf+1):(9*dnf)])] <- 9
dat$fold[dat$MRN %in% c(f[(9*df+1):length(f)],nf[(9*dnf+1):length(nf)])] <- 10

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)


## Tuning 

best_seednumber = 1234
best_test_profit = -200
best_param = list()

for (iter in 1:50) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    activation=sample(c("Tanh"),1),
    hidden = sample(list(c(200,200,200),c(150,150,150,150),c(100,100,100),c(100,100,100,100)),1)[[1]],
    input_dropout_ratio=runif(1,0,0.05),
    l1=sample(seq(0,1e-4,1e-6),1),
    l2=sample(seq(0,1e-4,1e-6),1),
    epochs = sample(c(50,25),1))
  
   h2o.dl.cs0.5 <- h2o.deeplearning(x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","fold"))], 
                                           y= "fraud", training_frame = x_train.hex,
                                           ignore_const_cols =T, seed =  seed.number, reproducible = T,
                                           loss = "CrossEntropy", fold_column = "fold", 
                                           epochs = hyper_params$epochs,
                                           initial_weight_distribution = "UniformAdaptive",
                                           hidden = hyper_params$hidden, 
                                           distribution = "bernoulli",activation = hyper_params$activation,
                                           input_dropout_ratio = hyper_params$input_dropout_ratio,
                                           l1=hyper_params$l1, l2=hyper_params$l2,
                                           keep_cross_validation_predictions = T,
                                           missing_values_handling = "MeanImputation")
  
  (t <- table((as.vector(h2o.getFrame(h2o.dl.cs0.5@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
              as.vector(x_train.hex$fraud)))
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param


#### Good model: profit  ; train seed: 123

x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)
h2o.dl.train_initial <- h2o.deeplearning(seed= 3339,
                                         x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                         y= "fraud", training_frame = x_train.hex,
                                         ignore_const_cols =T, reproducible = T,
                                         fold_column = "fold", 
                                         weights_column = "col_weight",
                                         hidden = c(100,100,100),loss = "CrossEntropy",
                                         distribution = "bernoulli", 
                                         initial_weight_distribution = "UniformAdaptive",
                                         activation = "Tanh", epochs = 50,
                                         input_dropout_ratio = 0.003407305,
                                         l1= 3.4e-06, l2= 5.3e-05,
                                         keep_cross_validation_predictions = T,
                                         missing_values_handling = "MeanImputation")

(t <- table((as.vector(h2o.getFrame(h2o.dl.train_initial@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  


############################# Model Stacking

library(h2o)
h2o.init(nthreads=-1, max_mem_size="20G")

NN_128 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/OOB_NN_v5.2_128.tsv", sep = '\t', header = TRUE)
NN_CS0.5 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/OOB_NN_cs0.5.tsv", sep = '\t', header = TRUE)
NN_53_OLD <- read.table(file = "/gpfs/group/asb17/default/renan/oob/OOB_NN_OLD_v4.2_53.tsv", sep = '\t', header = TRUE)
NN_87 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/OOB_NN_v4_87.tsv", sep = '\t', header = TRUE)
NN_35 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/OOB_NN_v5.2_35.tsv", sep = '\t', header = TRUE)   
NN_53_NEW <- read.table(file = "/gpfs/group/asb17/default/renan/oob/OOB_NN_NEW_v4.2_53.tsv", sep = '\t', header = TRUE)   
NN_INITIAL <- read.table(file = "/gpfs/group/asb17/default/renan/oob/OOB_NN_initial_fulldata.tsv", sep = '\t', header = TRUE)   

BART_V0 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/OOB_BART_v0.tsv", sep = '\t', header = TRUE)   
BART_53 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/OOB_BART_v4.2_53.tsv", sep = '\t', header = TRUE)   
BART_87 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/OOB_BART_v4_87.tsv", sep = '\t', header = TRUE)   
BART_128 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/OOB_BART_v5.2_128.tsv", sep = '\t', header = TRUE)   
BART_35 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/OOB_BART_v5.2_35.tsv", sep = '\t', header = TRUE)   


LOGISTIC <- read.table(file = "/gpfs/group/asb17/default/renan/oob/logistic_v0_oob.tsv", sep = '\t', header = TRUE)   
DAN_53_BEST_57 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/dan_train_v4.2_53.tsv.best.oob57.tsv", sep = '\t', header = TRUE)   
DAN_87_BEST_57 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/dan_train_v4_87.tsv.best.oob57.tsv", sep = '\t', header = TRUE)   
DAN_35_BEST_57 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/dan_train_v5.2_35.tsv.best.oob57.tsv", sep = '\t', header = TRUE)   
DAN_0.5_BEST_57 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/dan_train_cs0.5.tsv.best.oob57.tsv", sep = '\t', header = TRUE)   
DAN_128_BEST_57 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/dan_train_v5.2_128.tsv.best.oob57.tsv", sep = '\t', header = TRUE)   
DAN_V0_BEST_57 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/dan_train.csv.best.oob57.tsv", sep = '\t', header = TRUE)   

REN_xgbt_v0 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/oob_xgbt_v0.tsv", sep = '\t', header = TRUE)   
REN_xgbt_v5.2_128 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/oob_xgbt_v5.2_128.tsv", sep = '\t', header = TRUE)   
REN_xgbt_v4_87 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/oob_xgbt_v4_87.tsv", sep = '\t', header = TRUE)   
REN_xgbt_v4.2_53 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/oob_xgbt_v4.2_53.tsv", sep = '\t', header = TRUE)   
REN_xgbt_cs0.5 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/oob_xgbt_cs0.5.tsv", sep = '\t', header = TRUE)   
REN_xgbt_v5.2_35 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/oob_xgbt_v5.2_35.tsv", sep = '\t', header = TRUE)   

REN_xgbf_v0 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/oob_xgbf_v0.tsv", sep = '\t', header = TRUE)   
REN_xgbf_v5.2_128 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/oob_xgbf_v5.2_128.tsv", sep = '\t', header = TRUE)   
REN_xgbf_v4_87 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/oob_xgbf_v4_87.tsv", sep = '\t', header = TRUE)   
REN_xgbf_v4.2_53 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/oob_xgbf_v4.2_53.tsv", sep = '\t', header = TRUE)   
REN_xgbf_cs0.5 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/oob_xgbf_cs0.5.tsv", sep = '\t', header = TRUE)   
REN_xgbf_v5.2_35 <- read.table(file = "/gpfs/group/asb17/default/renan/oob/oob_xgbf_v5.2_35.tsv", sep = '\t', header = TRUE)   


stack_all <- data.frame(NN_128 = NN_128$pred,NN_CS0.5 = NN_CS0.5$pred,NN_53_OLD = NN_53_OLD$pred,
                        NN_87 = NN_87$pred, NN_35 = NN_35$pred,NN_53_NEW = NN_53_NEW$pred,
                        NN_INITIAL = NN_INITIAL$pred, 
                        
                        BART_53 = BART_53$pred, BART_87 = BART_87$pred,BART_V0 = BART_V0$pred,
                        BART_128 = BART_128$pred,BART_35 = BART_35$pred,
                        
                        LOGISTIC = LOGISTIC$pred,
                        DAN_53_BEST_57 = DAN_53_BEST_57$pred,
                        DAN_87_BEST_57 = DAN_87_BEST_57$pred,
                        DAN_35_BEST_57 = DAN_35_BEST_57$pred,
                        DAN_0.5_BEST_57 = DAN_0.5_BEST_57$pred, 
                        DAN_128_BEST_57 = DAN_128_BEST_57$pred,
                        DAN_V0_BEST_57 = DAN_V0_BEST_57$pred,
                        
                        REN_xgbt_v0 = REN_xgbt_v0$pred,
                        REN_xgbt_v5.2_128 = REN_xgbt_v5.2_128$pred,REN_xgbt_v4_87 = REN_xgbt_v4_87$pred,
                        REN_xgbt_v4.2_53 = REN_xgbt_v4.2_53$pred, REN_xgbt_cs0.5 = REN_xgbt_cs0.5$pred,
                        REN_xgbt_v5.2_35 = REN_xgbt_v5.2_35$pred,
                        REN_xgbf_v0 = REN_xgbf_v0$pred,
                        REN_xgbf_v5.2_128 = REN_xgbf_v5.2_128$pred,REN_xgbf_v4_87 = REN_xgbf_v4_87$pred,
                        REN_xgbf_v4.2_53 = REN_xgbf_v4.2_53$pred, REN_xgbf_cs0.5 = REN_xgbf_cs0.5$pred,
                        REN_xgbf_v5.2_35 = REN_xgbf_v5.2_35$pred)


stack_No_REN <- data.frame(NN_128 = NN_128$pred,NN_CS0.5 = NN_CS0.5$pred,NN_53_OLD = NN_53_OLD$pred,
                           NN_87 = NN_87$pred, NN_35 = NN_35$pred,NN_53_NEW = NN_53_NEW$pred,
                           NN_INITIAL = NN_INITIAL$pred, 
                           
                           BART_53 = BART_53$pred, BART_87 = BART_87$pred,BART_V0 = BART_V0$pred,
                           BART_128 = BART_128$pred,BART_35 = BART_35$pred,
                           
                           LOGISTIC = LOGISTIC$pred,
                           DAN_53_BEST_57 = DAN_53_BEST_57$pred,
                           DAN_87_BEST_57 = DAN_87_BEST_57$pred,
                           DAN_35_BEST_57 = DAN_35_BEST_57$pred,
                           DAN_0.5_BEST_57 = DAN_0.5_BEST_57$pred, 
                           DAN_128_BEST_57 = DAN_128_BEST_57$pred,
                           DAN_V0_BEST_57 = DAN_V0_BEST_57$pred)

stack_128 <- data.frame(NN_128 = NN_128$pred,NN_53_OLD = NN_53_OLD$pred,
                        NN_87 = NN_87$pred, NN_53_NEW = NN_53_NEW$pred,
                        
                        BART_53 = BART_53$pred, BART_87 = BART_87$pred,
                        BART_128 = BART_128$pred,BART_35 = BART_35$pred,
                        
                        DAN_53_BEST_57 = DAN_53_BEST_57$pred,
                        DAN_87_BEST_57 = DAN_87_BEST_57$pred,
                        DAN_128_BEST_57 = DAN_128_BEST_57$pred,
                        
                        REN_xgbt_v5.2_128 = REN_xgbt_v5.2_128$pred,
                        REN_xgbf_v5.2_128 = REN_xgbf_v5.2_128$pred)

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_cs0.5.tsv", sep = '\t', header = TRUE)
# dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v5.2_35.tsv", sep = '\t', header = TRUE)

# dat <- cbind(dat,stack_all)
# dat <- cbind(dat,stack_No_REN)
# dat <- cbind(dat,stack_128)

dat$MRN  <- as.character(seq(1,dim(dat)[1]))

set.seed(1)
x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)

t <- read.table(file = "/gpfs/group/asb17/default/oob_folds.tsv", sep = '\t', header = TRUE)
t <- t[order(t$ind),]
x_train.hex$fold <- as.h2o(t$fold)

## Tuning 

best_seednumber = 1234
best_test_profit = 0
best_param = list()

for (iter in 1:50) {
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  hyper_params <- list(
    activation=sample(c("Tanh"),1),
    hidden = sample(list(c(200,200,200),c(150,150,150,150),c(100,100,100),c(100,100,100,100)),1)[[1]],
    input_dropout_ratio=runif(1,0,0.05),
    l1=sample(seq(0,1e-4,1e-6),1),
    l2=sample(seq(0,1e-4,1e-6),1),
    epochs = sample(c(50,25),1))
  
  h2o.dl.stack <- h2o.deeplearning(x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","fold"))], 
                                   y= "fraud", training_frame = x_train.hex,
                                   ignore_const_cols =T, seed =  seed.number, reproducible = T,
                                   loss = "CrossEntropy",fold_column = "fold", 
                                   epochs = hyper_params$epochs,
                                   initial_weight_distribution = "UniformAdaptive",
                                   hidden = hyper_params$hidden, 
                                   distribution = "bernoulli",activation = hyper_params$activation,
                                   input_dropout_ratio = hyper_params$input_dropout_ratio,
                                   l1=hyper_params$l1, l2=hyper_params$l2,
                                   keep_cross_validation_predictions = T,
                                   missing_values_handling = "MeanImputation")
  
  (t <- table((as.vector(h2o.getFrame(h2o.dl.stack@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
              as.vector(x_train.hex$fraud)))
  
  (max_test_profit <- as.numeric(-5 * t[1,2]-25*t[2,1]+ 5*t[2,2]))
  
  
  if (max_test_profit > best_test_profit) {
    
    best_test_profit = max_test_profit
    best_seednumber = seed.number
    best_param =  hyper_params
  }
  
}

best_test_profit    
best_seednumber 
best_param


# stack all, 5/7

h2o.dl.stack_all_5_7 <- h2o.deeplearning(seed= 5660, x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","fold"))], 
                                         y= "fraud", training_frame = x_train.hex,
                                         ignore_const_cols =T, reproducible = T,
                                         fold_column = "fold", 
                                         hidden = c(200,200,200),loss = "CrossEntropy",
                                         distribution = "bernoulli", 
                                         initial_weight_distribution = "UniformAdaptive",
                                         activation = "Tanh", epochs = 50,
                                         input_dropout_ratio = 0.03809794,
                                         l1= 6.3e-05, l2= 5.6e-05,
                                         keep_cross_validation_predictions = T,
                                         missing_values_handling = "MeanImputation")

(t <- table((as.vector(h2o.getFrame(h2o.dl.stack_all_5_7@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 5/7)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])   # 475
(as.data.frame(h2o.varimp(h2o.dl.stack_all_5_7)))[1:20,"variable"]


# stack_No_REN, 5/7

h2o.dl.stack_No_REN_5_7 <- h2o.deeplearning(seed= 7651, x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","fold"))], 
                                         y= "fraud", training_frame = x_train.hex,
                                         ignore_const_cols =T, reproducible = T,
                                         fold_column = "fold", 
                                         hidden = c(200,200,200),loss = "CrossEntropy",
                                         distribution = "bernoulli", 
                                         initial_weight_distribution = "UniformAdaptive",
                                         activation = "Tanh", epochs = 50,
                                         input_dropout_ratio = 0.04586823,
                                         l1= 9.7e-05, l2= 3.8e-05,
                                         keep_cross_validation_predictions = T,
                                         missing_values_handling = "MeanImputation")

(t <- table((as.vector(h2o.getFrame(h2o.dl.stack_No_REN_5_7@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 5/7)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])   # 465
(as.data.frame(h2o.varimp(h2o.dl.stack_No_REN_5_7)))[1:25,"variable"]


# stack_128, 0.5

h2o.dl.stack_128_0.5 <- h2o.deeplearning(seed= 6609, x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","fold"))], 
                                            y= "fraud", training_frame = x_train.hex,
                                            ignore_const_cols =T, reproducible = T,
                                            fold_column = "fold", 
                                            hidden = c(200,200,200),loss = "CrossEntropy",
                                            distribution = "bernoulli", 
                                            initial_weight_distribution = "UniformAdaptive",
                                            activation = "Tanh", epochs = 50,
                                            input_dropout_ratio = 0.04111125,
                                            l1= 9.3e-05, l2= 7e-05,
                                            keep_cross_validation_predictions = T,
                                            missing_values_handling = "MeanImputation")

(t <- table((as.vector(h2o.getFrame(h2o.dl.stack_128_0.5@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 5/7)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])   # 475
(as.data.frame(h2o.varimp(h2o.dl.stack_128_0.5)))[1:25,"variable"]


