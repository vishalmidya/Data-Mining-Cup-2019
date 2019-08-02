qsub -I -A open -l nodes=1:ppn=8 -l walltime=4:00:00 -N sessionName=rf -l pmem=50gb
module load gcc

# setwd("D:/DMC 2019")

library(h2o)
h2o.init(nthreads=-1, max_mem_size="20G")

## NN for train_v4.2_53

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v4.2_53.tsv", sep = '\t', header = TRUE)

dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$col_weight <- ifelse(dat$fraud==0,25,10)

set.seed(142)

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
x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)

h2o.dl.440.train_v4.2_53 <- h2o.deeplearning(seed= 2725,
                                             x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                             y= "fraud", training_frame = x_train.hex,
                                             ignore_const_cols =T, reproducible = T,
                                             fold_column = "fold", 
                                             weights_column = "col_weight",
                                             hidden = c(100,100,100,99),
                                             loss = "CrossEntropy",
                                             distribution = "bernoulli", 
                                             initial_weight_distribution = "UniformAdaptive",
                                             activation = "Tanh", epochs = 25,
                                             input_dropout_ratio = 0.03031014,
                                             l1= 4.5e-05, l2= 8.4e-05,
                                             keep_cross_validation_predictions = T,
                                             missing_values_handling = "MeanImputation")

t <- table(as.vector(h2o.getFrame(h2o.dl.440.train_v4.2_53@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
           as.vector(x_train.hex$fraud))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 440

(as.data.frame(h2o.varimp(h2o.dl.440.train_v4.2_53)))[,"variable"]


### OOB preds

# w <- read.table(file = "/gpfs/group/asb17/default/oob_folds.tsv", sep = '\t', header = TRUE)
# w <- w[order(w$ind),]
# x_train.hex$fold <- as.h2o(t$fold)

h2o.dl.440.train_v4.2_53.oob <- h2o.deeplearning(seed= 2725,
                                             x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                             y= "fraud", training_frame = x_train.hex,
                                             ignore_const_cols =T, reproducible = T,
                                             fold_column = "fold", 
                                             weights_column = "col_weight",
                                             hidden = c(100,100,100,99),
                                             loss = "CrossEntropy",
                                             distribution = "bernoulli", 
                                             initial_weight_distribution = "UniformAdaptive",
                                             activation = "Tanh", epochs = 25,
                                             input_dropout_ratio = 0.03031014,
                                             l1= 4.5e-05, l2= 8.4e-05,
                                             keep_cross_validation_predictions = T,
                                             missing_values_handling = "MeanImputation")

t <- table(as.vector(h2o.getFrame(h2o.dl.440.train_v4.2_53.oob@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
           as.vector(x_train.hex$fraud))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 440


cvpreds_id <- h2o.dl.440.train_v4.2_53.oob@model$cross_validation_holdout_predictions_frame_id$name
cvpreds <- h2o.getFrame(cvpreds_id)
oob.h2o.dl.440.train_v4.2_53 <- as.data.frame(cvpreds)
oob.h2o.dl.440.train_v4.2_53$rowIndex <- seq(1,length(oob.h2o.dl.440.train_v4.2_53$predict))
oob.h2o.dl.440.train_v4.2_53$pred <- oob.h2o.dl.440.train_v4.2_53$p1

write.table(oob.h2o.dl.440.train_v4.2_53, 
            file="/gpfs/group/asb17/default/renan/oob/OOB_NN_OLD_v4.2_53.tsv", quote=T, sep='\t')

### Test Preds

dat_test <- read.table(file = "/gpfs/group/asb17/default/dan/test_v4.2_53.tsv", sep = '\t', header = TRUE)
x_test_deep = as.data.frame(data.matrix(dat_test))
x_test.hex <- as.h2o(x_test_deep)

test_pred <- as.data.frame(h2o.predict(h2o.dl.440.train_v4.2_53.oob, x_test.hex))


test_pred$pred <- test_pred$p1
write.table(test_pred, file="/gpfs/group/asb17/default/renan/testpreds/testpred_NN_OLD_v4.2_53.tsv", quote=T, sep='\t')

############################

# data v4_87
## NN (Profit 440)

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v4_87.tsv", sep = '\t', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$col_weight <- ifelse(dat$fraud==0,25,10)

set.seed(3)
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
x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)

h2o.dl.440.v4_87 <- h2o.deeplearning(seed= 3382,
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

(t <- table((as.vector(h2o.getFrame(h2o.dl.440.v4_87@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  ## 355


### OOB preds

# w <- read.table(file = "/gpfs/group/asb17/default/oob_folds.tsv", sep = '\t', header = TRUE)
# w <- w[order(w$ind),]
# x_train.hex$fold <- as.h2o(w$fold)

h2o.dl.440.v4_87.oob <- h2o.deeplearning(seed= 3382,
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

(t <- table((as.vector(h2o.getFrame(h2o.dl.440.v4_87.oob@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  ## 355


cvpreds_id <- h2o.dl.440.v4_87.oob@model$cross_validation_holdout_predictions_frame_id$name
cvpreds <- h2o.getFrame(cvpreds_id)
oob.h2o.dl.440.v4_87 <- as.data.frame(cvpreds)
oob.h2o.dl.440.v4_87$rowIndex <- seq(1,length(oob.h2o.dl.440.v4_87$predict))
oob.h2o.dl.440.v4_87$pred <- oob.h2o.dl.440.v4_87$p1

write.table(oob.h2o.dl.440.v4_87, 
            file="/gpfs/group/asb17/default/renan/oob/OOB_NN_v4_87.tsv", quote=T, sep='\t')


### Test Preds

dat_test <- read.table(file = "/gpfs/group/asb17/default/dan/test_v4_87.tsv", 
                       sep = '\t', header = TRUE)
x_test_deep = as.data.frame(data.matrix(dat_test))
x_test.hex <- as.h2o(x_test_deep)

test_pred <- as.data.frame(h2o.predict(h2o.dl.440.v4_87.oob, x_test.hex))
test_pred$pred <- test_pred$p1
write.table(test_pred, file="/gpfs/group/asb17/default/renan/testpreds/testpred_NN_v4_87.tsv", quote=T, sep='\t')


###################################### train_cs0.5

library(h2o)
h2o.init(nthreads=-1, max_mem_size="20G")

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_cs0.5.tsv", sep = '\t', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))
dat$col_weight <- ifelse(dat$fraud==0,25,10)

set.seed(100)

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

### OOB preds

# w <- read.table(file = "/gpfs/group/asb17/default/oob_folds.tsv", sep = '\t', header = TRUE)
# w <- w[order(w$ind),]
# x_train.hex$fold <- as.h2o(t$fold)

h2o.dl.420.train_cs0.5.oob <- h2o.deeplearning(seed= 3003,
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

t <- table(as.vector(h2o.getFrame(h2o.dl.420.train_cs0.5.oob@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
           as.vector(x_train.hex$fraud))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 420


cvpreds_id <- h2o.dl.420.train_cs0.5.oob@model$cross_validation_holdout_predictions_frame_id$name
cvpreds <- h2o.getFrame(cvpreds_id)
oob.h2o.dl.420.train_cs0.5 <- as.data.frame(cvpreds)
oob.h2o.dl.420.train_cs0.5$rowIndex <- seq(1,length(oob.h2o.dl.420.train_cs0.5$predict))
oob.h2o.dl.420.train_cs0.5$pred <- oob.h2o.dl.420.train_cs0.5$p1

write.table(oob.h2o.dl.420.train_cs0.5, 
            file="/gpfs/group/asb17/default/renan/oob/OOB_NN_train_cs0.5.tsv", quote=T, sep='\t')


### Test Preds

dat_test <- read.table(file = "/gpfs/group/asb17/default/dan/test_cs0.5.tsv", 
                       sep = '\t', header = TRUE)
x_test_deep = as.data.frame(data.matrix(dat_test))
x_test.hex <- as.h2o(x_test_deep)

test_pred <- as.data.frame(h2o.predict(h2o.dl.420.train_cs0.5.oob, x_test.hex))
test_pred$pred <- test_pred$p1
write.table(test_pred, file="/gpfs/group/asb17/default/renan/testpreds/testpred_NN_cs0.5.tsv", quote=T, sep='\t')


###################################### train_v5.2_128.tsv

library(h2o)
h2o.init(nthreads=-1, max_mem_size="20G")


dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v5.2_128.tsv", sep = '\t', header = TRUE)
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

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)
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

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 420

### OOB preds

# w <- read.table(file = "/gpfs/group/asb17/default/oob_folds.tsv", sep = '\t', header = TRUE)
# w <- w[order(w$ind),]
# x_train.hex$fold <- as.h2o(t$fold)

h2o.dl.400.train_v5.2_128.oob <- h2o.deeplearning(seed= 7933,
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

t <- table(as.vector(h2o.getFrame(h2o.dl.400.train_v5.2_128.oob@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
           as.vector(x_train.hex$fraud))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 340


cvpreds_id <- h2o.dl.400.train_v5.2_128.oob@model$cross_validation_holdout_predictions_frame_id$name
cvpreds <- h2o.getFrame(cvpreds_id)
oob.h2o.dl.400.train_v5.2_128 <- as.data.frame(cvpreds)
oob.h2o.dl.400.train_v5.2_128$rowIndex <- seq(1,length(oob.h2o.dl.400.train_v5.2_128$predict))
oob.h2o.dl.400.train_v5.2_128$pred <- oob.h2o.dl.400.train_v5.2_128$p1

write.table(oob.h2o.dl.400.train_v5.2_128, 
            file="/gpfs/group/asb17/default/renan/oob/OOB_NN_train_v5.2_128.tsv", quote=T, sep='\t')


### Test Preds

dat_test <- read.table(file = "/gpfs/group/asb17/default/dan/test_v5.2_128.tsv", 
                       sep = '\t', header = TRUE)
x_test_deep = as.data.frame(data.matrix(dat_test))
x_test.hex <- as.h2o(x_test_deep)

test_pred <- as.data.frame(h2o.predict(h2o.dl.400.train_v5.2_128.oob, x_test.hex))
test_pred$pred <- test_pred$p1
write.table(test_pred, file="/gpfs/group/asb17/default/renan/testpreds/testpred_NN_v5.2_128.tsv", quote=T, sep='\t')



### NN for train_v4.2_53

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v4.2_53.tsv", sep = '\t', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))

set.seed(1492)

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
x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)

h2o.dl.365.train_v4.2_53 <- h2o.deeplearning(seed= 9735,
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

(t <- table((as.vector(h2o.getFrame(h2o.dl.365.train_v4.2_53@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 365


### OOB preds

# w <- read.table(file = "/gpfs/group/asb17/default/oob_folds.tsv", sep = '\t', header = TRUE)
# w <- w[order(w$ind),]
# x_train.hex$fold <- as.h2o(t$fold)

h2o.dl.365.train_v4.2_53.oob <- h2o.deeplearning(seed= 9735,
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

t <- table(as.vector(h2o.getFrame(h2o.dl.365.train_v4.2_53.oob@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"predict"]),
           as.vector(x_train.hex$fraud))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 365



cvpreds_id <- h2o.dl.365.train_v4.2_53.oob@model$cross_validation_holdout_predictions_frame_id$name
cvpreds <- h2o.getFrame(cvpreds_id)
oob.h2o.dl.365.train_v4.2_53 <- as.data.frame(cvpreds)
oob.h2o.dl.365.train_v4.2_53$rowIndex <- seq(1,length(oob.h2o.dl.365.train_v4.2_53$predict))
oob.h2o.dl.365.train_v4.2_53$pred <- oob.h2o.dl.365.train_v4.2_53$p1

write.table(oob.h2o.dl.365.train_v4.2_53, 
            file="/gpfs/group/asb17/default/renan/oob/OOB_NN_NEW_v4.2_53.tsv", quote=T, sep='\t')


### Test Preds

dat_test <- read.table(file = "/gpfs/group/asb17/default/dan/test_v4.2_53.tsv", sep = '\t', header = TRUE)
x_test_deep = as.data.frame(data.matrix(dat_test))
x_test.hex <- as.h2o(x_test_deep)

test_pred <- as.data.frame(h2o.predict(h2o.dl.365.train_v4.2_53.oob, x_test.hex))
test_pred$pred <- test_pred$p1
write.table(test_pred, file="/gpfs/group/asb17/default/renan/testpreds/testpred_NN_NEW_v4.2_53.tsv", quote=T, sep='\t')


#####################################3

## NN for train_v5.2_35

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v5.2_35.tsv", sep = '\t', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))

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

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)
x_train.hex$col_weight <- ifelse(x_train.hex$fraud=="0",50,20)

h2o.dl.350.train_v5.2_35 <- h2o.deeplearning(seed= 5730,
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

(t <- table((as.vector(h2o.getFrame(h2o.dl.350.train_v5.2_35@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 355


### OOB preds

# w <- read.table(file = "/gpfs/group/asb17/default/oob_folds.tsv", sep = '\t', header = TRUE)
# w <- w[order(w$ind),]
# x_train.hex$fold <- as.h2o(t$fold)

h2o.dl.350.train_v5.2_35.oob <- h2o.deeplearning(seed= 5730,
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

(t <- table((as.vector(h2o.getFrame(h2o.dl.350.train_v5.2_35.oob@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 350

cvpreds_id <- h2o.dl.350.train_v5.2_35.oob@model$cross_validation_holdout_predictions_frame_id$name
cvpreds <- h2o.getFrame(cvpreds_id)
oob.h2o.dl.350.train_v5.2_35 <- as.data.frame(cvpreds)
oob.h2o.dl.350.train_v5.2_35$rowIndex <- seq(1,length(oob.h2o.dl.350.train_v5.2_35$predict))
oob.h2o.dl.350.train_v5.2_35$pred <- oob.h2o.dl.350.train_v5.2_35$p1

write.table(oob.h2o.dl.350.train_v5.2_35, 
            file="/gpfs/group/asb17/default/renan/oob/OOB_NN_v5.2_35.tsv", quote=T, sep='\t')

### Test Preds

dat_test <- read.table(file = "/gpfs/group/asb17/default/dan/test_v5.2_35.tsv", sep = '\t', header = TRUE)
x_test_deep = as.data.frame(data.matrix(dat_test))
x_test.hex <- as.h2o(x_test_deep)

test_pred <- as.data.frame(h2o.predict(h2o.dl.350.train_v5.2_35.oob, x_test.hex))
test_pred$pred <- test_pred$p1
write.table(test_pred, file="/gpfs/group/asb17/default/renan/testpreds/testpred_NN_v5.2_35.tsv", quote=T, sep='\t')

# r <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_NN_350_v5.2_35.tsv", sep = '\t', header = TRUE)

################# BART train_v4.2_53

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

### OOB preds

bart_v4.2_53.oob <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                          y = dat$fraud, verbose = F,use_missing_data=F,
                          folds_vec = as.integer(dat$fold), num_trees = 100, 
                          k = 1, prob_rule_class = 0.5, seed = 2655)

t <- t(bart_v4.2_53.oob$confusion_matrix[1:2,1:2])
(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 225

oob.bart_v4.2_53 <- data.frame(pred = 1 - bart_v4.2_53.oob$phat)
oob.bart_v4.2_53$rowIndex <- seq(1,length(oob.bart_v4.2_53$pred))

write.table(oob.bart_v4.2_53, 
            file="/gpfs/group/asb17/default/renan/oob/OOB_BART_v4.2_53.tsv", quote=T, sep='\t')


### Test Preds

bart_v4.2_53.full <- bartMachine(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                                 y = dat$fraud, verbose = F,use_missing_data=F,
                                 num_trees = 100, k = 2, prob_rule_class = 0.5, seed = 2655)

dat_test <- read.table(file = "/gpfs/group/asb17/default/dan/test_v4.2_53.tsv", sep = '\t', header = TRUE)
x_test_deep = as.data.frame((dat_test))

test_pred <- predict(bart_v4.2_53.full, x_test_deep[,!(colnames(x_test_deep) %in% c("fraud"))], prob_rule_class = 0.5, type = "prob")
test_pred <- data.frame(pred = 1- test_pred)

write.table(test_pred, file="/gpfs/group/asb17/default/renan/testpreds/testpred_BART_v4.2_53.tsv", quote=T, sep='\t')


################# BART train_v4_87

options(java.parameters = "-Xmx20000m")
library("bartMachine")

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v4_87.tsv", sep = '\t', header = TRUE)
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

### OOB preds

bart_v4_87.oob <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                              y = dat$fraud, verbose = F,use_missing_data=F,
                              folds_vec = as.integer(dat$fold), num_trees = 50, 
                              k = 1, prob_rule_class = 0.5, seed = 5399)

t <- t(bart_v4_87.oob$confusion_matrix[1:2,1:2])
(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 200

oob.bart_v4_87 <- data.frame(pred = 1 - bart_v4_87.oob$phat)
oob.bart_v4_87$rowIndex <- seq(1,length(oob.bart_v4_87$pred))

write.table(oob.bart_v4_87, 
            file="/gpfs/group/asb17/default/renan/oob/OOB_BART_v4_87.tsv", quote=T, sep='\t')


### Test Preds

bart_v4_87.full <- bartMachine(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                                 y = dat$fraud, verbose = F,use_missing_data=F,
                                 num_trees = 50, k = 1, prob_rule_class = 0.5, seed = 5399)

dat_test <- read.table(file = "/gpfs/group/asb17/default/dan/test_v4_87.tsv", sep = '\t', header = TRUE)
x_test_deep = as.data.frame((dat_test))

test_pred <- predict(bart_v4_87.full, x_test_deep[,!(colnames(x_test_deep) %in% c("fraud"))], prob_rule_class = 0.5, type = "prob")
test_pred <- data.frame(pred = 1- test_pred)

write.table(test_pred, file="/gpfs/group/asb17/default/renan/testpreds/testpred_BART_v4_87.tsv", quote=T, sep='\t')


## NN for train initial

dat <- read.table(file = "/gpfs/group/asb17/default/DMC2019/task/train.csv", sep = '|', header = TRUE)
dat$MRN  <- as.character(seq(1,dim(dat)[1]))

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

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)
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
                                         activation = "Tanh", epochs = 25,
                                         input_dropout_ratio = 0.003407305,
                                         l1= 3.4e-06, l2= 5.3e-05,
                                         keep_cross_validation_predictions = T,
                                         missing_values_handling = "MeanImputation")

(t <- table((as.vector(h2o.getFrame(h2o.dl.train_initial@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  


### OOB preds

# w <- read.table(file = "/gpfs/group/asb17/default/oob_folds.tsv", sep = '\t', header = TRUE)
# w <- w[order(w$ind),]
# x_train.hex$fold <- as.h2o(t$fold)

h2o.dl.train_initial.oob <- h2o.deeplearning(seed= 3339,
                                             x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","col_weight","fold"))], 
                                             y= "fraud", training_frame = x_train.hex,
                                             ignore_const_cols =T, reproducible = T,
                                             fold_column = "fold", 
                                             weights_column = "col_weight",
                                             hidden = c(100,100,100),loss = "CrossEntropy",
                                             distribution = "bernoulli", 
                                             initial_weight_distribution = "UniformAdaptive",
                                             activation = "Tanh", epochs = 25,
                                             input_dropout_ratio = 0.003407305,
                                             l1= 3.4e-05, l2= 5.3e-05,
                                             keep_cross_validation_predictions = T,
                                             missing_values_handling = "MeanImputation")

(t <- table((as.vector(h2o.getFrame(h2o.dl.train_initial.oob@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 0

cvpreds_id <- h2o.dl.train_initial.oob@model$cross_validation_holdout_predictions_frame_id$name
cvpreds <- h2o.getFrame(cvpreds_id)
oob.h2o.dl.train_initial <- as.data.frame(cvpreds)
oob.h2o.dl.train_initial$rowIndex <- seq(1,length(oob.h2o.dl.train_initial$predict))
oob.h2o.dl.train_initial$pred <- oob.h2o.dl.train_initial$p1

write.table(oob.h2o.dl.train_initial, 
            file="/gpfs/group/asb17/default/renan/oob/OOB_NN_initial_fulldata.tsv", quote=T, sep='\t')

### Test Preds

dat_test <- read.table(file = "/gpfs/group/asb17/default/DMC2019/task/test.csv", sep = '|', header = TRUE)
x_test_deep = as.data.frame(data.matrix(dat_test))
x_test.hex <- as.h2o(x_test_deep)

test_pred <- as.data.frame(h2o.predict(h2o.dl.train_initial.oob, x_test.hex))
test_pred$pred <- test_pred$p1

write.table(test_pred, file="/gpfs/group/asb17/default/renan/testpreds/testpred_NN_initial_fulldata.tsv", quote=T, sep='\t')


################# BART train_v5.2_35

options(java.parameters = "-Xmx20000m")
library("bartMachine")

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v5.2_35.tsv", sep = '\t', header = TRUE)
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

### OOB preds

bart_v5.2_35.oob <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                            y = dat$fraud, verbose = F,use_missing_data=F,
                            folds_vec = as.integer(dat$fold), num_trees = 100, 
                            k = 1, prob_rule_class = 0.5, seed = 4923)

t <- t(bart_v5.2_35.oob$confusion_matrix[1:2,1:2])
(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 215

oob.bart_v5.2_35 <- data.frame(pred = 1 - bart_v5.2_35.oob$phat)
oob.bart_v5.2_35$rowIndex <- seq(1,length(oob.bart_v5.2_35$pred))

write.table(oob.bart_v5.2_35, 
            file="/gpfs/group/asb17/default/renan/oob/OOB_BART_v5.2_35.tsv", quote=T, sep='\t')


### Test Preds

bart_v5.2_35.full <- bartMachine(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                               y = dat$fraud, verbose = F,use_missing_data=F,
                               num_trees = 100, k = 1, prob_rule_class = 0.5, seed = 4923)

dat_test <- read.table(file = "/gpfs/group/asb17/default/dan/test_v5.2_35.tsv", sep = '\t', header = TRUE)
x_test_deep = as.data.frame((dat_test))

test_pred <- predict(bart_v5.2_35.full, x_test_deep[,!(colnames(x_test_deep) %in% c("fraud"))],
                     prob_rule_class = 0.5, type = "prob")

test_pred <- data.frame(pred = 1- test_pred)

write.table(test_pred, file="/gpfs/group/asb17/default/renan/testpreds/testpred_BART_v5.2_35.tsv", quote=T, sep='\t')



################# BART train_v5.2_128

options(java.parameters = "-Xmx20000m")
library("bartMachine")

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_v5.2_128.tsv", sep = '\t', header = TRUE)
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

### OOB preds

bart_v5.2_128.oob <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                              y = dat$fraud, verbose = F,use_missing_data=F,
                              folds_vec = as.integer(dat$fold), num_trees = 200, 
                              k = 2, prob_rule_class = 0.5, seed = 1896)

t <- t(bart_v5.2_128.oob$confusion_matrix[1:2,1:2])
(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 225


oob.bart_v5.2_128 <- data.frame(pred = 1 - bart_v5.2_128.oob$phat)
oob.bart_v5.2_128$rowIndex <- seq(1,length(oob.bart_v5.2_128$pred))

write.table(oob.bart_v5.2_128, 
            file="/gpfs/group/asb17/default/renan/oob/OOB_BART_v5.2_128.tsv", quote=T, sep='\t')


################# # BART v0

options(java.parameters = "-Xmx20000m")
library("bartMachine")

dat <- read.table(file = "/gpfs/group/asb17/default/DMC2019/task/train.csv", sep = '|', header = TRUE)
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

### OOB preds

bart_v0.oob <- k_fold_cv(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                               y = dat$fraud, verbose = F,use_missing_data=F,
                               folds_vec = as.integer(dat$fold), num_trees = 50, 
                               k = 2, prob_rule_class = 0.5, seed = 8170)

t <- t(bart_v0.oob$confusion_matrix[1:2,1:2])
(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # 85


oob.bart_v0 <- data.frame(pred = 1 - bart_v0.oob$phat)
oob.bart_v0$rowIndex <- seq(1,length(oob.bart_v0$pred))

write.table(oob.bart_v0, 
            file="/gpfs/group/asb17/default/renan/oob/OOB_BART_v0.tsv", quote=T, sep='\t')


### Test Preds

bart_v0.full <- bartMachine(X = as.data.frame(dat[,!(colnames(dat) %in% c("fraud","MRN","fold"))]), 
                                 y = dat$fraud, verbose = F,use_missing_data=F,
                                 num_trees = 50, k = 2, prob_rule_class = 0.5, seed = 8170)

dat_test <- read.table(file = "/gpfs/group/asb17/default/DMC2019/task/test.csv", sep = '|', header = TRUE)
x_test_deep = as.data.frame((dat_test))

test_pred <- predict(bart_v0.full, x_test_deep[,!(colnames(x_test_deep) %in% c("fraud"))],
                     prob_rule_class = 0.5, type = "prob")

test_pred <- data.frame(pred = 1- test_pred)

write.table(test_pred, file="/gpfs/group/asb17/default/renan/testpreds/testpred_BART_v0.tsv", quote=T, sep='\t')


####################
# Stacking

## OOBS

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


dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_cs0.5.tsv", sep = '\t', header = TRUE)

dat <- cbind(dat,stack_all)
# dat <- cbind(dat,stack_No_REN)

set.seed(1)

x_train_deep = as.data.frame(data.matrix(dat))
x_train.hex <- as.h2o(x_train_deep)
x_train.hex$fraud <- as.factor(x_train.hex$fraud)

t <- read.table(file = "/gpfs/group/asb17/default/oob_folds.tsv", sep = '\t', header = TRUE)
t <- t[order(t$ind),]
x_train.hex$fold <- as.h2o(t$fold)

# ALL SUBMODELS
h2o.dl.stack <- h2o.deeplearning(seed= 5660, x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","fold"))],
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


# ## No_REN MODELS
# 
# h2o.dl.stack <- h2o.deeplearning(seed= 7651, x= colnames(dat)[!(colnames(dat) %in% c("fraud","MRN","fold"))], 
#                                  y= "fraud", training_frame = x_train.hex,
#                                  ignore_const_cols =T, reproducible = T,
#                                  fold_column = "fold", 
#                                  hidden = c(200,200,200),loss = "CrossEntropy",
#                                  distribution = "bernoulli", 
#                                  initial_weight_distribution = "UniformAdaptive",
#                                  activation = "Tanh", epochs = 50,
#                                  input_dropout_ratio = 0.04586823,
#                                  l1= 9.7e-05, l2= 3.8e-05,
#                                  keep_cross_validation_predictions = T,
#                                  missing_values_handling = "MeanImputation")


(t <- table((as.vector(h2o.getFrame(h2o.dl.stack@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 5/7)+0,
            as.vector(x_train.hex$fraud)))

(Profit <- -5 * t[1,2]-25*t[2,1]+ 5*t[2,2])  # avg: 425.8 (all submodels)  # avg: 431.9 (No_REN)

(as.data.frame(h2o.varimp(h2o.dl.stack)))[1:25,"variable"]

cvpreds_id <- h2o.dl.stack@model$cross_validation_holdout_predictions_frame_id$name
cvpreds <- h2o.getFrame(cvpreds_id)
oob.h2o.dl.stack <- as.data.frame(cvpreds)
oob.h2o.dl.stack$rowIndex <- seq(1,length(oob.h2o.dl.stack$predict))
oob.h2o.dl.stack$true_fraud <- dat$fraud

table((oob.h2o.dl.stack$p1>5/7)+0, oob.h2o.dl.stack$true_fraud)

oob.h2o.dl.stack$p1[(oob.h2o.dl.stack$p1>5/7)+0 == 0 & oob.h2o.dl.stack$true_fraud == 1]
oob.h2o.dl.stack$p1[(oob.h2o.dl.stack$p1>5/7)+0 == 1 & oob.h2o.dl.stack$true_fraud == 0]


## TEST preds

NN_128 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_NN_v5.2_128.tsv", sep = '\t', header = TRUE)
NN_CS0.5 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_NN_cs0.5.tsv", sep = '\t', header = TRUE)
NN_53_OLD <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_NN_OLD_v4.2_53.tsv", sep = '\t', header = TRUE)
NN_87 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_NN_v4_87.tsv", sep = '\t', header = TRUE)
NN_35 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_NN_v5.2_35.tsv", sep = '\t', header = TRUE)   
NN_53_NEW <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_NN_NEW_v4.2_53.tsv", sep = '\t', header = TRUE)   
NN_INITIAL <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_NN_initial_fulldata.tsv", sep = '\t', header = TRUE)   

BART_V0 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_BART_v0.tsv", sep = '\t', header = TRUE)   
BART_53 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_BART_v4.2_53.tsv", sep = '\t', header = TRUE)   
BART_87 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_BART_v4_87.tsv", sep = '\t', header = TRUE)   
BART_128 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_BART_v5.2_128.tsv", sep = '\t', header = TRUE)   
BART_35 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_BART_v5.2_35.tsv", sep = '\t', header = TRUE)   

LOGISTIC <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_logistic_v0.tsv", sep = '\t', header = TRUE)   
DAN_53_BEST_57 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_dan_train_v4.2_53.tsv.best.oob57.tsv", sep = '\t', header = TRUE)   
DAN_87_BEST_57 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_dan_train_v4_87.tsv.best.oob57.tsv", sep = '\t', header = TRUE)   
DAN_35_BEST_57 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_dan_train_v5.2_35.tsv.best.oob57.tsv", sep = '\t', header = TRUE)   
DAN_0.5_BEST_57 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_dan_train_cs0.5.tsv.best.oob57.tsv", sep = '\t', header = TRUE)   
DAN_128_BEST_57 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_dan_train_v5.2_128.tsv.best.oob57.tsv", sep = '\t', header = TRUE)   
DAN_V0_BEST_57 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_dan_train.csv.best.oob57.tsv", sep = '\t', header = TRUE)   


REN_xgbt_v0 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_xgbt_v0.tsv", sep = '\t', header = TRUE)   
REN_xgbt_v5.2_128 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_xgbt_v5.2_128.tsv", sep = '\t', header = TRUE)   
REN_xgbt_v4_87 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_xgbt_v4_87.tsv", sep = '\t', header = TRUE)   
REN_xgbt_v4.2_53 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_xgbt_v4.2_53.tsv", sep = '\t', header = TRUE)   
REN_xgbt_cs0.5 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_xgbt_cs0.5.tsv", sep = '\t', header = TRUE)   
REN_xgbt_v5.2_35 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_xgbt_v5.2_35.tsv", sep = '\t', header = TRUE)   

REN_xgbf_v0 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_xgbf_v0.tsv", sep = '\t', header = TRUE)   
REN_xgbf_v5.2_128 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_xgbf_v5.2_128.tsv", sep = '\t', header = TRUE)   
REN_xgbf_v4_87 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_xgbf_v4_87.tsv", sep = '\t', header = TRUE)   
REN_xgbf_v4.2_53 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_xgbf_v4.2_53.tsv", sep = '\t', header = TRUE)   
REN_xgbf_cs0.5 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_xgbf_cs0.5.tsv", sep = '\t', header = TRUE)   
REN_xgbf_v5.2_35 <- read.table(file = "/gpfs/group/asb17/default/renan/testpreds/testpred_xgbf_v5.2_35.tsv", sep = '\t', header = TRUE)   



stack_all_testpred <- data.frame(NN_128 = NN_128$pred,NN_CS0.5 = NN_CS0.5$pred,
                                 NN_53_OLD = NN_53_OLD$pred,NN_87 = NN_87$pred, 
                                 NN_35 = NN_35$pred,NN_53_NEW = NN_53_NEW$pred,
                                 NN_INITIAL = NN_INITIAL$pred, 
                    
                                 BART_V0 = BART_V0$pred, BART_53 = BART_53$pred, 
                                 BART_87 = BART_87$pred, BART_128 = BART_128$pred,
                                 BART_35 = BART_35$pred,
                    
                    
                                 LOGISTIC = LOGISTIC$pred, DAN_53_BEST_57 = DAN_53_BEST_57$pred,
                                 DAN_87_BEST_57 = DAN_87_BEST_57$pred,
                                 DAN_35_BEST_57 = DAN_35_BEST_57$pred,
                                 DAN_0.5_BEST_57 = DAN_0.5_BEST_57$pred, 
                                 DAN_128_BEST_57 = DAN_128_BEST_57$pred,
                                 DAN_V0_BEST_57 = DAN_V0_BEST_57$pred,
                    
                                 REN_xgbt_v0 = REN_xgbt_v0$xgbt_v0,
                    REN_xgbt_v5.2_128 = REN_xgbt_v5.2_128$xgbt_v5.2_128,
                    REN_xgbt_v4_87 = REN_xgbt_v4_87$xgbt_v4_87,
                    REN_xgbt_v4.2_53 = REN_xgbt_v4.2_53$xgbt_v4.2_53, 
                    REN_xgbt_cs0.5 = REN_xgbt_cs0.5$xgbt_cs0.5,
                    REN_xgbt_v5.2_35 = REN_xgbt_v5.2_35$xgbt_v5.2_35,REN_xgbf_v0 = REN_xgbf_v0$xgbf_v0,
                    REN_xgbf_v5.2_128 = REN_xgbf_v5.2_128$xgbf_v5.2_128,
                    REN_xgbf_v4_87 = REN_xgbf_v4_87$xgbf_v4_87,
                    REN_xgbf_v4.2_53 = REN_xgbf_v4.2_53$xgbf_v4.2_53, 
                    REN_xgbf_cs0.5 = REN_xgbf_cs0.5$xgbf_cs0.5,
                    REN_xgbf_v5.2_35 = REN_xgbf_v5.2_35$xgbf_v5.2_35)


# stack_No_REN_testpred <- data.frame(NN_128 = NN_128$pred,NN_CS0.5 = NN_CS0.5$pred,
#                                  NN_53_OLD = NN_53_OLD$pred,NN_87 = NN_87$pred, 
#                                  NN_35 = NN_35$pred,NN_53_NEW = NN_53_NEW$pred,
#                                  NN_INITIAL = NN_INITIAL$pred, 
#                                  
#                                  BART_V0 = BART_V0$pred, BART_53 = BART_53$pred, 
#                                  BART_87 = BART_87$pred, BART_128 = BART_128$pred,
#                                  BART_35 = BART_35$pred,
#                                  
#                                  
#                                  LOGISTIC = LOGISTIC$pred, DAN_53_BEST_57 = DAN_53_BEST_57$pred,
#                                  DAN_87_BEST_57 = DAN_87_BEST_57$pred,
#                                  DAN_35_BEST_57 = DAN_35_BEST_57$pred,
#                                  DAN_0.5_BEST_57 = DAN_0.5_BEST_57$pred, 
#                                  DAN_128_BEST_57 = DAN_128_BEST_57$pred,
#                                  DAN_V0_BEST_57 = DAN_V0_BEST_57$pred)


dat_test <- as.data.frame(read.table(file = "/gpfs/group/asb17/default/dan/test_cs0.5.tsv", sep = '\t', header = TRUE))
dat_test <- cbind(dat_test,stack_all_testpred)

x_test_deep = as.data.frame(data.matrix(dat_test))
x_test.hex <- as.h2o(x_test_deep)

stack_test_pred <- as.data.frame(h2o.predict(h2o.dl.stack, x_test.hex))

stack_test_pred$rowIndex <- seq(1,dim(stack_test_pred)[1])
stack_test_pred$trustLevel <- dat_test$trustLevel
stack_test_pred$fraud <- (stack_test_pred$p1>5/7)+0
stack_test_pred <- stack_test_pred[,!(colnames(stack_test_pred) %in% c("predict"))]

table(stack_test_pred$trustLevel, stack_test_pred$fraud) 
sum(stack_test_pred$fraud)/dim(stack_test_pred)[1]  # 0.05828303

stack_test_pred$p1[stack_test_pred$trustLevel > 0 & stack_test_pred$fraud == 1]
stack_test_pred$p1[stack_test_pred$trustLevel == -0.294479333041493 & stack_test_pred$fraud == 1]


# for(i in 1:dim(stack_test_pred)[1]){
#   if(stack_test_pred$trustLevel[i] >-0.88006968342905 & stack_test_pred$fraud[i]== 1) 
#     stack_test_pred$fraud[i] = 0
# }

round(table(stack_test_pred$trustLevel, stack_test_pred$fraud)/rowSums(table(stack_test_pred$trustLevel, stack_test_pred$fraud)),4)


# write.table(stack_test_pred, file="/gpfs/group/asb17/default/renan/stackpreds/stackpreds_NN_allsubmodel.tsv", quote=T, sep='\t')

