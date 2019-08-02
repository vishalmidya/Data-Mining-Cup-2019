qsub -I -A open -l nodes=1:ppn=5 -l walltime=5:00:00 -N sessionName=rf -l pmem=30gb

# setwd("D:/DMC 2019")

####################  Average Profit

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


N <- 5
Profit_5_7 <- rep(0,N);

for(i in 1:N){

dat <- read.table(file = "/gpfs/group/asb17/default/dan/train_cs0.5.tsv", sep = '\t', header = TRUE)

dat <- cbind(dat,stack_all)
# dat <- cbind(dat,stack_No_REN)
# dat <- cbind(dat,stack_128)

dat$MRN  <- as.character(seq(1,dim(dat)[1]))

set.seed(sample(seq(1,100000),1))

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

(t_5_7 <- table((as.vector(h2o.getFrame(h2o.dl.stack_all_5_7@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 5/7)+0,
            as.vector(x_train.hex$fraud)))

(Profit_5_7[i] <-  -5 * t_5_7[1,2]-25*t_5_7[2,1]+ 5*t_5_7[2,2])

# (t_0.5 <- table((as.vector(h2o.getFrame(h2o.dl.stack_128_0.5@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])[,"p1"])> 0.5)+0,
#                 as.vector(x_train.hex$fraud)))
# 
# (Profit_0.5 <- Profit_0.5 + -5 * t_0.5[1,2]-25*t_0.5[2,1]+ 5*t_0.5[2,2])


}

mean(Profit_5_7);
sd(Profit_5_7);

# Profit_0.5/N


# h2o.dl.stack_all_5_7
## Profit_5_7/N : 425.8
## Profit_0.5/N : 417

# h2o.dl.stack_No_REN_5_7
## Profit_5_7/N : 431.9
## Profit_0.5/N : 424.1

# h2o.dl.stack_128_0.5
## Profit_5_7/N : 422.3
## Profit_0.5/N : 403

# 431.2 with cs0.5 cutoff 5/7
# 424.2 with cs0.5 cutoff 0.5


# 387 with v5.2_35 cutoff 0.5
# 419 with v4.2_53 cutoff 0.5
# 394.375 with train_cs0.5 cutoff 0.5
# 364 with v4_87 cutoff 0.5 


# 470.6 with v5.2_35 cutoff MAX F1 rate from training data 
# 461.2 with train_v4.2_53 cutoff MAX F1 rate from training data 
# 470.875 with train_cs0.5 cutoff MAX F1 rate from training data 
# 464.625 with v4_87  cutoff MAX F1 rate from training data 

#  412.125 with v5.2_35 cutoff 5/7
#  396.625  with v4.2_53 cutoff 5/7
#  422.875 with train_cs0.5 cutoff 5/7
#  381 with v4_87 cutoff 5/7

