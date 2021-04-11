frequent_patter<-csapde(traffic.train.parameter-list(suport-0.5))
library(bnlearn)

#data("lizards")
#lizards = read.table("lizards.txt",header = True)
#str(lizards)
#summary(lizards)

'''

用爬山算法
data(asia)


score(hc(asia),asia)

'''
# load dataset 

# cancer dataset

load('D:/paper/BayesianNetwork/R bayesian/cancer.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/cancer.csv')

# survey dataset

load('D:/paper/BayesianNetwork/R bayesian/survey.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/survey.csv')

# aisa dataset
load('D:/paper/BayesianNetwork/R bayesian/asia.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/asia.csv')

# insurance dataset
load('D:/paper/BayesianNetwork/R bayesian/insurance.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/insurance.csv')

# mildew dataset
load('D:/paper/BayesianNetwork/R bayesian/mildew.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/mildew.csv')

# alarm dataset
load('D:/paper/BayesianNetwork/R bayesian/alarm.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/alarm.csv')

# hailfinder dataset
load('D:/paper/BayesianNetwork/R bayesian/hailfinder.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/hailfinder.csv')

# hepar2 dataset
load('D:/paper/BayesianNetwork/R bayesian/hepar2.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/hepar2.csv')

# win95pts dataset
load('D:/paper/BayesianNetwork/R bayesian/win95pts.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/win95pts.csv')

# andes dataset
load('D:/paper/BayesianNetwork/R bayesian/andes.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/andes.csv')

# diabetes dataset
load('D:/paper/BayesianNetwork/R bayesian/diabetes.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/diabetes.csv')

# pigs dataset
load('D:/paper/BayesianNetwork/R bayesian/pigs.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/pigs.csv')


# sachs
load('D:/paper/BayesianNetwork/R bayesian/sachs.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/sachs.csv')

# earthquake
load('D:/paper/BayesianNetwork/R bayesian/earthquake.rda')
data = rbn(bn,100000)
write.csv(data,file = 'D:/paper/BayesianNetwork/R bayesian/earthquake.csv')
