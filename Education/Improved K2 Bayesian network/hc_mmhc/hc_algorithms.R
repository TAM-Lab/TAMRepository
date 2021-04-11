# hill-climbing 绠楁硶
library(bnlearn)

# asia
asias = read.csv('D:/paper/BayesianNetwork/R bayesian/newasia.csv')

asias<-data.frame(asias)

#asias<-asias[,2:9]
asias
asiass<-as.data.frame(lapply(asias,as.character))
score(hc(asiass),asiass,type="bd")

for (i in 1:10){
  k =i
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("asia",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  a_ti = proc.time()
  print(score(hc(asiass[start:end,]),asiass[start:end,]),type="bd")
  a_ti2 = proc.time()
  a_time = a_ti2-a_ti
  print(paste0('执行时间：',a_time[3][[1]],'秒'))
  # print(cancers[start:end,])
  cat(sep="\n")
}


# cancer

cancers = read.csv('D:/paper/BayesianNetwork/R bayesian/newcancer.csv')

cancers<-data.frame(cancers)
cancers
#which(is.na(alarms),arr.ind = T)
#alarms<-alarms[,2:36]
# which(is.na(alarms),arr.ind = T)

#str(insurances)


cancerss<-as.data.frame(lapply(cancers,as.character))


score(hc(cancerss),cancerss,type="bd")
# which (rowSums(is.na(alarms))==NA)
for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("cancer",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  can_ti = proc.time()
  print(score(hc(cancerss[start:end,]),cancerss[start:end,]),type="bd")
  can_ti2 = proc.time()
  can_time = can_ti2-can_ti
  print(paste0('执行时间：',can_time[3][[1]],'秒'))
  # print(cancers[start:end,])
  cat(sep="\n")
}


# survey
surveys = read.csv('D:/paper/BayesianNetwork/R bayesian/newsurvey.csv')

surveys<-data.frame(surveys)
#surveys<-surveys[,2:7]
surveyss<-as.data.frame(lapply(surveys,as.character))
surveyss
score(hc(surveyss),surveyss)

for (i in 1:10){
  k =i
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("surveys",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(hc(surveyss[start:end,]),surveyss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}

# insurance
insurances = read.csv('D:/paper/BayesianNetwork/R bayesian/newinsurance.csv')

insurances<-data.frame(insurances)
insurances
#insurances<-insurances[,2:28]
#str(insurances)
insurancess<-as.data.frame(lapply(insurances,as.character))
score(hc(insurancess),insurancess)

for (i in 1:10){
  k =i
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("insurances",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(hc(insurancess[start:end,]),insurancess[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}

# mildew
mildews = read.csv('D:/paper/BayesianNetwork/R bayesian/newmildew.csv')

mildews<-data.frame(mildews)
mildews
#mildews<-mildews[,2:36]
#str(insurances)
mildewss<-as.data.frame(lapply(mildews,as.character))
score(hc(mildewss),mildewss)

for (i in 1:10){
  k =i
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("mildews",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(hc(mildewss[start:end,]),mildewss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}

# alarm
alarms = read.csv('D:/paper/BayesianNetwork/R bayesian/newalarm.csv')

alarms<-data.frame(alarms)
alarms
#which(is.na(alarms),arr.ind = T)
#alarms<-alarms[,2:36]
# which(is.na(alarms),arr.ind = T)

#str(insurances)


alarmss<-as.data.frame(lapply(alarms,as.character))


score(hc(alarmss),alarmss)
# which (rowSums(is.na(alarms))==NA)
for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("alarm",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(hc(alarmss[start:end,]),alarmss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}

# hailfinder
hailfinders = read.csv('D:/paper/BayesianNetwork/R bayesian/newhailfinder.csv')

hailfinders<-data.frame(hailfinders)
hailfinders
#hailfinders<-hailfinders[,2:57]
#which(is.na(hailfinders))
#str(insurances)
hailfinderss<-as.data.frame(lapply(hailfinders,as.character))

score(hc(hailfinderss),hailfinderss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("hailfinders",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(hc(hailfinderss[start:end,]),hailfinderss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}

# hepar2
hepar2s = read.csv('D:/paper/BayesianNetwork/R bayesian/newhepar2.csv')

hepar2s<-data.frame(hepar2s)
hepar2s

hepar2ss<-as.data.frame(lapply(hepar2s,as.character))
#str(insurances)
score(hc(hepar2ss),hepar2ss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("hepar2s",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(hc(hepar2ss[start:end,]),hepar2ss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}

# win95pts
win95ptss = read.csv('D:/paper/BayesianNetwork/R bayesian/newwin95pts.csv')

win95ptss<-data.frame(win95ptss)
win95ptss

win95ptsss<-as.data.frame(lapply(win95ptss,as.character))
#str(insurances)
score(hc(win95ptsss),win95ptsss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("win95ptss",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(hc(win95ptsss[start:end,]),win95ptsss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}



# andes
andess = read.csv('D:/paper/BayesianNetwork/R bayesian/newandes.csv')

andess<-data.frame(andess)
andess
andesss<-as.data.frame(lapply(andess,as.character))

#str(insurances)
score(hc(andesss),andesss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("andess",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(hc(andesss[start:end,]),andesss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}

# diabetes
diabetess = read.csv('D:/paper/BayesianNetwork/R bayesian/newdiabetes.csv')

diabetess<-data.frame(diabetess)
diabetess
diabetesss<-as.data.frame(lapply(diabetess,as.character))
#str(insurances)
score(hc(diabetesss),diabetesss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("diabetess",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(hc(diabetesss[start:end,]),diabetesss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}

# pigss
pigss = read.csv('D:/paper/BayesianNetwork/R bayesian/newpigs.csv')

pigss<-data.frame(pigss)
pigss

pigss=as.data.frame(lapply(pigss,as.character))
#str(insurances)
score(hc(pigss),pigss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("pigss",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(hc(pigss[start:end,]),pigss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}

###############################################################
## andes



## earthquake
earthquakes = read.csv('D:/paper/BayesianNetwork/R bayesian/newearthquaker.csv')

earthquakes<-data.frame(earthquakes)
earthquakes
#which(is.na(alarms),arr.ind = T)
#alarms<-alarms[,2:36]
# which(is.na(alarms),arr.ind = T)

#str(insurances)


earthquakess<-as.data.frame(lapply(earthquakes,as.character))


score(hc(earthquakess),earthquakess,type="k2")
# which (rowSums(is.na(alarms))==NA)
for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("earthquake",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  can_ti = proc.time()
  print(score(hc(earthquakess[start:end,]),earthquakess[start:end,]),type="k2")
  can_ti2 = proc.time()
  can_time = can_ti2-can_ti
  print(paste0('执行时间：',can_time[3][[1]],'秒'))
  # print(cancers[start:end,])
  cat(sep="\n")
}


###################################
## concept_map1


concept_map1 = read.csv('D:/paper/construct concept map/LPG-algorithm-datasets-master/files/qc_g1_mean.csv')

concept_map1<-data.frame(concept_map1)
concept_map1
# pigss<-pigss[,2:442]
concept_map1s=as.data.frame(lapply(concept_map1,as.character))
#str(insurances)
t1 <-  proc.time()
score(hc(concept_map1s),concept_map1s)
t2 <- proc.time()
t= t2-t1
print(paste0('执行时间：',t[3][[1]],'秒'))

## concept_map2

concept_map2 = read.csv('D:/paper/construct concept map/LPG-algorithm-datasets-master/files/new_qc_g2.csv')

concept_map2<-data.frame(concept_map2)
concept_map2
# pigss<-pigss[,2:442]
concept_map2s=as.data.frame(lapply(concept_map2,as.character))
#str(insurances)
t3 <-  proc.time()
score(hc(concept_map2s),concept_map2s)
t4 <- proc.time()
tt= t4-t3
print(paste0('执行时间：',tt[3][[1]],'秒'))

# concept_map3
concept_map3 = read.csv('D:/paper/construct concept map/LPG-algorithm-datasets-master/files/new_qc_g3.csv')

concept_map3<-data.frame(concept_map2)
concept_map3
# pigss<-pigss[,2:442]
concept_map3s=as.data.frame(lapply(concept_map3,as.character))
#str(insurances)
t5 = proc.time()
score(hc(concept_map3s),concept_map3s)
t6 = proc.time()
ttt=t6-t5
print(paste0('执行时间：',ttt[3][[1]],'秒'))

# # concept_map4
concept_map4 = read.csv('D:/paper/construct concept map/LPG-algorithm-datasets-master/files/new_qc_g4.csv')

concept_map4<-data.frame(concept_map4)
concept_map4
# pigss<-pigss[,2:442]
concept_map4s=as.data.frame(lapply(concept_map4,as.character))
#str(insurances)
t7 = proc.time()
score(hc(concept_map4s),concept_map4s)
t8 = proc.time()
tttt = t8-t7
print(paste0('执行时间：',tttt[3][[1]],'秒'))


# # concept_map5
concept_map5 = read.csv('D:/paper/construct concept map/LPG-algorithm-datasets-master/files/qc_g1_mean.csv')

concept_map5<-data.frame(concept_map5)
concept_map5
# pigss<-pigss[,2:442]
concept_map5s=as.data.frame(lapply(concept_map5,as.character))
#str(insurances)
t9 = proc.time()
score(hc(concept_map5s),concept_map5s)
t10 = proc.time()
ttttt = t10-t9
print(paste0('执行时间：',ttttt[3][[1]],'秒'))

### qc_g1_mean
concept_map1_mean = read.csv('D:/paper/construct concept map/LPG-algorithm-datasets-master/files/qc_g1_mean.csv')
concept_map1_mean<-data.frame(concept_map1_mean)
#concept_map1_mean<- factor(concept_map1_mean, levels=c(1, 2))

# pigss<-pigss[,2:442]
concept_map1s_mean=as.data.frame(lapply(concept_map1_mean,as.character))
#str(insurances)
t1 <-  proc.time()
score(hc(concept_map1s_mean),concept_map1s_mean)
t2 <- proc.time()
t= t2-t1
print(paste0('执行时间：',t[3][[1]],'秒'))

### qc_g2_mean
concept_map2 = read.csv('D:/paper/construct concept map/LPG-algorithm-datasets-master/files/qc_g2_mean.csv')

concept_map2<-data.frame(concept_map2)
concept_map2
# pigss<-pigss[,2:442]
concept_map2s=as.data.frame(lapply(concept_map2,as.character))
str(concept_map2s)
#str(insurances)
t3 <-  proc.time()
score(hc(concept_map2s),concept_map2s)
t4 <- proc.time()
tt= t4-t3
print(paste0('执行时间：',tt[3][[1]],'秒'))