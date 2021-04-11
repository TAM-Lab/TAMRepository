# asia

asias = read.csv('D:/paper/BayesianNetwork/R bayesian/newasia.csv')

asiass<-as.data.frame(lapply(asias,as.character))


score(mmhc(asiass),asiass)
# which (rowSums(is.na(alarms))==NA)
for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("asia",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  mmas_ti = proc.time()
  print(score(mmhc(asiass[start:end,]),asiass[start:end,]))
  mmas_ti2 = proc.time()
  mmas_time = mmas_ti2-mmas_ti
  print(paste0('执行时间：',mmas_time[3][[1]],'秒'))
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


score(mmhc(cancerss),cancerss)
# which (rowSums(is.na(alarms))==NA)
for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("cancer",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  mmca_ti = proc.time()
  print(score(mmhc(cancerss[start:end,]),cancerss[start:end,]))
  mmca_ti2 = proc.time()
  mmca_time =mmca_ti2-mmca_ti
  print(paste0('执行时间：',mmca_time[3][[1]],'秒'))
  # print(cancers[start:end,])
  cat(sep="\n")
}


# survey
surveys = read.csv('D:/paper/BayesianNetwork/R bayesian/newsurvey.csv')

surveys<-data.frame(surveys)
#surveys<-surveys[,2:7]
surveyss<-as.data.frame(lapply(surveys,as.character))
surveyss
score(mmhc(surveyss),surveyss)

for (i in 1:10){
  k =i
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("surveys",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(mmhc(surveyss[start:end,]),surveyss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}


# insurances

insurances = read.csv('D:/paper/BayesianNetwork/R bayesian/newinsurance.csv')

insurances<-data.frame(insurances)
insurances
#insurances<-insurances[,2:28]
#str(insurances)
insurancess<-as.data.frame(lapply(insurances,as.character))
score(mmhc(insurancess),insurancess)

for (i in 1:10){
  k =i
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("insurances",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(mmhc(insurancess[start:end,]),insurancess[start:end,]))
  
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
score(mmhc(mildewss),mildewss)

for (i in 1:10){
  k =i
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("mildews",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(mmhc(mildewss[start:end,]),mildewss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}


##################################################
## alarm
alarms = read.csv('D:/paper/BayesianNetwork/R bayesian/newalarm.csv')

alarms<-data.frame(alarms)
alarms
#which(is.na(alarms),arr.ind = T)
#alarms<-alarms[,2:36]
# which(is.na(alarms),arr.ind = T)

#str(insurances)


alarmss<-as.data.frame(lapply(alarms,as.character))


score(mmhc(alarmss),alarmss)
# which (rowSums(is.na(alarms))==NA)
for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("alarm",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(mmhc(alarmss[start:end,]),alarmss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}


############################################################################
# hailfinder
hailfinders = read.csv('D:/paper/BayesianNetwork/R bayesian/newhailfinder.csv')

hailfinders<-data.frame(hailfinders)
hailfinders
#hailfinders<-hailfinders[,2:57]
#which(is.na(hailfinders))
#str(insurances)
hailfinderss<-as.data.frame(lapply(hailfinders,as.character))

score(mmhc(hailfinderss),hailfinderss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("hailfinders",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(mmhc(hailfinderss[start:end,]),hailfinderss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}


# hepar2
hepar2s = read.csv('D:/paper/BayesianNetwork/R bayesian/newhepar2.csv')

hepar2s<-data.frame(hepar2s)
hepar2s

hepar2ss<-as.data.frame(lapply(hepar2s,as.character))
#str(insurances)
score(mmhc(hepar2ss),hepar2ss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("hepar2s",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(mmhc(hepar2ss[start:end,]),hepar2ss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}

# win95pts
win95ptss = read.csv('D:/paper/BayesianNetwork/R bayesian/newwin95pts.csv')

win95ptss<-data.frame(win95ptss)
win95ptss

win95ptsss<-as.data.frame(lapply(win95ptss,as.character))
#str(insurances)
score(mmhc(win95ptsss),win95ptsss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("win95ptss",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(mmhc(win95ptsss[start:end,]),win95ptsss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}
# andes

andess = read.csv('D:/paper/BayesianNetwork/R bayesian/newandes.csv')

andess<-data.frame(andess)
andess
andesss<-as.data.frame(lapply(andess,as.character))

#str(insurances)
score(mmhc(andesss),andesss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("andess",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(mmhc(andesss[start:end,]),andesss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}


# diabetes
diabetess = read.csv('D:/paper/BayesianNetwork/R bayesian/newdiabetes.csv')

diabetess<-data.frame(diabetess)
diabetess
diabetesss<-as.data.frame(lapply(diabetess,as.character))
#str(insurances)
score(mmhc(diabetesss),diabetesss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("diabetess",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(mmhc(diabetesss[start:end,]),diabetesss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}

## pigs 

pigss = read.csv('D:/paper/BayesianNetwork/R bayesian/newpigs.csv')

pigss<-data.frame(pigss)
pigss

pigss=as.data.frame(lapply(pigss,as.character))
#str(insurances)
#score(mmhc(pigss),pigss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("pigss",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(mmhc(pigss[start:end,]),pigss[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}


# earthquake
earthquakes = read.csv('D:/paper/BayesianNetwork/R bayesian/newearthquake.csv')

earthquakes<-data.frame(earthquakes)
earthquakes

earthquakes=as.data.frame(lapply(earthquakes,as.character))
#str(insurances)
#score(mmhc(pigss),pigss)

for (i in 1:10){
  
  start = (i-1)*10000+1
  end = i*10000
  print(start)
  print(end)
  cat("earthquakes",i)
  #cancers[start,end]
  #paste("cancer",i,":", cancers[start,end])
  print(score(mmhc(earthquakes[start:end,]),earthquakes[start:end,]))
  
  # print(cancers[start:end,])
  cat(sep="\n")
}


###############################################

## concept_map1
library(tibble)
concept_map1 = read.csv('D:/paper/construct concept map/LPG-algorithm-datasets-master/files/qc_g1_mean')

concept_map1 <- as_tibble(concept_map1)
concept_map1
# pigss<-pigss[,2:442]
#concept_map1s=as.data.frame(lapply(concept_map1,as.character))
#str(insurances)
start = proc.time()
concept_map1s <- as_tibble(concept_map1s)
score(mmhc(concept_map1s),concept_map1s)
end = proc.time()
mmcp1 = end-start
print(paste0('执行时间：',mmcp1[3][[1]],'秒'))
#################################################
## concept_map2

concept_map2 = read.csv('D:/paper/construct concept map/LPG-algorithm-datasets-master/files/new_qc_g2.csv')

concept_map2<-data.frame(concept_map2)
concept_map2
# pigss<-pigss[,2:442]
concept_map2s=as.data.frame(lapply(concept_map2,as.character))
#str(insurances)
start = proc.time()
score(mmhc(concept_map2s),concept_map2s)
end = proc.time()
mmcp2 = end-start
print(paste0('执行时间：',mmcp2[3][[1]],'秒'))
###################################################
# concept_map3
concept_map3 = read.csv('D:/paper/construct concept map/LPG-algorithm-datasets-master/files/new_qc_g3.csv')

concept_map3<-data.frame(concept_map2)
concept_map3
# pigss<-pigss[,2:442]
concept_map3s=as.data.frame(lapply(concept_map3,as.character))
#str(insurances)
start = proc.time()
score(mmhc(concept_map3s),concept_map3s)
end = proc.time()
mmcp3 = end-start
print(paste0('执行时间：',mmcp3[3][[1]],'秒'))
###############################################
# # concept_map4
concept_map4 = read.csv('D:/paper/construct concept map/LPG-algorithm-datasets-master/files/new_qc_g4.csv')

concept_map4<-data.frame(concept_map4)
concept_map4
# pigss<-pigss[,2:442]
concept_map4s=as.data.frame(lapply(concept_map4,as.character))
#str(insurances)
start = proc.time()
score(mmhc(concept_map4s),concept_map4s)
end = proc.time()
mmcp4 = end-start
print(paste0('执行时间：',mmcp4[3][[1]],'秒'))
################################################
# # concept_map5
concept_map5 = read.csv('D:/paper/construct concept map/LPG-algorithm-datasets-master/files/new_qc_g5.csv')

concept_map5<-data.frame(concept_map5)
concept_map5
# pigss<-pigss[,2:442]
concept_map5s=as.data.frame(lapply(concept_map5,as.character))
#str(insurances)
start = proc.time()
score(mmhc(concept_map5s),concept_map5s)
end = proc.time()
mmcp5 = end-start
print(paste0('执行时间：',mmcp5[3][[1]],'秒'))
