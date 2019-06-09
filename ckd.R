#K-Fold
library(neuralnet)
#Randomly shuffle the data
mydata<-read.csv("dataset.csv",header=T)
mydata<-na.omit(mydata)
#Max-Min Normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

mydata <- as.data.frame(lapply(mydata, normalize))
result<-array()
err<-array()
#Create 10 equally size folds
folds <- cut(seq(1,nrow(mydata)),breaks=5,labels=FALSE)
#Perform 5 fold cross validation
for(i in 1:10){
  #Segement my data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testset <- mydata[testIndexes, ]
  trainset <- mydata[-testIndexes, ]
  #neuralnet
  nn <- neuralnet(cla ~ sg+su+rbc+pcc+ba+rc+htn+cad+appet+pe+ane, data=trainset, hidden=c(4,3), linear.output=TRUE, threshold=0.01)
  result[i]<-nn$result.matrix[1]
  #plot(nn)
  
  
  #Test the resulting output
  temp_test <- subset(testset, select = c("sg","su","rbc","pcc","ba","rc","htn","cad","appet","pe","ane"))
  head(temp_test)
  nn.results <- compute(nn, temp_test)
  #Accuracy
  results <- data.frame(actual = testset$cla, prediction = nn.results$net.result)
  results
  roundedresults<-sapply(results,round,digits=0)
  roundedresultsdf=data.frame(roundedresults)
  roundedresultsdf
  plot(results$prediction,results$prediction,col='red', ylab = "predicted rating NN", xlab = "real rating",pch=4,cex=2)
  points(results$actual,results$actual,col='blue',pch=18,cex=2)
  abline(0,1)
  legend('bottomright',legend='NN',pch=18,col='red', bty='n')
  attach(roundedresultsdf)
  
  confusionmatrix<-table(actual,prediction)
  confusionmatrix<-matrix(confusionmatrix,nrow = 2, byrow = TRUE)
  err[i]<-(confusionmatrix[2,1]*confusionmatrix[1,2])/sum(confusionmatrix)
  
}
avgerror<-sum(result)/length(result)
avgaccuracy<-(1-avgerror)*100
avgaccuracy

boxplot(result,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)