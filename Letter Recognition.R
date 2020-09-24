library(shiny)
library(tidyverse)
library(dplyr)
library(caret)
library(ggplot2)
library(MASS)
library(lattice)
library(klaR)
library(ElemStatLearn)
library(mlbench)
library(keras)
library(tensorflow)

#K Nearest Neighbours 

train_data<-read.csv(file = "emnist-letters-train.csv", header = TRUE, sep = ",")
test_data<-read.csv(file = "emnist-letters-test.csv", header = TRUE, sep = ",")

train_data$X23 <- as.factor(as.character(train_data$X23))

train_x = train_data[,-1]
train_y = data.frame(train_data[,1])

train.pca <- prcomp(train_x, center = TRUE)
a<-train.pca$x[,1:100]

x_train_s <- a[1:4000,]
x_test_s <- a[4001:24000,]
y_train_s <-data.frame(train_y[1:4000,1])
y_test_s <- train_y[4001:24000,1]

trains_s = cbind(y_train_s,x_train_s)

colnames(trains_s)[1]= "y"

model_knn_s <- train(y ~ ., data = trains_s, method = "knn",tuneGrid = data.frame(k = seq(5,10)))

predict_knn_s <- predict(model_knn_s, newdata = x_test_s)

#confusionMatrix(predict_knn_s, y_test_s)

x=seq(5:15)
y=c(0.6525319, 0.6524192,0.6544898,0.6559439,0.6579781, 0.6580839,0.6560452,0.6546288,0.6525285,0.6526287,0.6512710) 
z= data.frame(cbind(x,y))

ggplot(z, aes(x=x, y=y)) + geom_point() +ggtitle("Acuraccy vs K") +labs(y= "Accuracy", x = "k")+geom_line()    


#Artificial Neural Networks

train_data<-read.csv(file = "emnist-letters-train.csv", header = TRUE, sep = ",")
test_data<-read.csv(file = "emnist-letters-test.csv", header = TRUE, sep = ",")
  
train_data[,2:785] <- train_data[,2:785]/255.0;
  
trainx <- train_data[,-1]
train_pca <- prcomp(trainx,center = TRUE)
  
a <- train_pca$x[,1:250]
  
trainy <- train_data[1:75000,1]
testy <- test_data[75001:88799,1]
trainx <- a[1:75000,]
testx <- a[75001:88799,-1]
  
trainy=trainy-1
testy = testy-1
  
y_train <- to_categorical(trainy, 26)
  
model_keras <- keras_model_sequential()
  
model_keras<-model_keras %>% 
  layer_dense(units = 128, activation = 'relu', input_shape = c(250)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%  
  layer_dense(units = 26, activation = 'softmax')
  
model_keras %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)
  
annplot <- model_keras %>% 
  fit(
    trainx, trainy, 
    epochs = 400, batch_size = 128, 
    validation_split = 0.2
  )

plot(annplot)


#Convolutional Neural Netowrk

train_data<-read.csv(file = "emnist-letters-train.csv", header = TRUE, sep = ",")
test_data<-read.csv(file = "emnist-letters-test.csv", header = TRUE, sep = ",")

testdata[,2:785] <- testdata[,2:785]/255.0;
traindata[,2:785] <- traindata[,2:785]/255.0;
  
train_index <- createDataPartition(traindata$X23, p = 0.8, list = FALSE)
traindiv <- traindata[train_index, ]
testdiv <- traindata[-train_index, ]
  
trainy <- traindata[,1]
testy <- testdata[4001:6001,1]
trainy<-traindata[,1]
  
trainy=trainy-1
y_train <- to_categorical(trainy, 26)
y_test <- to_categorical(testy, 26)
img_rows <- 28
img_cols <- 28

a=data.matrix(traindata[,-1], rownames.force = NA)
trainx <- array_reshape(a, c(nrow(a), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

model_keras <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 26, activation = 'softmax')

model_keras %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy'))

cnnplot <- model_keras %>% 
  fit(
    trainx, trainy, 
    epochs = 30, batch_size = 128, 
    validation_split = 0.2
  )

plot(cnnplot)

