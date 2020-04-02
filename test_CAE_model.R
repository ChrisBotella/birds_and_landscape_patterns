# created on 20/03/20

# load packages
require(mxnet)
require(raster)
require(ggplot2)

.pardefault <- par()

dataDir = "/home/leguilln/workspace/birds_and_landscape/birds_and_landscape_patterns/data"
dataPath = paste(dataDir, 'data_birds_landscape_CNN.RData', sep="/")
load(dataPath)
setwd(dataDir)

modelName = "test_cae_best_4"
train = FALSE
loadModel = "test_cae_best_4" #NULL
iteration = 0
n_it = 5000

newBirdsArray = array( unlist(birds_array[1,,]) , dim=dim(birds_array[1,,]) )
dimnames(newBirdsArray) = dimnames(birds_array)[2:3]
inY = colnames(newBirdsArray)
inX = as.character(dimnames(raster_array)[3][[1]])
inXandY = intersect(inX,inY)
inXandYnotNa = setdiff(inXandY,raster_with_na)

Xarray= raster_array[,,dimnames(raster_array)[3][[1]]%in%inXandYnotNa] 
dim(Xarray) = c(dim(Xarray)[1],dim(Xarray)[2],1,dim(Xarray)[3])

MainDevice = mx.cpu()

# create train, test and validation data sets

nRasters = dim(Xarray)[4]
nValid = 500
nTest = 100

bag = 1:nRasters
trainSample = sample(bag, nRasters - nValid - nTest)
validSample = sample( setdiff(bag,trainSample) , nValid )
testSample = setdiff( setdiff(bag,trainSample) , validSample)

Xtrain = Xarray[,,,trainSample,drop=F]
Xvalid = Xarray[,,,validSample,drop=F]
Xtest = Xarray[,,,testSample,drop=F]

# save(Xtrain, Xvalid, Xtest, file = "train_valid_test.RData")

load("train_valid_test.RData")

### define the convolutional autoencoder architectur  e

data = mx.symbol.Variable(name = "data")
label = mx.symbol.Variable(name = "label")

## the encoder part of the network is made of 2 max-pooling convolutional layers

# 1st convolutional layer
conv1 = mx.symbol.Convolution(data = data, 
                              kernel = c(3, 3), 
                              pad = c(2, 2),
                              num_filter = 16,
                              name = "conv1")
relu1 = mx.symbol.Activation(data = conv1,
                             act_type = "relu",
                             name = "relu1")
pool1 = mx.symbol.Pooling(data = relu1,
                          pool_type = "max",
                          kernel = c(2,2),
                          stride=c(2,2),
                          name = "pool1")

# 2nd convolutional layer
conv2 = mx.symbol.Convolution(data = pool1, 
                              kernel = c(3, 3), 
                              pad = c(1, 1),
                              num_filter = 4,
                              name = "conv2")
relu2 = mx.symbol.Activation(data = conv2,
                             act_type = "relu",
                             name = "relu2")
pool2 = mx.symbol.Pooling(data = relu2,
                          pool_type = "max",
                          kernel = c(2,2),
                          stride=c(2,2),
                          name = "pool2")

## the decoder part of the network is made of 2 transposed convolutional layers

# 1st transposed convolutional layer
deconv1 = mx.symbol.Deconvolution(data = pool2, 
                              kernel = c(2, 2), 
                              stride = c(2, 2),
                              num_filter = 16,
                              layout='NCHW',
                              name = "deconv1")

# 2nd transposed convolutional layer
deconv2 = mx.symbol.Deconvolution(data = deconv1, 
                                  kernel = c(2, 2), 
                                  stride = c(2, 2),
                                  num_filter = 1,
                                  layout='NCHW',
                                  name = "deconv2")

# mean squared error
loss <- mx.symbol.LinearRegressionOutput(deconv2, name = "loss")

### mxnet utilities

mx.callback.early.stop.and.save.checkpoint <- function(train.metric = NULL, eval.metric = NULL, bad.steps = NULL, maximize = FALSE, verbose = FALSE, prefix = "") {
  function (iteration, nbatch, env, verbose = verbose) 
  {
    if (!is.null(env$metric)) {
      if (!is.null(train.metric)) {
        result <- env$metric$get(env$train.metric)
        if ((!maximize && result$value < train.metric) || 
            (maximize && result$value > train.metric)) {
          return(FALSE)
        }
      }
      if (!is.null(eval.metric)) {
        if (!is.null(env$eval.metric)) {
          result <- env$metric$get(env$eval.metric)
          if ((!maximize && result$value < eval.metric) || 
              (maximize && result$value > eval.metric)) {
            return(FALSE)
          }
        }
      }
    }
    if (!is.null(bad.steps)) {
      if (iteration == 1) {
        mx.best.iter <<- 1
        if (maximize) {
          mx.best.score <<- 0
        }
        else {
          mx.best.score <<- Inf
        }
      }
      if (!is.null(env$eval.metric)) {
        result <- env$metric$get(env$eval.metric)
        if ((!maximize && result$value > mx.best.score) || 
            (maximize && result$value < mx.best.score)) {
          if (mx.best.iter == bad.steps) {
            if (verbose) {
              message("Best score=", mx.best.score, ", iteration [", 
                      iteration - bad.steps, "]")
            }
            return(FALSE)
          }
          else {
            mx.best.iter <<- mx.best.iter + 1
          }
        }
        else {
          mx.best.score <<- result$value
          mx.best.iter <<- 1
          mx.model.save(env$model, prefix, 0)
          cat(sprintf("Model checkpoint saved to %s-0.params\n", prefix))
        }
      }
    }
    return(TRUE)
  }
}

### learn and save model

# Training parameters
bad.steps = 50
batch.size = 32
saveDir = dataDir
prefix = paste(saveDir, modelName, sep="/")

if(!is.null(loadModel)) {
  # OR load pre-trained model
  setwd(saveDir)
  model = mx.model.load(modelName, iteration = 0)
}

if(train == TRUE) {
  if(is.null(loadModel)) {
    # Randomly initialize the model weights
    mx.set.seed(2019)
    model = mx.model.FeedForward.create(symbol=loss,
                                        X=Xtrain,
                                        y=Xtrain,
                                        eval.data=list(data=Xvalid, label=Xvalid),
                                        ctx=MainDevice,
                                        begin.round=1,
                                        num.round=n_it,
                                        array.batch.size=batch.size,
                                        optimizer="adagrad",
                                        initializer=mx.init.Xavier(),
                                        eval.metric=mx.metric.mse,
                                        epoch.end.callback=mx.callback.early.stop.and.save.checkpoint(bad.steps=bad.steps, prefix=prefix, verbose=TRUE)
    )
  } else {
    #continue training
    mx.best.iter <<- 0
    model = mx.model.FeedForward.create(model$symbol,
                                        X=Xtrain,
                                        
                                        eval.data=list(data=Xvalid, label=Xvalid),
                                        ctx=MainDevice,
                                        begin.round=iteration,
                                        num.round=n_it,
                                        array.batch.size=batch.size,
                                        optimizer="adagrad",
                                        eval.metric=mx.metric.mse,
                                        arg.params=model$arg.params, 
                                        aux.params=model$aux.params,
                                        epoch.end.callback=mx.callback.early.stop.and.save.checkpoint(bad.steps=bad.steps, prefix=prefix, verbose=TRUE)
    )
  }
}

predicted = predict(model, X=Xtest)
par(mfrow=c(3,2))
for(i in 1:3) {
  image(Xtest[,,1,i], useRaster=TRUE, axes=FALSE)
  image(predicted[,,1,i], useRaster=TRUE, axes=FALSE)
}

encode <- function(input, model)
{
  arg.params = model$arg.params[c("conv1_weight", "conv1_bias", "conv2_weight", "conv2_bias")]
  
  data = mx.symbol.Variable("data")
  # 1st convolutional layer
  conv1 = mx.symbol.Convolution(data = data, 
                                # weight = model$arg.params$conv1_weight,
                                # bias = arg.params["conv1_bias"],
                                kernel = c(3, 3), 
                                pad = c(2, 2),
                                num_filter = 16,
                                name = "conv1")
  relu1 = mx.symbol.Activation(data = conv1,
                               act_type = "relu",
                               name = "relu1")
  pool1 = mx.symbol.Pooling(data = relu1,
                            pool_type = "max",
                            kernel = c(2,2),
                            stride=c(2,2),
                            name = "pool1")
  
  # 2nd convolutional layer
  conv2 = mx.symbol.Convolution(data = pool1, 
                                # weight = arg.params["conv2_weight"],
                                # bias = arg.params["conv2_bias"],
                                kernel = c(3, 3), 
                                pad = c(1, 1),
                                num_filter = 4,
                                name = "conv2")
  relu2 = mx.symbol.Activation(data = conv2,
                               act_type = "relu",
                               name = "relu2")
  pool2 = mx.symbol.Pooling(data = relu2,
                            pool_type = "max",
                            kernel = c(2,2),
                            stride=c(2,2),
                            name = "pool2")
  
  flatten = mx.symbol.flatten(data = pool2)
  # transpose = mx.symbol.transpose(data = flatten)
  
  encoder_model = list(symbol = flatten, arg.params = arg.params, aux.params = list())
  class(encoder_model) = "MXFeedForwardModel"
  
  output <- predict(encoder_model, X=input) #, array.layout = "colmajor")
  
  return(output)
}

latent_vector <- function(data, result, index)
{
  image(Xtest[,,1,index], useRaster=TRUE, axes=FALSE)
  print(result[,,,i])
}

predicted = encode(Xarray, model)
print(dim(predicted))

# t-SNE analysis

library(Rtsne)
library("ggplot2")
library("ggimage")

par(mfrow=c(1,1))
dev.off()

tsne = Rtsne(t(predicted), check_duplicates = FALSE, theta=0.0)


for(i in 1:nRasters) {
  writePNG(1-Xarray[,,1,i]/255., target=sprintf('./land_test_images/land_%04d.png', i))
}
img = list.files(path="./land_test_images", pattern="*.png")
img = paste("./land_test_images/", img, sep="")
d = data.frame(x=tsne$Y[,1], y=tsne$Y[,2], image=img)

p = ggplot(d, aes(x, y)) + geom_image(aes(image=image), size=.05)
print(p)