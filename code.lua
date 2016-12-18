require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'


load_images = require 'load_images'
torch.setdefaulttensortype('torch.FloatTensor')

-- load train set images
n_sit_images = 123
sit_images = load_images.load('sit', n_sit_images)
n_stand_images = 156
stand_images = load_images.load('stand', n_stand_images)
trainset = {
  data = torch.Tensor(n_sit_images + n_stand_images, 3, 320, 240),
  label = torch.Tensor(n_sit_images + n_stand_images),
  size = function() return n_sit_images + n_stand_images end
}
for i = 1,n_sit_images do
  trainset.data[i] = sit_images[i]
  trainset.label[i] = torch.Tensor(1):fill(1)
end
for i = n_sit_images+1, n_stand_images + n_sit_images do
  trainset.data[i] = stand_images[i-n_sit_images]
  trainset.label[i] = torch.Tensor(1):fill(2)
end
-- load valset images
n_val_sit_images = 119
val_sit_images = load_images.load('val_sit', n_val_sit_images)
n_val_stand_images = 119
val_stand_images = load_images.load('val_stand', n_val_stand_images)
valset = {
  data = torch.Tensor(n_val_sit_images + n_val_stand_images, 3, 320, 240),
  label = torch.Tensor(n_val_sit_images + n_val_stand_images),
  size = function() return n_val_sit_images + n_val_stand_images end
}
for i = 1,n_val_sit_images do
  valset.data[i] = val_sit_images[i]
  valset.label[i] = torch.Tensor(1):fill(1)
end
for i = n_val_sit_images+1, n_val_stand_images + n_val_sit_images do
  valset.data[i] = val_stand_images[i-n_val_sit_images]
  valset.label[i] = torch.Tensor(1):fill(2)
end
-- preprocess data
trainset.normdata = trainset.data:clone()
valset.normdata = valset.data:clone()

mean = {
  trainset.normdata[{{}, {1}, {}, {}}]:mean(),
  trainset.normdata[{{}, {2}, {}, {}}]:mean(),
  trainset.normdata[{{}, {3}, {}, {}}]:mean()
}

std = {
  trainset.normdata[{{}, {1}, {}, {}}]:std(),
  trainset.normdata[{{}, {2}, {}, {}}]:std(),
  trainset.normdata[{{}, {3}, {}, {}}]:std()
}

for i =1,3 do
  trainset.normdata[{{}, {i}, {}, {}}]:add(-mean[i])
  trainset.normdata[{{}, {i}, {}, {}}]:div(std[i])
  valset.normdata[{{}, {i}, {}, {}}]:add(-mean[i])
  valset.normdata[{{}, {i}, {}, {}}]:div(std[i])
end

--create a model
--[[model = nn.Sequential()
model:add(nn.View(640 * 480 * 3))  -- This View layer vectorizes the images from a 3,32,32 tensor to a 3*32*32 vector.
model:add(nn.Linear(640 * 480 * 3, 2))  -- Linear transformation y = Wx + b
model:add(nn.LogSoftMax())  -- Log SoftMax function.]]
--local temp = torch.Tensor(2, 3, 1280, 960):fill(0)
--local output = model:forward(temp)
--print(#output)
local model = nn.Sequential()
local first = nn.Sequential()
local second = nn.Sequential()
--[[model:add(nn.SpatialConvolution(3, 8, 5, 5, ))  -- 3 input channels, 8 output channels (8 filters), 5x5 kernels.
model:add(nn.SpatialBatchNormalization(8, 1e-3))  -- BATCH NORMALIZATION LAYER.
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- Max pooling in 2 x 2 area.
model:add(nn.SpatialConvolution(8, 16, 5, 5))  -- 8 input channels, 16 output channels (16 filters), 5x5 kernels.
model:add(nn.SpatialBatchNormalization(16, 1e-3))  -- BATCH NORMALIZATION LAYER.
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))  -- Max pooling in 2 x 2 area.
model:add(nn.View(16*77*57))    -- Vectorize the output of the convolutional layers.
model:add(nn.Linear(16*77*57, 2))
model:add(nn.ReLU())
model:add(nn.Linear(30000, 10000))
model:add(nn.ReLU())
model:add(nn.Linear(10000, 5000))
model:add(nn.ReLU())
model:add(nn.Linear(5000, 1000))
model:add(nn.ReLU())
model:add(nn.Linear(1000, 200))
model:add(nn.ReLU())
model:add(nn.Linear(200, 2))
model:add(nn.LogSoftMax())]]
first:add(nn.SpatialConvolution(3,64,11,11,4,4,2,2))
first:add(nn.SpatialBatchNormalization(64))
first:add(nn.ReLU())
first:add(nn.SpatialMaxPooling(3, 3, 2, 2))
first:add(nn.SpatialConvolution(64,192,5,5,1,1,2,2))
first:add(nn.SpatialBatchNormalization(192))
first:add(nn.ReLU())
first:add(nn.SpatialMaxPooling(3, 3, 2, 2))
first:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))
first:add(nn.SpatialBatchNormalization(384))
first:add(nn.ReLU())
first:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))
first:add(nn.SpatialBatchNormalization(256))
first:add(nn.ReLU())
first:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))
first:add(nn.SpatialBatchNormalization(256))
first:add(nn.ReLU())
first:add(nn.SpatialMaxPooling(3, 3, 2, 2))
model:add(first)
second:add(nn.View(256*9*6))
second:add(nn.Dropout(0.5))
second:add(nn.Linear(256*9*6, 4096))
--second:add(nn.SpatialBatchNormalization(4096))
second:add(nn.ReLU())
second:add(nn.Dropout(0.5))
second:add(nn.Linear(4096, 4096))
--second:add(nn.SpatialBatchNormalization(4096))
second:add(nn.ReLU())
second:add(nn.Linear(4096, 2))
second:add(nn.LogSoftMax())

model:add(second)

--[[local temp = torch.Tensor(64, 3, 320, 240):fill(0)
local output = model:forward(temp)
print(#output)]]

criterion = nn.ClassNLLCriterion()
--model:cuda()
--criterion:cuda()



function trainModel(model, opt, features, preprocessFn)
  local params, gradParams = model:getParameters()

  local opt = opt or {}
  local batchSize = opt.batchSize or 64  -- The bigger the batch size the most accurate the gradients.
  local learningRate = opt.learningRate or 0.001  -- This is the learning rate parameter often referred to as lambda.
  local momentumRate = opt.momentumRate or 0.9
  local numEpochs = opt.numEpochs or 3
  local velocityParams = torch.zeros(gradParams:size())
  local train_features, val_features
  if preprocessFn then
      train_features = trainset.data:float():div(255)
      val_features = valset.data:float():div(255)
  else
      train_features = (features and features.train_features) or trainset.normdata
      val_features = (features and features.val_features) or valset.normdata
  end
  for epoch = 1, numEpochs do
    local sum_loss = 0
    local correct = 0
    model:training()
    for i = 1, trainset.normdata:size(1)/batchSize do
      local inputs
      if preprocessFn then
        inputs = torch.Tensor(batchSize, 3, 320, 240)
        --print('preprocess not implemented yet.')
      else
        -- 4096?
        inputs = (features and torch.Tensor(batchSize, 4096)) or torch.Tensor(batchSize, 3, 320, 240)
      end
      local labels = torch.Tensor(batchSize)
      for bi = 1, batchSize do
        local rand_id = torch.random(1,train_features:size(1))
        if preprocessFn then
          inputs[bi] = preprocessFn(train_features[rand_id])
        else
          inputs[bi] = train_features[rand_id]
        end
        labels[bi] = trainset.label[rand_id]
      end
      --print(#inputs)
      local predictions = model:forward(inputs)

      for i = 1, predictions:size(1) do
          local _, predicted_label = predictions[i]:max(1)
          --print(i)
          if predicted_label[1] == labels[i] then correct = correct + 1 end
      end
      sum_loss = sum_loss + criterion:forward(predictions, labels)

      model:zeroGradParameters()
      local gradPredictions = criterion:backward(predictions, labels)
      model:backward(inputs, gradPredictions)

      velocityParams:mul(momentumRate)
      velocityParams:add(learningRate, gradParams)
      params:add(-1, velocityParams)

    end
    print(('train epoch=%d, sum-loss=%.6f, avg-accuracy = %.2f')
        :format(epoch, sum_loss, correct / trainset.normdata:size(1)))
    -- validate the train set.
    local validation_accuracy = 0
    local nBatches = val_features:size(1) / batchSize
    model:evaluate()

    for i = 1, nBatches do

        -- 1. Sample a batch.
        if preprocessFn then
            inputs = torch.Tensor(batchSize, 3, 320, 240)
        else
          -- 4096????
            inputs = (features and torch.Tensor(batchSize, 4096)) or torch.Tensor(batchSize, 3, 320, 240)
        end
        local labels = torch.Tensor(batchSize)
        for bi = 1, batchSize do
            local rand_id = torch.random(1, val_features:size(1))
            if preprocessFn then
                inputs[bi] = preprocessFn(val_features[rand_id])
            else
                inputs[bi] = val_features[rand_id]
            end
            labels[bi] = valset.label[rand_id]
        end

        -- 2. Perform the forward pass (prediction mode).
        local predictions = model:forward(inputs)

        -- 3. evaluate results.
        for i = 1, predictions:size(1) do
            local _, predicted_label = predictions[i]:max(1)
            if predicted_label[1] == labels[i] then validation_accuracy = validation_accuracy + 1 end
        end
    end
    validation_accuracy = validation_accuracy / (nBatches * batchSize)
    print(('\nvalidation accuracy at epoch = %d is %.4f'):format(epoch, validation_accuracy))

  end
end
function preprocess(im)
    local output_image = image.scale(im:clone(), 320, 240)
--[[    for i = 1, 3 do -- channels
        output_image[{{i},{},{}}]:add(-meanStd.mean[i])
        output_image[{{i},{},{}}]:div(meanStd.std[i])
    end]]
    return output_image
end

trainModel(model)
