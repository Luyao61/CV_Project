require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')


-- load train set images
trainset = torch.load('data/train.t7')
n_empty_images = trainset.n_empty_images
n_stand_images = trainset.n_stand_images
n_sit_images = trainset.n_sit_images

train_size = 8900
test_size = 1285
trainset.size = function() return trainset.n_empty_images+trainset.n_stand_images+trainset.n_sit_images end

print("Loda image data... DONE")

local model = nn.Sequential()
local first = nn.Sequential()
local second = nn.Sequential()

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
second:add(nn.Linear(4096, 3))
second:add(nn.LogSoftMax())
model:add(second)

criterion = nn.ClassNLLCriterion()

optimState = {
  learningRate = 0.01
}
optimMethod = optim.sgd
batchSize = 5

criterion:cuda()
model:cuda()
parameters,gradParameters = model:getParameters()
print("Build AlexNet model... DONE")

shuffle = torch.randperm(trainset:size())
for epoch = 1,20 do
  local f = 0
  local correct_count = 0
  model:training()
  for t = 1,train_size,batchSize do

    local inputs = torch.CudaTensor(batchSize,3,320,240)
    local targets = torch.CudaTensor(batchSize)
    for i = t,t+batchSize-1 do
      local input = trainset.normdata[shuffle[i]]
      local target = trainset.label[shuffle[i]]

      inputs[i - t + 1] = input
      targets[i - t + 1] = target

    end
    local feval = function(x)
      if x~= parameters then
        parameters:copy(x)
      end

      gradParameters:zero()
      inputs = inputs:cuda()
      targets = targets:cuda()
      local output = model:forward(inputs)
      --print(output)
      for i = 1, output:size(1) do
          local _, predicted_label = output[i]:max(1)
--[[            print (predicted_label[1])
            print (targets[i])
            print (" ")]]
          if predicted_label[1] == targets[i] then correct_count = correct_count + 1 end
      end
      local err = criterion:forward(output, targets)
      f = f + err
      local df_do = criterion:backward(output, targets)
      model:backward(inputs, df_do)

      return f,gradParameters
    end
    optimMethod(feval, parameters, optimState)
  end
  print(("epoch = %d; train mse = %.6f; Accuracy = %.3f"):format(epoch,f/train_size, correct_count/train_size))


  f=0
  correct_count = 0
  model:evaluate()
  for t = 1,test_size,batchSize do
    local inputs = torch.CudaTensor(batchSize,3,320,240)
    local targets = torch.CudaTensor(batchSize)
    for i = t,t+batchSize-1 do
      local input = trainset.normdata[shuffle[train_size+i]]
      local target = trainset.label[shuffle[train_size+i]]
      inputs[i - t + 1] = input
      targets[i - t + 1] = target
    end
    inputs = inputs:cuda()
    targets = targets:cuda()
    local output = model:forward(inputs)
    for i = 1, output:size(1) do
        local _, predicted_label = output[i]:max(1)
        --print(i)
        if predicted_label[1] == targets[i] then correct_count = correct_count + 1 end
    end
    local err = criterion:forward(output, targets)
    f = f + err
  end
  print(("epoch = %d; test mse = %.6f; Accuracy = %.3f"):format(epoch,f/test_size,correct_count/test_size))
end
torch.save("models/AlexNet_NW_epoch20.t7",model)

test_temp = torch.CudaTensor(5,3,320,240)
test_temp[1] = image.load("test_temp/empty110.jpg")
test_temp[2] = image.load("test_temp/sit46.jpg")
test_temp[3] = image.load("test_temp/sit1765.jpg")
test_temp[4] = image.load("test_temp/stand1778.jpg")
test_temp[5] = image.load("test_temp/stand2335.jpg")

test_output = model:forward(test_temp)
--[[for j = 1,5 do
  local scores, classIds = test_output[j][1]:exp():sort(true)
  print(("img %d"):format(j))
  for i = 1, 3 do
      print(('[%d] = %.5f'):format(classIds[i], scores[i]))
  end
end]]
print(test_output)
