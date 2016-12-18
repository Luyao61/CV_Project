require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

-- load train set images
trainset = torch.load('data/VGG_train.t7')
n_empty_images = trainset.n_empty_images
n_stand_images = trainset.n_stand_images
n_sit_images = trainset.n_sit_images

train_size = 8900
test_size = 1280

trainset.size = function() return trainset.n_empty_images+trainset.n_stand_images+trainset.n_sit_images end
print("Loda image data... DONE")

local model = nn.Sequential()
local features = nn.Sequential()
local classifier = nn.Sequential()
--cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'}
do
  local iChannels = 3;
  for k,v in ipairs(cfg) do
     if v == 'M' then
        features:add(nn.SpatialMaxPooling(2,2,2,2))
     else
        local oChannels = v;
        local conv3 = nn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
        features:add(conv3)
        features:add(nn.ReLU(true))
        iChannels = oChannels;
     end
  end
end
features:cuda()
classifier:add(nn.View(512*10*7))
classifier:add(nn.Linear(512*10*7, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.BatchNormalization(4096, 1e-3))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.BatchNormalization(4096, 1e-3))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 3))
classifier:add(nn.LogSoftMax())
classifier:cuda()
model:add(features):add(classifier)

criterion = nn.ClassNLLCriterion()

optimState = {
  learningRate = 0.01
}
optimMethod = optim.sgd
batchSize = 20

criterion:cuda()
model:cuda()
parameters,gradParameters = model:getParameters()
print("Build VGG model... DONE")

shuffle = torch.randperm(trainset:size())
for epoch = 1,10 do
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

      for i = 1, output:size(1) do
          local _, predicted_label = output[i]:max(1)


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
torch.save("models/VGG16_NW_epoch10.t7",model)

--[[
image.display(trainset.data[1])
output = model:forward(trainset.normdata[1])
print(output)]]

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
