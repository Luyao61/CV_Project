require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'optim'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')
load_model = require 'load_model'


-- load train set images
trainset = torch.load('data/train_new.t7')
n_empty_images = trainset.n_empty_images
n_stand_images = trainset.n_stand_images
n_sit_images = trainset.n_sit_images

train_size = 11000
test_size = 2055
trainset.size = function() return trainset.n_empty_images+trainset.n_stand_images+trainset.n_sit_images end
print("Loda image data... DONE")

local model = nn.Sequential()
local first = nn.Sequential()
local second = nn.Sequential()

model = load_model.loadAlexNet()

criterion = nn.ClassNLLCriterion()

optimState = {
  learningRate = 0.01
}
optimMethod = optim.sgd
batchSize = 5

criterion:cuda()
parameters,gradParameters = model:getParameters()
print("Loda AlexNET model... DONE")

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
torch.save("models/AlexNet_W_epoch20_new.t7",model)

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
