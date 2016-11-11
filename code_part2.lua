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


-- preprocess data
trainset.normdata = trainset.data:clone()

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
  --valset.normdata[{{}, {i}, {}, {}}]:add(-mean[i])
  --valset.normdata[{{}, {i}, {}, {}}]:div(std[i])
end


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
second:add(nn.Linear(4096, 2))
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


for epoch = 1,5 do

  shuffle = torch.randperm(270)
  local f = 0
  model:training()
  for t = 1,200,batchSize do

    local inputs = torch.CudaTensor(batchSize,1,128,128)
    local targets = torch.CudaTensor(batchSize,2,128,128)
    for i = t,t+batchSize-1 do
      local input = trainset.data[shuffle[i]]
      local target = trainset.labels[shuffle[i]]

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
      local err = criterion:forward(output, targets)
      f = f + err
      local df_do = criterion:backward(output, targets)
      model:backward(inputs, df_do)

      return f,gradParameters
    end
    optimMethod(feval, parameters, optimState)
  end
  print(("epoch = %d; train mse = %.6f"):format(epoch,f/6750))


  f=0
  model:evaluate()
  for t = 1,70,batchSize do
    local inputs = torch.CudaTensor(batchSize,1,128,128)
    local targets = torch.CudaTensor(batchSize,2,128,128)
    for i = t,t+batchSize-1 do
      local input = trainset.data[shuffle[200+i]]
      local target = trainset.labels[shuffle[200+i]]
      inputs[i - t + 1] = input
      targets[i - t + 1] = target
    end
    inputs = inputs:cuda()
    targets = targets:cuda()
    local output = model:forward(inputs)
    local err = criterion:forward(output, targets)
    f = f + err
  end
  print(("epoch = %d; test mse = %.6f"):format(epoch,f/75))
end
