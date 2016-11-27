require 'torch'
require 'nn'
require 'loadcaffe'

local module = {}

function module.loadVGG(height, width)
  model = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt', 'models/VGG_ILSVRC_19_layers.caffemodel', 'nn')
  model:remove(46)
  model:remove(45)
  model:remove(44)
  model:remove(43)
  model:remove(42)
  model:remove(41)
  model:remove(40)
  model:remove(39)
  model:remove(38)
  local classifier = nn.Sequential()
  classifier:add(nn.View(512*10*8))
  classifier:add(nn.Linear(512*10*8, 4096))
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
  model:add(classifier)
  model:cuda()

  return model
end

function module.loadAlexNet(height, width)
  model = torch.load("models/alexnetowtbn_epoch55_cpu.t7")
  model:remove(2)
  print(model)
  local classifier = nn.Sequential()

  classifier:add(nn.View(256*9*6))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(256*9*6, 4096))
  classifier:add(nn.BatchNormalization(4096))
  classifier:add(nn.ReLU())
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(4096, 4096))
  classifier:add(nn.BatchNormalization(4096))
  classifier:add(nn.ReLU())
  classifier:add(nn.Linear(4096, 3))
  classifier:add(nn.LogSoftMax())

  classifier:cuda()
  model:add(classifier)
  model:cuda()
  return model
end


return module
