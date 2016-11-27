-- https://github.com/torch/image

require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'optim'

load_images = require 'load_images'
torch.setdefaulttensortype('torch.FloatTensor')

n_sit_images = 3670
sit_images = load_images.load('sit', n_sit_images)
n_stand_images = 2910
stand_images = load_images.load('stand', n_stand_images)
n_empty_images = 3005
empty_images = load_images.load('empty', n_empty_images)
train_size = 8600
test_size = 985

--

trainset = {
  data = torch.Tensor(n_sit_images + n_stand_images + n_empty_images, 3, 320, 240),
  label = torch.Tensor(n_sit_images + n_stand_images + n_empty_images),
  size = function() return n_sit_images + n_stand_images + n_empty_images end
}

for i = 1,n_sit_images do
  trainset.data[i] = sit_images[i]
  trainset.label[i] = 1
end
for i = n_sit_images+1, n_stand_images + n_sit_images do
  trainset.data[i] = stand_images[i-n_sit_images]
  trainset.label[i] = 2
end
for i = n_sit_images+n_stand_images+1, n_stand_images + n_sit_images + n_empty_images do
  trainset.data[i] = empty_images[i-n_sit_images-n_stand_images]
  trainset.label[i] = 3
end

stand_images = nil
sit_images = nil
empty_images = nil
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
end
trainset.data = nil
trainset.mean = mean
trainset.std = std
trainset.n_sit_images = 3670
trainset.n_stand_images = 2910
trainset.n_empty_images = 3005

torch.save("data/train.t7", trainset)