-- https://github.com/torch/image

require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'optim'
require 'image'
function main()

    n_sit_images = 3670
    n_stand_images = 2910
    n_empty_images = 3605
    h = 240
    w = 320
    --
    trainset = {
      normdata = torch.FloatTensor(n_sit_images + n_stand_images + n_empty_images, 3, 320, 240),
      label = torch.Tensor(n_sit_images + n_stand_images + n_empty_images),
      size = function() return n_sit_images + n_stand_images + n_empty_images end
    }

    for i = 1,n_sit_images do
      local img = torch.Tensor(3, h, w)
      local img = image.load(string.format('sit/sit%d.jpg', i))
      trainset.normdata[i] = preprocess(img):float():clone()
      img = nil
      trainset.label[i] = 1
    end

    collectgarbage()
    for i = n_sit_images+1, n_stand_images + n_sit_images do
      local img = torch.Tensor(3, h, w)
      local img = image.load(string.format('stand/stand%d.jpg', i-n_sit_images))
      trainset.normdata[i] = preprocess(img):float():clone()
      img = nil
      trainset.label[i] = 2
    end
    collectgarbage()

    for i = n_sit_images+n_stand_images+1, n_stand_images + n_sit_images + n_empty_images do
      local img = torch.Tensor(3, h, w)
      local img = image.load(string.format('empty/empty%d.jpg', i-n_sit_images-n_stand_images))
      trainset.normdata[i] = preprocess(img):float():clone()

      trainset.label[i] = 3
    end
    stand_images = nil
    sit_images = nil
    empty_images = nil

    -- preprocess data
    trainset.n_sit_images = 3670
    trainset.n_stand_images = 2910
    trainset.n_empty_images = 3605

    if paths.dir("data") == nil then
      paths.mkdir("data")
    end
    torch.save("data/VGG_train.t7", trainset)
end
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end
main()
