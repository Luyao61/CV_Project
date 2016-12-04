--https://github.com/torch/demos/blob/master/mst-based-segmenter/run.lua
--https://github.com/torch/image/blob/master/doc/drawing.md
--https://github.com/clementfarabet/lua---ffmpeg/blob/master/init.lua

require 'ffmpeg'
require 'image'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'qtwidget'

require 'xlua'
require 'torch'
require 'qt'
require 'imgraph'
require 'nnx'
torch.setdefaulttensortype('torch.FloatTensor')

local file = "demo1"
local video_path = "data/" .. file .. ".MP4"
local image_width = 320
local image_height = 240
local zoom = 2
local batchSize = 5
local video_length = 300
local fps = 10
local classes = {"Sit", "Stand", "Empty"}

--trainset = torch.load('data/train.t7')
--mean = trainset.mean
--std = trainset.std
-- mean and std is retrived from preprocess_image.lua
mean = {
  0.51186811272022,
  0.47379257591268,
  0.43529572474515
}
std = {
  0.27868262464025,
  0.27238984884143,
  0.27127185550684
}
-- extract frames from video using ffmpeg
video = ffmpeg.Video{path=video_path, width=image_width, height=image_height,
                    length = video_length, fps = fps, delete = false}
frames = {
  size = function() return fps*video_length end
}
frames.data = video:totensor{}
frames.normdata = frames.data:clone()
--preprocess data
for i = 1,3 do
  frames.normdata[{{},{i},{},{}}]:add(-mean[i])
  frames.normdata[{{},{i},{},{}}]:div(std[i])
end

model = torch.load("models/VGG19_W_epoch30.t7")
--[[temp = torch.CudaTensor(5,3,320,240):fill(1)
output = model:forward(temp)
local scores, classIds = output[1]:exp():sort(true)

for i = 1,5 do
  print(scores[i])
  print(classIds[i])
end
]]
for i = 1, frames:size(),batchSize do
  inputs = torch.Tensor(batchSize,3,320,240)
  for j = i,i+batchSize-1 do
    local input = frames.normdata[j]
    inputs[j-i+1] = input
  end
  inputs = inputs:cuda()
  local outputs = model:forward(inputs)

  for j = 1, outputs:size(1) do
      --local _, predicted_label = outputs[j]:exp():sort(true)
      local scores, classIds = outputs[j]:exp():sort(true)
      --print(outputs[j])
      --print(("Frames: %d; Predict Result: %s. "):format(i+j-1,classes[classIds[1]]))
      frames.data[i+j-1] = image.drawText(frames.data[i+j-1], classes[classIds[1]], 10, 10, {bg = {255, 255, 255}, size = 4})
  end
end





if not win or not widget then
   win = qtwidget.newwindow(image_width*zoom, image_height*zoom,
                            'CV_Project Demo')
end
function process()
  current = current or 1
  frame = frames.data[current]
  current = current + 1
  if current > frames:size() then current = 1 end
end
function display()
   image.display{image=frame, win=win, zoom=zoom}
end


timer = qt.QTimer()
timer.interval = 1000/fps/4
timer.singleShot = true
qt.connect(timer,
           'timeout()',
           function()
              process()
              win:gbegin()
              win:showpage()
              display()
              win:gend()
              timer:start()
           end)
timer:start()


--img = image.drawText(image.lena(), "hello\nworld", 10, 10)
--image.display(img)
