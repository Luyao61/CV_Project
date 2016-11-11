
require 'torch'
require 'image'

local module = {}

function module.load(path, n)
    h = 320
    w = 240

    images = torch.Tensor(n, 3, h, w)

    for i=1,n do
--        print('loading', i)
        I = image.load(string.format('%s/%s%d.jpg', path, path, i))
--        print('done with image load')
        images[i] = I
--        print('done loading')
    end

--    print('after for loop')

    return images
end

return module
