# Image classification script
# Homework 1 of CS423

# getting all filenames in current directory
# we only need the files with 'png' extension
images = filter(x->contains(x,".png"),readdir(pwd()))

x = zeros(5000,1024)
y = zeros(length(images),10)
function extractFeature(image)
    # now simply return vectorised greyscale image
    rimg = map(x->convert(Int64,x),rawview(channelview(Gray.(image))))
    return reshape(rimg, 1, 32^2)
end
for i=1:1

    # load image
    imm = load(images[i])

    # label extraction
    if contains(images[i],"airplane")
        y[i,1] = 1 
    elseif contains(images[i], "automobile")
        y[i,2] = 1
    elseif contains(images[i], "bird")
        y[i,3] = 1
    elseif contains(images[i], "cat")
        y[i,4] = 1
    elseif contains(images[i], "deer")
        y[i,5] = 1
    elseif contains(images[i], "dog")
        y[i,6] = 1
    elseif contains(images[i], "frog")
        y[i,7] = 1
    elseif contains(images[i], "horse")
        y[i,8] = 1
    elseif contains(images[i], "ship")
        y[i,9] = 1
    elseif contains(images[i], "truck")
        y[i,10] = 1
    end

    # feature extraction
    x[i,:] = extractFeature(imm)
end




# classification step
using BackpropNeuralNet   # we will use simple neural network

# initialise the network with 1024 input nodes, 
# 2 hidden layers with 10 nodes each 
# and output layer with 10 nodes to match our 10-class problem
net = init_network([1024,10,10,10])

# ---------------
# Training phase
# ---------------
epoch = 1000     # maximum learning epoch for each image

for i=1:1            # learning from all of the training images 
    for j=1:epoch
        train(net, x[i,:],y[i,:])
    end
end


# ---------------
# Testing phase
# ---------------

error = 0    # our initial error count is zero
for i=1:1    # loop over the unseen test images
    # the error is simple hamming distance between predicted output vector
    # and the true output vector
    error = error + sum(abs((round(net_eval(net, x[i,:]))-y[i,:])))
end
