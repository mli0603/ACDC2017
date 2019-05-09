import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# helper function to show images from a batch of tensor
def imshow_batch(inp, denormalize=False):
    """Imshow for batch of Tensor."""
    
    # Make a grid from batch
    inp = torchvision.utils.make_grid(inp)

    # convert and display
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    if denormalize:
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
# helper function to show images
def imshow(img, denormalize=False):   
    # param:
        # img: tensor or np array (W,H,C)
    img = np.array(img)
    
    mean = 0.5
    std = 0.5
    if denormalize:
        img = std * img + mean
#         print(np.max(img))
#         print(np.min(img))
        img = np.clip(img, 0, 1)
        

    plt.imshow(img)
    plt.show()