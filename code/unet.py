from torch import nn
import torch

# define helper functions to add conv
def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
  if useBN:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1)
    )
  else:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU(),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU()
    )

def add_merge_stage(ch_coarse, ch_fine, in_coarse, in_fine, upsample):
  conv = nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False)
  torch.cat(conv, in_fine)

  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False)
  )
  upsample(in_coarse)

def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
    nn.ReLU()
  )

# define unet model
num_class = 12
class unet(nn.Module):
  def __init__(self, useBN=False):
    super(unet, self).__init__()
    # Downgrade stages
    self.conv1   = add_conv_stage(10, 26, useBN=useBN)
    self.conv2   = add_conv_stage(26, 52, useBN=useBN)
    self.conv3   = add_conv_stage(52, 104, useBN=useBN)
    self.conv4   = add_conv_stage(104, 208, useBN=useBN)
    self.conv5   = add_conv_stage(208, 416, useBN=useBN)
    # Upgrade stages
    self.conv4m = add_conv_stage(416, 208, useBN=useBN)
    self.conv3m = add_conv_stage(208, 104, useBN=useBN)
    self.conv2m = add_conv_stage(104,  52, useBN=useBN)
    self.conv1m = add_conv_stage( 52,  26, useBN=useBN)
    # Maxpool
    self.max_pool = nn.MaxPool2d(2)
    # Upsample layers
    self.upsample54 = upsample(416, 208)
    self.upsample43 = upsample(208, 104)
    self.upsample32 = upsample(104, 52)
    self.upsample21 = upsample(52, 26)
    ## weight initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if m.bias is not None:
          m.bias.data.zero_()

    self.outconv = nn.Conv2d(26,num_class,1)
    
  def forward(self, x):
    conv1_out = self.conv1(x)
    conv2_out = self.conv2(self.max_pool(conv1_out))
    conv3_out = self.conv3(self.max_pool(conv2_out))
    conv4_out = self.conv4(self.max_pool(conv3_out))
    conv5_out = self.conv5(self.max_pool(conv4_out))

    conv5m_out_ = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
    conv4m_out = self.conv4m(conv5m_out_)
    
    conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
    conv3m_out = self.conv3m(conv4m_out_)

    conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
    conv2m_out = self.conv2m(conv3m_out_)

    conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
    conv1m_out = self.conv1m(conv2m_out_)

    ## Design your last layer & activations
    outconv_out = self.outconv(conv1m_out)    
    
#     print('dimension of x', x.shape)
#     print('dimension of conv1_out', conv1_out.shape)
#     print('dimension of conv2_out', conv2_out.shape)
#     print('dimension of conv3_out', conv3_out.shape)

#     print('dimension of conv4_out', conv4_out.shape)
#     print('dimension of conv4m_out_', conv4m_out_.shape)

#     print('dimension of conv3m_out', conv3m_out.shape)
#     print('dimension of conv3m_out_', conv3m_out_.shape)

#     print('dimension of conv2m_out', conv2m_out.shape)
#     print('dimension of conv2m_out_', conv2m_out_.shape)

#     print('dimension of conv1m_out', conv1m_out.shape)
#     print('dimension of outconv_out', outconv_out.shape)

    return outconv_out