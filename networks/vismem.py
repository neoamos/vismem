
import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
from deeplab_resnet import Res_Deeplab

class VisMem(nn.Module):

    def __init__(self,input_size,gru_input_size,hidden_size,kernel_size,cuda_flag):
        super(VisMem,self).__init__()
        self.input_size      = input_size
        self.gru_input_size  = gru_input_size
        self.cuda_flag       = cuda_flag
        self.hidden_size     = hidden_size
        self.kernel_size     = kernel_size
        self.ConvGates       = nn.Conv2d(self.gru_input_size + self.hidden_size,2 * self.hidden_size,3,padding=self.kernel_size//2-2)
        self.Conv_ct         = nn.Conv2d(self.gru_input_size + self.hidden_size,self.hidden_size,3,padding=self.kernel_size//2-2)
        self.Input_reduction_1 = nn.Conv2d(self.input_size, 1024, 1)
        self.Input_reduction_2 = nn.Conv2d(1024, 512, 1)
        self.Input_reduction_3 = nn.Conv2d(512, gru_input_size-1, 1)
        self.Mask_reduction  = nn.Conv2d(self.hidden_size, 2, 1)
        dtype                = torch.FloatTensor

    def forward(self,appearance,mask,hidden):
        self.rescale = nn.UpsamplingBilinear2d(size = (appearance.size()[2], appearance.size()[3] ))
        mask = self.rescale(mask)
        #input = torch.cat([appearance, mask], dim=1)
        #input = f.tanh(self.Input_reduction_1(appearance))
        #input = f.tanh(self.Input_reduction_2(input))
        #input = f.tanh(self.Input_reduction_3(input))
        input = torch.cat([appearance, mask], dim=1)
        if hidden is None:
           size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
           if self.cuda_flag  == True:
              hidden    = Variable(torch.zeros(size_h)).cuda()
           else:
              hidden    = Variable(torch.zeros(size_h))
        c1           = self.ConvGates(torch.cat((input,hidden),1))
        (rt,ut)      = c1.chunk(2, 1)
        reset_gate   = f.sigmoid(rt)
        update_gate  = f.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate,hidden)
        p1           = self.Conv_ct(torch.cat((input,gated_hidden),1))
        ct           = f.tanh(p1)
        next_h       = torch.mul(update_gate,hidden) + (1-update_gate)*ct
        mask_pred    = self.Mask_reduction(next_h)
        return (mask_pred, next_h)

#not updated for vismem yet
def test(num_seqs,channels_img,\
         size_image,max_epoch,model,cuda_test):
    input_image   = torch.rand(num_seqs,1,channels_img,size_image,size_image)
    target_image  = torch.rand(num_seqs,1,channels_img,size_image,size_image)
    print('\n\n ==> Create Autograd Variables:')
    input_gru     = Variable(input_image)
    target_gru    = Variable(target_image)
    if cuda_test == True:
       input_gru  = input_gru.cuda()
       target_gru = target_gru.cuda()
    print('\n\n ==> Create a MSE criterion:')
    MSE_criterion = nn.MSELoss()
    if cuda_test == True:
        print("==> test on the GPU active")
        MSE_criterion = MSE_criterion.cuda()
    err           = 0
    for e in xrange(max_epoch):
        for time in xrange(num_seqs):
            h_next = model(input_gru[time], None)
            err   += MSE_criterion(h_next [0], target_gru[time])
            print(err.data[0])
