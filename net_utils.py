import numpy as np
import torch
from torch.autograd import Variable


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, volatile=False, requires_grad=False):
    v = Variable(torch.from_numpy(x).type(dtype), volatile=volatile, requires_grad=requires_grad)
    if is_cuda:
        v = v.cuda()
    return v


def load_net(fname, net, optimizer=None):
    sp = torch.load(fname)
    step = sp['step']
    try:
        learning_rate = sp['learning_rate']
    except:
        import traceback
        traceback.print_exc()
        learning_rate = 0.001
    opt_state = sp['optimizer']
    sp = sp['state_dict']
    for k, v in net.state_dict().items():
        try:
            param = sp[k]
            v.copy_(param)
        except:
            try:
                param = param.mean(1)
                param = param.unsqueeze(1)
                v.copy_(param)
            except:
                # param = sp[k]
                # if k == 'conv11.bias':
                #  v[:] = -50
                # v[0:3840] = param
                pass
            import traceback
            traceback.print_exc()

    if optimizer is not None:
        try:
            optimizer.load_state_dict(opt_state)
        except:
            import traceback
            traceback.print_exc()

    print(fname)
    return step, learning_rate
