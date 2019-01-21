import numpy as np
import torch.nn as nn
import os
from torch.nn import MaxPool2d, LeakyReLU, Conv2d, Dropout2d, InstanceNorm2d, LogSoftmax
import torch
import torch.nn.functional as F
import net_utils
import argparse
import ocr_gen
from torch.nn import CTCLoss
from torch.autograd import Variable
import editdistance
import cv2


f = open('codec.txt', 'r')
codec = f.readlines()[0]
f.close()
print(len(codec))


def print_seq_ext(wf):
    prev = 0
    word = ''
    start_pos = 0
    end_pos = 0
    dec_splits = []
    hasLetter = False
    for cx in range(0, wf.shape[0]):
        c = wf[cx]
        if prev == c:
            if c > 2:
                end_pos = cx
            continue
        if 3 < c < len(codec):
            ordv = codec[c - 4]
            char = ordv
            if char == ' ':
                if hasLetter:
                    dec_splits.append(cx + 1)
            else:
                hasLetter = True
            word += char
            end_pos = cx
        elif c > 0:
            if hasLetter:
                dec_splits.append(cx + 1)
                word += ' '
                end_pos = cx

        if len(word) == 0:
            start_pos = cx

        prev = c
    conf2 = [start_pos, end_pos + 1]
    return word.strip(), np.array([conf2]), np.array([dec_splits])


class OCRModel(nn.Module):
    def __init__(self):
        super(OCRModel, self).__init__()
        self.conv1 = Conv2d(1, 32, (3, 3), padding=1, bias=False)
        self.conv2 = Conv2d(32, 32, (3, 3), padding=1, bias=False)
        self.conv3 = Conv2d(32, 64, (3, 3), padding=1, bias=False)
        self.conv4 = Conv2d(64, 64, (3, 3), padding=1, bias=False)
        self.conv5 = Conv2d(64, 128, (3, 3), padding=1, bias=False)
        self.conv6 = Conv2d(128, 128, (3, 3), padding=1, bias=False)
        self.conv7 = Conv2d(128, 256, (3, 3), padding=1, bias=False)
        self.conv8 = Conv2d(256, 256, (3, 3), padding=1, bias=False)
        self.conv9 = Conv2d(256, 512, (2, 3), padding=(0, 1), bias=False)
        self.conv10 = Conv2d(512, 512, (1, 5), padding=(0, 2), bias=False)
        self.conv11 = Conv2d(512, 94, (1, 1), padding=(0, 0))

        self.conv_attenton = Conv2d(512, 1, (1, 1), padding=0)

        self.batch1 = InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.batch2 = InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.batch3 = InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.batch5 = InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.batch7 = InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.batch8 = InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.batch9 = InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.batch10 = InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.drop1 = Dropout2d(p=0.2, inplace=False)
        self.leaky = LeakyReLU(negative_slope=0.01, inplace=False)
        self.max1 = MaxPool2d((2, 2), stride=None)
        self.max2 = MaxPool2d((2, 1), stride=(2, 1))

    def forward(self, x):
        try:
            x = x.cuda(0)
        except:
            pass
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.leaky(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.leaky(x)
        x = self.max1(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.leaky(x)

        x = self.conv4(x)
        x = self.leaky(x)
        x = self.conv4(x)
        x = self.leaky(x)

        x = self.max1(x)
        x = self.conv5(x)
        x = self.batch5(x)
        x = self.leaky(x)

        x = self.conv6(x)
        x = self.leaky(x)
        x = self.conv6(x)
        x = self.leaky(x)

        x = self.max2(x)
        x = self.conv7(x)
        x = self.batch7(x)
        x = self.leaky(x)

        x = self.conv8(x)
        x = self.leaky(x)
        x = self.conv8(x)
        x = self.leaky(x)

        x = self.max2(x)
        x = self.conv9(x)
        x = self.batch9(x)
        x = self.leaky(x)
        x = self.conv10(x)

        x = self.batch10(x)
        x = self.leaky(x)

        x = self.drop1(x)
        x = self.conv11(x)
        x = x.squeeze(2)

        x = x.permute(0, 2, 1)
        # print(x.size(), "#")
        # a += 1
        # y = x
        # x = x.contiguous().view(-1, x.data.shape[2])
        x = F.log_softmax(x, dim=-1)
        # x = x.view_as(y)
        x = x.permute(0, 2, 1)

        return x


base_lr = 0.001
lr_decay = 0.99
momentum = 0.9
weight_decay = 0.0005
batch_per_epoch = 1000
disp_interval = 10

buckets = [54, 80, 124, 182, 272, 410, 614, 922, 1383, 2212]


def test(net, list_file='/home/liepieshov/dataset/en_words/test.csv'):
    net = net.eval()
    fout = open('./valid.txt', 'w')

    dir_name = os.path.dirname(list_file)
    images, bucket, label = ocr_gen.get_info_csv(list_file)

    it = 0
    correct = 0
    ted = 0
    gt_all = 0
    while True:

        imageNo = it

        if imageNo >= len(images):
            break

        image_name = images[imageNo]
        gt_txt = label[imageNo]

        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(image_name)
            continue
        if img.shape[0] > img.shape[1] * 2 and len(gt_txt) > 3:
            img = np.transpose(img)
            img = cv2.flip(img, flipCode=1)

        scaled = np.expand_dims(img, axis=2)
        scaled = np.expand_dims(scaled, axis=0)

        scaled = np.asarray(scaled, dtype=np.float)
        scaled /= 128
        scaled -= 1

        scaled_var = net_utils.np_to_variable(scaled, is_cuda=args.cuda, volatile=False).permute(0, 3, 1, 2)
        ctc_f = net(scaled_var)
        ctc_f = ctc_f.data.cpu().numpy()
        ctc_f = ctc_f.swapaxes(1, 2)

        labels = ctc_f.argmax(2)
        det_text, conf, dec_s = print_seq_ext(labels[0, :])

        it += 1

        edit_dist = editdistance.eval(str(det_text).lower(), str(gt_txt).lower())
        ted += edit_dist
        gt_all += len(str(gt_txt))

        if str(det_text).lower() == str(gt_txt).lower():
            correct += 1
        else:
            print('{0} - {1} / {2:.2f} - {3:.2f}'.format(det_text, gt_txt, correct / float(it), ted / 3.0))

        fout.write('{0}|{1}|{2}|{3}\n'.format(os.path.basename(image_name), gt_txt, det_text, edit_dist))

    print('Test accuracy: {0:.3f}, {1:.2f}, {2:.3f}'.format(correct / float(it), ted / 3.0, ted / float(gt_all)))

    fout.close()
    net.train()

from tqdm import tqdm
def main(opts):
    # pairs = c1, c2, label

    model_name = 'ICCV_OCR'
    net = OCRModel()

    if opts.cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
    step_start = 0
    if os.path.exists(opts.model):
        print('loading model from %s' % args.model)
        step_start, learning_rate = net_utils.load_net(args.model, net)
    else:
        learning_rate = base_lr
    print('train')
    net.train()

    # test(net)

    ctc_loss = CTCLoss(blank=0).cuda()

    data_generator = ocr_gen.get_batch(num_workers=opts.num_readers,
                                       batch_size=opts.batch_size,
                                       train_list=opts.train_list, in_train=True)

    train_loss = 0
    cnt = 0
    tq = tqdm(range(step_start, 10000000))
    for step in tq:

        # batch
        images, labels, label_length = next(data_generator)
        im_data = net_utils.np_to_variable(images, is_cuda=opts.cuda, volatile=False).permute(0, 3, 1, 2)
        labels_pred = net(im_data)

        # backward
        '''
    acts: Tensor of (seqLength x batch x outputDim) containing output from network
        labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        act_lens: Tensor of (batch) containing label length of each example
    '''
        torch.backends.cudnn.deterministic = True
        probs_sizes = Variable(
            torch.IntTensor([(labels_pred.permute(2, 0, 1).size()[0])] * (labels_pred.permute(2, 0, 1).size()[1]))).long()
        label_sizes = Variable(torch.IntTensor(torch.from_numpy(np.array(label_length)).int())).long()
        labels = Variable(torch.IntTensor(torch.from_numpy(np.array(labels)).int())).long()
        optimizer.zero_grad()
        #probs = nn.functional.log_softmax(labels_pred, dim=94)

        labels_pred = labels_pred.permute(2, 0, 1)

        loss = ctc_loss(labels_pred, labels, probs_sizes, label_sizes) / opts.batch_size  # change 1.9.
        if loss.item() == np.inf:
            continue
        #
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        cnt += 1
        # if step % disp_interval == 0:
        #     train_loss /= cnt
        #     print('epoch %d[%d], loss: %.3f, lr: %.5f ' % (
        #         step / batch_per_epoch, step, train_loss, learning_rate))
        #
        #     train_loss = 0
        #     cnt = 0
        tq.set_description('epoch %d[%d], loss: %.3f, lr: %.5f ' % (
            step / batch_per_epoch, step, train_loss/cnt, learning_rate))
#
        if step > step_start and (step % batch_per_epoch == 0):
            save_name = os.path.join(opts.save_path, '{}_{}.h5'.format(model_name, step))
            state = {'step': step,
                     'learning_rate': learning_rate,
                     'state_dict': net.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(state, save_name)
            print('save model: {}'.format(save_name))

            test(net)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_list', default='/home/liepieshov/dataset/en_words/train.csv')
    parser.add_argument('-valid_list', default='/home/liepieshov/dataset/en_words/test.csv')
    parser.add_argument('-save_path', default='./')
    parser.add_argument('-model', default='/mnt/textspotter/tmp/DeepSemanticText/backup2/MLT_OCR_SYNTH_2302000.h5')
    parser.add_argument('-debug', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=30)
    parser.add_argument('-num_readers', type=int, default=8)
    parser.add_argument('-cuda', type=bool, default=True)

    args = parser.parse_args()
    main(args)
