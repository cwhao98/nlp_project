import os
import torch
import torch.nn as nn
import random 
import time
from torch.nn.utils.rnn import pad_sequence
from abc import ABCMeta, abstractmethod

pad = 0 # here is the same as OOV

class Base(nn.Module, metaclass=ABCMeta):

    def __init__(self, args):
        
        super(Base, self).__init__()

        self.args = args
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)

    def run(self, train_data, val_data):
        best_acc, best_acc_epoch = 0., 0
        for epoch in range(self.args.num_epoch):
            # training
            self.run_train(train_data)

            # validate 
            acc = self.run_test(val_data)

            # save best model
            if acc > best_acc:
                best_acc, best_acc_epoch = acc, epoch+1
                self.save(epoch=epoch+1)

            print('epoch: %d, accuracy: %.4f' % (epoch+1, acc))
            if (epoch+1)  % 10 == 0:
                print('\nbest acc epoch: %d, acc: %.4f\n' % (best_acc_epoch, best_acc))

    @abstractmethod
    def forward(self, inputs):
        pass

    def run_train(self, data):
        self.train()
        random.shuffle(data)
        for batch in self.iterate(data, self.args.batchsize):
            self.optimizer.zero_grad()
            output = self.forward(batch['feat'])
            loss = self.loss(output, batch['label'])
            # print(output, batch['label'])
            loss.backward()
            # torch.nn.utils.clip_grad_norm(self.parameters(), 40.)
            self.optimizer.step()
           
            # print('Training loss: %.2f' % loss.cpu().item())
            # time.sleep(1)
            # break

    def run_test(self, data):
        self.eval()
        acc = 0.
        for batch in self.iterate(data, self.args.batchsize):
            output = self.forward(batch['feat'])
            acc += (output.argmax(dim=1) == batch['label']).float().sum().cpu().item()
        return acc / len(data)

    def iterate(self, data, batchsize):
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        for i in range(0, len(data), batchsize):
            tasks = data[i:i+batchsize]
            # feature
            feat = [torch.tensor(task[0], device=device, dtype=torch.long) for task in tasks]
            pad_feat = pad_sequence(feat, batch_first=True, padding_value=pad)
            # label
            label = torch.stack([torch.tensor(task[1][0] if self.args.num_emotion == 2 else task[1][1], device=device, dtype=torch.long) for task in tasks], dim=0)
            batch = {'feat': pad_feat, 'label': label}
            yield batch

    def save(self, epoch=0):
        path = os.path.join(self.args.log_dir, 'best_val_acc')
        torch.save({'epoch':epoch, 'state_dict': self.state_dict(), 'optimizer':self.optimizer.state_dict()}, path)
        print('save model to:', path)

    def load(self, path=None):
        path = path or os.path.join(self.args.log_dir, 'best_val_acc')
        ckpt = torch.load(path)
        self.load_state_dict(ckpt['state_dict'])
        print('load model from:', path)







