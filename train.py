
import warnings
warnings.filterwarnings('ignore')
import time
import os
import torch
from torch.optim import SGD, Adam
from data.dataset import ImageDataset
from data.dataload import DataLoader
from net.dbnet import Db
from util.adjust_lr import DecayLearningRate



def main():

    args = {'distributed': False,
            'imgdir': ['./resource/ann_sample/img'],
            'gtdir': ['./resource/ann_sample/gt'],
            'batch_size': 8, 'num_workers': 4, 'epochs': 5000,
            'lr': 0.01, 'momentum':0.9, 'weight_decay': 0.00005,
            'resume': None,
            'save_margin': 100, 'save_dir': './ckt', 'log_interval': 1,
            'pre_train': './ckt/epoch_100000.pth'
            }
    print("config :")
    print(args)
    dataset = ImageDataset(args['imgdir'], args['gtdir'])
    dataloader = DataLoader(dataset, args['batch_size'], num_workers=args['num_workers'], distributed=False)
    net = Db(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), distributed=False, local_rank=0)
    params = torch.load(args['pre_train'])
    net.load_state_dict(params)


    optimizer = SGD(net.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])

    decay_lr = DecayLearningRate(lr=args['lr'], epochs=args['epochs'])
    net.train()
    epoch = 1
    while epoch < args['epochs']:
        step = 1
        for batch in dataloader:
            try:
                lr = float(decay_lr.get_learning_rate(epoch, step))
                for group in optimizer.param_groups:
                    group['lr'] = lr

                optimizer.zero_grad()

                loss, pred, metrics = net.forward(batch, training=True)

                loss.backward()
                optimizer.step()
            except Exception as e:
                print(e)
                print("err")
                continue
            if step % args['log_interval'] == 0:
                    print('step: %6d, epoch: %3d, loss: %f, lr: %f' % (step, epoch, loss.item(), lr))

            if epoch % args['save_margin'] == 0:
                torch.save(net.state_dict(), '%s/epoch_%d.pth' % (args['save_dir'], epoch))

            step += 1

        epoch += 1

if __name__ == '__main__':
    main()

