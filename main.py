import os
import math
import logging
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataloader import VQA_Dataset

import argparse
from model import SAN

parser = argparse.ArgumentParser()

parser.add_argument_group('Optimization related arguments')
parser.add_argument('--num_epochs', type=int, default=10, help='Total number of epochs')
parser.add_argument('--num_iters', type=int, default=75001, help='Total number of iterations')
parser.add_argument('--batch_size', type=int, default=100, help='Batch Size')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning Rate')
parser.add_argument('--decay_factor', type=float, default=0.99997592083, help='Decay factor of learning rate')
parser.add_argument('-min_lr', default=5e-5, type=float, help='Minimum learning rate')
parser.add_argument('--lr_decay_start', type=int, default=-1, help='When to start decay of learning rate')
parser.add_argument('-weight_init', default='xavier', choices=['xavier', 'kaiming'], help='Weight initialization strategy')

parser.add_argument_group('SAN model related arguments')
parser.add_argument('--embedding_length', type=int, default=512, help='Embedding length of features')
parser.add_argument('--lstm_layers', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--image_dim', type=int, default=512, help='Image dimension')
parser.add_argument('--hidden_size', type=int, default=1024, help='Size of the hidden layer of LSTM')
parser.add_argument('--attention_size', type=int, default=512, help='Size of the attention vector')
parser.add_argument('--num_answers', type=int, default=1000, help='Number of top answers to be included')

parser.add_argument_group('Checkpoints related arguments')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='Checkpoints directory')
parser.add_argument('--save_step', type=int, default=100, help='Save checkpoints after save_step iterations')

parser.add_argument_group('Logger related arguments')
parser.add_argument('--log_path', type=str, default='logs', help='Log directory')
parser.add_argument('--log_step', type=int, default=100, help='Save log INFO after log_step iterations')

# Add command line arguments defined in data_loader.py
VQA_Dataset.extend_args(parser)
args = parser.parse_args()


# Logger setting
logger = logging.getLogger('__name__')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler = logging.FileHandler(args.log_dir)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

print ('Loading training dataset .............')
dataset = VQA_Dataset(args, 'train')
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

iter_per_epoch = math.ceil(dataset.num_data_points['train'] / args.batch_size)

print ('Building Model ................')
model = SAN(args, args.batch_size)

num_params = 0
for p in model.parameters():
    num_params += p.numel()

print(model)
print("Total number of parameters: {}".format(num_params))

criterion = nn.CrossEntropyLoss()
optim = optim.Adam(list(model.parameters(), lr=args.lr))
scheduler = lr_scheduler.StepLR(optim, step_size=1, gamma=args.decay_factor)
model.cuda()
criterion.cuda()
model.train()

train_start_time = datetime.datetime.utcnow()
print ('Training start time : {}'.format(datetime.datetime.strftime(train_start_time, '%d-%b-%Y-%H:%M:%S')))

for epoch in range(args.num_epochs):
    for idx, batch in enumerate(dataloader):
        optim.zero_grad()
        prediction = model(batch)
        loss = criterion(prediction, batch['ans'])
        loss.backward()
        optim.step()

        if optim.param_groups[0]['lr'] > args.min_lr:
            scheduler.step()

        if idx % 100 == 0:
            print ('[{}] [Epoch : {:3d}] [Iter : {:6d}] [Loss : {:6f}] [lr : {:7f}]'.format(
                datetime.datetime.utcnow() - train_start_time, epoch,
                (epoch - 1) * iter_per_epoch + idx, loss, optim.param_groups[0]['lr']))

        if idx % args.log_step == 0:
            end_time = datetime.datetime.utcnow() - train_start_time
            log = 'Elapsed [{}], Epoch [{}/{}], Idx [{}], Loss [{:4f}]'.format(end_time, epoch + 1,
                                                                               args.num_epochs, idx, loss)
            logger.info(log)
            print log

        # Save checkpoints and final model
        if idx % args.save_step == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optim.state_dict()},
                os.path.join(args.checkpoint_path, 'model_epoch_{}.pth'.format(epoch)))

# Save final model
torch.save({
    'model': model.state_dict(),
    'optimizer': optim.state_dict()},
    os.path.join(args.checkpoint_path, 'final_model.pth'))

# Testing model

print ('Loading testing dataset .............')
dataset = VQA_Dataset(args, 'test')
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
model.eval()

test_start_time = datetime.datetime.utcnow()
print ('Testing start time : {}'.format(datetime.datetime.strftime(test_start_time, '%d-%b-%Y-%H:%M:%S')))

with torch.no_grad():
    for idx, batch in enumerate(dataloader):
        prediction = model(batch)
        loss = criterion(prediction, batch['ans'])

        if idx % 100 == 0:
            print ('[{}] [Iter : {:6d}] [Loss : {:6f}]'.format(datetime.datetime.utcnow() - test_start_time, idx, loss))
