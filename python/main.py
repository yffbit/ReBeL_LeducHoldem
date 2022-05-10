import torch
from model import Net
import rela
import time
from datetime import datetime
import os
import argparse
import logging

def input_output_dim():
    return 1 + 1 + 1 + 6 + 2 * 6, 6

def huber_loss(diff:torch.Tensor, delta=1):
    diff_abs = diff.abs()
    return (diff_abs>delta).float() * (2*delta*diff_abs - delta**2) + (diff_abs<=delta).float() * diff.pow(2)

if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    handle1 = logging.StreamHandler()
    handle2 = logging.FileHandler('../log/train.txt')
    fmt = logging.Formatter('%(asctime)s - %(message)s')
    handle1.setFormatter(fmt)
    handle2.setFormatter(fmt)
    log.addHandler(handle1)
    log.addHandler(handle2)
    log.info(rela)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path', required=True, type=str)
    # parser.add_argument('--n_actor_thread', type=int, default=4)
    # parser.add_argument('--actor_device', type=str, default='cpu')
    # parser.add_argument('--buffer_size', type=int, default=100000)
    input_dim, output_dim = input_output_dim()
    hidden_dims = [256, 256, 128]
    model = Net(input_dim, hidden_dims, output_dim)
    script_model = torch.jit.script(model)
    # save_name = 'test_model'
    # torch.save(model, save_name+'.ckpt')
    # torch.save(model.state_dict(), save_name+'.weight')
    # torch.jit.save(script_model, save_name+'.torchscript')
    # model = torch.load(save_name+'.ckpt')
    # x = torch.zeros(2, input_dim)
    # print(model.forward(x))
    # model.load_state_dict(torch.load(save_name+'.weight'))
    # print(model.forward(x))
    # script_model = torch.jit.load(save_name+'.torchscript')
    # print(script_model.forward(x))
    n_actor_thread = 4
    actor_device = 'cpu'
    models = []
    for i in range(n_actor_thread):
        act_model = Net(input_dim, hidden_dims, output_dim)
        act_model.to(actor_device)
        act_model = torch.jit.script(act_model)
        act_model.eval()
        models.append(act_model)
    model_lock = rela.ModelLock(models, actor_device)
    buffer = rela.ReplayBuffer(2**20, 2022)
    context = rela.Context()
    game = rela.Game()
    cfr_param = rela.CFRParam()
    cfr_param.discount = True
    cfr_param.alpha = 1.5
    cfr_param.beta = 0
    cfr_param.gamma = 2
    param = rela.SolverParam()
    # print(cfr_param)
    # print(param)
    # print((input_dim, output_dim))
    for i in range(n_actor_thread):
        net = rela.TrainDataNet(model_lock, i, buffer)
        loop = rela.DataLoop(game, cfr_param, param, net, 2022+i)
        context.push(loop)
    train_device = torch.device('cuda:0')
    context.start()
    model.train()
    model.to(train_device)
    opt = torch.optim.Adam(model.parameters())
    max_epochs = 10000
    purging_epochs = set([20])
    epoch_batch = 50
    batch_size = 512
    epoch_size = epoch_batch * batch_size
    decrease_lr_every = 400
    network_sync_epochs = 1
    train_gen_ratio = 4
    grad_clip = 5.0
    ckpt_every = 10
    # ckpt_dir = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_dir = '../model/test'
    time_format = '%Y%m%d %H:%M:%S'
    # while True:
    #     time.sleep(30)
    #     print('%s\t%d' % (datetime.now().strftime(time_format), buffer.num_add()))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    for epoch in range(max_epochs):
        if purging_epochs is not None and epoch in purging_epochs:
            size = buffer.size()
            buffer.pop_until(size // 2)
            log.info('purging buffer size at epoch %d : %d-->%d' % (epoch, size, buffer.size()))
        if epoch != 0 and epoch % decrease_lr_every == 0:
            for param_group in opt.param_groups:
                param_group['lr'] /= 2
            log.info('decrease lr at epoch %d ' % epoch)
        if train_gen_ratio is not None:
            while True:
                num_add = buffer.num_add()
                if num_add * train_gen_ratio >= epoch * epoch_size:
                    break
                log.info('%d*%d < %d*%d' % (num_add, train_gen_ratio, epoch, epoch_size))
                time.sleep(60)

        for batch in range(epoch_batch):
            data = buffer.sample(batch_size)
            # print('num_add:%d' % buffer.num_add())
            feature, target = data.feature.to(train_device), data.target.to(train_device)
            loss = (huber_loss(model(feature) - target)).mean()
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            log.info('%d\t%d\t%f' % (epoch, batch, loss))

        if (epoch+1) % network_sync_epochs == 0:
            model_lock.load_state_dict(model)
            log.info('update model at epoch %d ' % epoch)
        if (epoch+1) % ckpt_every == 0:
            ckpt_path = '%s/epoch_%d' % (ckpt_dir, epoch)
            torch.save(model, ckpt_path+'.ckpt')
            torch.jit.save(torch.jit.script(model), ckpt_path+'.torchscript')
            log.info('save ' + ckpt_path+'.ckpt')
            log.info('save ' + ckpt_path+'.torchscript')
    
    context.stop()
    context.join()
