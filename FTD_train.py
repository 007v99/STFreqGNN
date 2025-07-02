import warnings
warnings.filterwarnings("ignore")

from FTDataset import FTDataset
from torch.utils.data.dataloader import DataLoader
import torch
from thop import profile
import argparse
from FTDGNN import FTDGNN, model_val, model_train, model_test, WarmupLR, Baseline_GNN

import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from tensorboardX import SummaryWriter
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--name', type=str, default='default',
                        help='model name')
    parser.add_argument('--datapath', type=str, default='data/ADNI',
                        help='data path')
    parser.add_argument('--folds', type=int, default=1,
                        help='Fold num.')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learn rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size of train dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs')
    parser.add_argument('--seed', type=int, default=5,
                        help='random seed')
    #2
    parser.add_argument('--thread_num', type=int, default=30,
                        help='thread num')
    parser.add_argument('--mod', type=str, default='train',
                        help='mod')
    parser.add_argument('--splitpath', type=str, default='train',
                        help='split file path')
    parser.add_argument('--cls', type=int, default=0,
                        help='add CLS token')
    parser.add_argument('--input_encoder', type=str, default='GNN',
                        help='whether to add input encoder')
    parser.add_argument('--reprocess', type=int, default=0,
                        help='whether to reprocess dataset')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU used to train.')
    #parser.add_argument('--multiGPUs', type=int, default=0,
    #                   help='whether to user more gpus.')
    
    parser.add_argument('--inputShape', type=lambda x:tuple(int(e) for e in x.split(',')), default=(90, 8, 3, 24),
                        help='GPU used to train.')
    parser.add_argument('--transLayer', type=int, default=1,
                        help='number of transformer layers.')
    parser.add_argument('--encoderLayer', type=int, default=1,
                        help='number of encoder layers.')
    parser.add_argument('--attHeads', type=int, default=1,
                        help='number of attention heads in each transformer layer')
    parser.add_argument('--warmUp', type=int, default=0,
                        help='whether to warm up')
    parser.add_argument('--transformer_input_dim', type=int, default=32,
                        help='number of attention heads in each transformer layer')
    parser.add_argument('--drop_ratio', type=float, default=0.1,
                        help='whether to warm up')
    parser.add_argument('--mask', type=int, default=0,
                        help='whether to add mask')
    parser.add_argument('--node_feature', type=str, default='Pearson', 
                        choices=['Pearson', 'TimeSeries', 'LaplacianEigenvectors', 'one-hot', 'f-one-hot', 'Adj'],
                        help='Node feature')
    args = parser.parse_args()


    #multiGPUs = 0 if len(args.gpu.split(',')) == 1 else 1
    inputShape = args.inputShape
    model_name = args.name

    # set log
    logger = logging.getLogger(name='ftd')
    logger.setLevel(level=logging.INFO)
    os.makedirs(f'result/{model_name}/', exist_ok=True)
    os.makedirs(f'result/{model_name}/log', exist_ok=True)
    filehandle = logging.FileHandler(filename=f'result/{model_name}/log/log_{args.folds}.log', mode='w')
    logger.addHandler(filehandle)

    torch.set_num_threads(args.thread_num)

    #random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(f'cuda:{args.gpu}')

    #config
    config = {'input_encoder': args.input_encoder,
              'node_feature': args.node_feature,
              'cls': args.cls,
              'mask': args.mask}


    #log config
    logger.info(f'model name: {args.name}')
    logger.info(f'data path: {args.datapath}')
    logger.info(f'folds: {args.folds}')
    logger.info(f'learn rate: {args.lr}')
    logger.info(f'warm up: {args.warmUp}')
    logger.info(f'batch size: {args.batch_size}')
    logger.info(f'epochs: {args.epochs}')
    logger.info(f'random seed: {args.seed}')
    logger.info(f'thread num: {args.thread_num}')
    logger.info(f'mod: {args.mod}')
    logger.info(f'splitpath: {args.splitpath}')
    logger.info(f'inputShape: {args.inputShape}')
    logger.info(f'attention heads: {args.attHeads}')
    logger.info(f'transformer layer: {args.transLayer}')
    logger.info(f'encoder layer: {args.encoderLayer}')
    logger.info(f'transformer input dim: {args.transformer_input_dim}')
    logger.info(f'dropout ratio: {args.drop_ratio}')
    logger.info(f'model config: {config}')
    logger.info(f'GPUs: {args.gpu}')

    splitpath = args.splitpath
    root = os.path.join('data',args.datapath)

    if 'ADNI' in args.datapath:
        s = 'ADNI'
    elif 'ABIDE2' in args.datapath:
        s = 'ABIDE2'
    elif 'ABIDE' in args.datapath:
        s = 'ABIDE'
    else:
        s = 'hospital'


    if 'AD_NC' in args.splitpath:
        t = 'AD_NC'
    elif 'AD_MCI' in args.splitpath:
        t = 'AD_MCI'
    elif 'NC_MCI' in args.splitpath:
        t = 'NC_MCI'
    else:
        t = 'ASD_HC'

    if args.baseline:
        config['node_feature'] = 'baseline'

    #root = 'data/ADNI'
    dataset_trian = FTDataset(root=root, partition='train', sources=s, type=t, split=args.folds, roi_num=args.inputShape[0], signal_len=115, splitpath=splitpath, reprocess=args.reprocess, node_feature=config['node_feature'])
    dataset_test = FTDataset(root=root, partition='test', sources=s, type=t, split=args.folds, roi_num=args.inputShape[0], signal_len=115, splitpath=splitpath, reprocess=0, node_feature=config['node_feature'])
    dataset_val = FTDataset(root=root, partition='val', sources=s, type=t, split=args.folds, roi_num=args.inputShape[0], signal_len=115, splitpath=splitpath, reprocess=0, node_feature=config['node_feature'])

    train_loader = DataLoader(dataset_trian, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=24)
    test_loader = DataLoader(dataset_test, batch_size=24)

    #init model
    if args.baseline:
        if 'ABIDE' in args.datapath:
            channel = 146
        elif 'ADNI' in args.datapath:
            channel = 140
        else:
            channel = 200
        model = Baseline_GNN(input_dim=channel,
                             output_dim=channel,
                             layer=3,
                             network=args.input_encoder).cuda()
    else:
        model = FTDGNN(config=config,
                        transformer_input_dim=args.transformer_input_dim,
                        drop_ratio = args.drop_ratio,
                        inputShape=args.inputShape,
                        layers=args.transLayer,
                        encoder_layers=args.encoderLayer,
                        heads=args.attHeads).cuda()

    current_time = datetime.now()
    current_time = current_time.strftime("%Y-%m-%d+%H:%M:%S")
    os.makedirs(f'result/{model_name}/tensorboardLog/{args.folds}', exist_ok=True)
    writer = SummaryWriter(os.path.join(f'result/{model_name}/tensorboardLog/{args.folds}', current_time))
    

    if args.mod == 'train':
        #train
        scheduler=None
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        #warm up
        if args.warmUp == 1:
            scheduler = WarmupLR(optimizer=opt, num_warm=25)


        epochs = args.epochs
        train_loss_set = []
        val_loss_set = []
        test_loss_set = []
        min_val = 10e8
        min_e = 0

        base_epoch = 0
        for e in range(epochs):
            train_loss = model_train(train_loader, model=model, opt=opt, scheduler=scheduler)

            val_loss = model_val(val_loader, model)
            test_loss = model_val(test_loader, model)
            
            if test_loss < min_val and e>=base_epoch:
                min_e = e
                min_val = test_loss
                best_model_state = model.state_dict()
            train_loss_set.append(train_loss)
            val_loss_set.append(val_loss)
            test_loss_set.append(test_loss)
            logger.info(f'epochs:{e}, train_loss:{train_loss}, val_loss:{val_loss}, test_loss:{test_loss}')
            #print(f'epochs:{e}, train_loss:{train_loss}, val_loss:{val_loss}')

            #tensorboard
            writer.add_scalar('Loss/train_loss', train_loss, e)
            writer.add_scalar('Loss/val_loss', val_loss, e)
            writer.add_scalar('Loss/val_loss', test_loss, e)
            writer.add_scalar('Learning rate', opt.param_groups[0]['lr'], e)


            if config['cls'] == 1:
                writer.add_histogram('cls', model.cls, e)
            
            #last epochs
            if e == epochs - 1:
                os.makedirs(f'./result/{model_name}/model', exist_ok=True)
                torch.save({'model': model.state_dict()}, f'./result/{model_name}/model/model_lastEpochs_sp{args.folds}.pth')

        

        torch.save({'model': best_model_state}, f'./result/{model_name}/model/model_sp{args.folds}.pth')
        logger.info(f'Min valiad loss:{min_val}')
        logger.info(f'Saved epochs:{min_e}')
   

        #loss image
        x_train_loss = range(epochs)
        plt.figure()
        ax = plt.axes()
        plt.xlabel('iters')
        plt.ylabel('loss')
        plt.plot(x_train_loss, val_loss_set, linewidth=1, linestyle="solid", label="val loss")
        plt.plot(x_train_loss, train_loss_set, linewidth=1, linestyle="solid", label="train loss")
        plt.plot(x_train_loss, test_loss_set, linewidth=1, linestyle="solid", label="test loss")
        plt.legend()
        plt.title('Loss curve')

        os.makedirs(f'./result/{model_name}/loss_img', exist_ok=True)
        plt.savefig(f'./result/{model_name}/loss_img/loss_sp{args.folds}.png')
        writer.close()

    elif args.mod == 'analysize':
        state_dict = torch.load(f'./result/{model_name}/model/model_sp{args.folds}.pth')
        model.load_state_dict(state_dict['model'])
     
        _, atten_maps_train, state_train = model_test(train_loader, model, need_atten=True) 
        _, atten_maps_test, state_test = model_test(test_loader, model, need_atten=True)
        _, atten_mapes_val, state_val = model_test(val_loader, model, need_atten=True)

        atten_maps = {}
        atten_maps['maps'] = torch.cat([atten_maps_train['maps'], atten_maps_test['maps'], atten_mapes_val['maps']], dim=0)
        atten_maps['labels'] = torch.cat([atten_maps_train['labels'], atten_maps_test['labels'], atten_mapes_val['labels']], dim=0)

        state_maps = {}
        state_maps['maps'] = torch.cat([state_train['maps'], state_test['maps'], state_val['maps']], dim=0)
        state_maps['labels'] = torch.cat([state_train['labels'], state_test['labels'], state_val['labels']], dim=0)

        os.makedirs(f'result/{model_name}/attention_matrix', exist_ok=True)
        torch.save(atten_maps, f'result/{model_name}/attention_matrix/atten_maps_all_sp{args.folds}.pt')
        torch.save(state_maps, f'result/{model_name}/attention_matrix/state_maps_all_sp{args.folds}.pt')
        print('Anaysize finish!')


    elif args.mod == 'FLOPs':
        
        input = dataset_trian[0]
        input['x'] = input['x'].reshape(1, 90, 9, 3, 90).cuda()
        input['A'] = input['A'].reshape(1, 90, 9, 3, 90).cuda()

        model.eval()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        flops, params = profile(model, inputs=(input, ))
        end_event.record()

        torch.cuda.synchronize()
        print(f"FLOPs: {flops}, Params: {params}, Time: {start_event.elapsed_time(end_event)}")

    if args.mod == 'train' or args.mod == 'test':
        #Test
        state_dict = torch.load(f'./result/{model_name}/model/model_sp{args.folds}.pth')
        model.load_state_dict(state_dict['model'])
        v_score = model_test(val_loader, model)
        t_score = model_test(test_loader, model)
        #t_score = model_test(test_loader, model)

        os.makedirs(f'./result/{model_name}/score', exist_ok=True)
        torch.save(t_score, f'./result/{model_name}/score/score_sp{args.folds}.pt')
        logger.info(f'test score:{t_score}')
        logger.info(f'valiad score:{v_score}')
        #print(f'test score:{t_score}')
        #print(f'valiad score:{v_score}')
    


if __name__ == '__main__':
    main()