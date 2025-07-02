import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from base_model.transformer import TransformerEncoderLayer, TransformerEncoder
from tqdm import *
from sklearn.metrics import roc_auc_score
from einops import repeat
import einops
from distributed_utils import reduce_value


class Mixer_encoder(nn.Module):
    def __init__(self, rois, input_dim, output_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(rois)
        self.node_mlp = nn.Sequential(nn.Linear(rois, rois),
                                      nn.GELU(),
                                      nn.Linear(rois, rois))
        
        self.ln2 = nn.LayerNorm(rois)
        self.channel_mlp = nn.Sequential(nn.Linear(input_dim, input_dim),
                                         nn.GELU(),
                                         nn.Linear(input_dim, output_dim))
        



    def forward(self, x):
        '''
        x (b roi T H roi)
        return_x (b roi1 T H output_dim)
        '''
        x_shape = x.shape
        
        x = einops.rearrange(x, 'b roi1 T H roi2 -> b T H roi2 roi1')

        x = self.node_mlp(self.ln1(x))

        x = einops.rearrange(x, 'b T H roi2 roi1 -> b roi1 T H roi2')

        x = self.channel_mlp(self.ln2(x)) #(b roi1 T H output_dim)

        return x



class Baseline_GNN(nn.Module):
    def __init__(self, input_dim, output_dim, network='GCN', rois=90, layer=3):
        super().__init__()
        self.gnns = nn.ModuleList([self.create_gnn(input_dim, input_dim, network) for _ in range(layer)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(input_dim) for _ in range(layer)])
        self.layer = layer
        self.MLP = nn.Sequential(nn.Linear(output_dim, output_dim//2),
                                 nn.BatchNorm1d(output_dim//2),
                                 nn.ReLU(),
                                 nn.Linear(output_dim//2, 2))


    def create_gnn(self, input_dim, output_dim, network):
        if network == 'GIN': return spGNN(input_dim, input_dim, output_dim)
        elif network == 'GCN': return GCNConv(input_dim, output_dim)
        elif network == 'GAT': return GATConv(input_dim, output_dim)
        else: raise

    def _collate_adjacency_nosparsity(self, a):
        '''
        a  ( b, T, roi,  roi)
        '''
        
        i_list = []
        v_list = []

        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                #thresholded_a = (_a > self.percentile(_a, 100-sparsity))
                thresholded_a = (_a != 0)
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)
        return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3])).coalesce()

    def forward(self, data, need_atten=False):
        '''
        x (b, roi, T)
        a (b, roi, roi)
        '''
        x, a = data['x'].to(torch.float32), data['A']
        roi = x.shape[1]
        a = einops.repeat(a, 'b roi1 roi2 -> b 1 roi1 roi2')
        #a = einops.rearrange(a, 'b roi1 roi2 -> b H roi1 roi2', H=1)
        a = self._collate_adjacency_nosparsity(a)
        x = einops.rearrange(x, 'b roi T -> (b roi) T')
        for h in range(self.layer):
            x = self.gnns[h](x, a)
            x = self.bns[h](x)
            x = F.elu(x)
        x = einops.rearrange(x, '(b roi) T -> b roi T', roi=roi)

        x = torch.mean(x, dim=1) #(b out)
        x = self.MLP(x)
        return x




class GFRN_encoder(nn.Module):
    def __init__(self, input_dim, output_dim, network='GNN', rois=90, H=3):
        super().__init__()
        self.rois = rois
        self.H = H
        self.rnns = nn.ModuleList([nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1, batch_first=True) for _ in range(rois*H)])
        #self.rnns = nn.ModuleList([nn.RNN(input_size=input_dim, hidden_size=output_dim, num_layers=1, batch_first=True) for _ in range(rois*H)])
        #self.rnns = nn.ModuleList([nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=1, batch_first=True) for _ in range(rois*H)])
        self.gnns = nn.ModuleList([self.create_gnn(input_dim, input_dim, network) for _ in range(H)])
        
    

    def _collate_adjacency_nosparsity(self, a):
        '''
        a  ( b, T, roi,  roi)
        '''
        
        i_list = []
        v_list = []

        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                #thresholded_a = (_a > self.percentile(_a, 100-sparsity))
                thresholded_a = (_a != 0)
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)
        return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3])).coalesce()



    def forward(self, x, a):
        '''
        a (b, roi, T, H, roi)
        x (b, roi, T, H, F)
        '''
        #a = einops.rearrange(a, 'b roi1 T H roi2 -> b (T H) roi1 roi2')
        a = einops.rearrange(a, 'b roi1 T H roi2 -> b T H roi1 roi2')
        
        x_shape = x.shape
        #x = einops.rearrange(x, 'b roi T H F -> (b T H roi) F')

        tmp = []
        for h in range(self.H):
            h_a = self._collate_adjacency_nosparsity(a[:,:,h,:,:])
            h_x = einops.rearrange(x[:,:,:,h,:], 'b roi T F -> (b T roi) F')
            h_x = self.gnns[h](h_x, h_a) #(b T roi) F
            h_x = einops.rearrange(h_x, '(b T roi) F -> b roi T F', b=x_shape[0], roi=x_shape[1], T=x_shape[2])
            tmp.append(h_x)
        x = torch.stack(tmp, dim=3) #b roi T H F
        x = einops.rearrange(x, 'b roi T H F -> b (roi H) T F')


        #x = einops.rearrange(x, '(b T H roi) F -> b roi T H F', b=x_shape[0], roi=x_shape[1], T=x_shape[2], H=x_shape[3])
        #x = einops.rearrange(x, '(b T H roi) F -> (b H) roi T F', b=x_shape[0], roi=x_shape[1], T=x_shape[2], H=x_shape[3])
        x = F.relu(x)
        
        tmp = []
        for r in range(self.rois * self.H):
            #RNN input -> (B T F)
            tmp.append(self.rnns[r](x[:,r,:,:])[0])
        x = torch.stack(tmp, dim=1) #b (roi H) T F

        x = einops.rearrange(x, 'b (roi H) T F -> b roi T H F', H=x_shape[3])
        return x

    
    def create_gnn(self, input_dim, output_dim, network):
        if network == 'GNN': return spGNN(input_dim, input_dim, output_dim)
        elif network == 'GFN': return spGNN(input_dim, input_dim, output_dim)
        elif network == 'GCN': return GCNConv(input_dim, output_dim)
        elif network == 'GAT': return GATConv(input_dim, output_dim)
        else: raise



class GRN_encoder(nn.Module):
    def __init__(self, input_dim, output_dim, network='GNN', rois=90, dropout=0.5):
        super().__init__()
        self.rois = rois
        self.rnns = nn.ModuleList([nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1, batch_first=True) for _ in range(rois)])
        self.gnn = self.create_gnn(input_dim, input_dim, network)
        self.dp = nn.Dropout(dropout)

    def _collate_adjacency_nosparsity(self, a):
        '''
        a  ( b, T, roi,  roi)
        '''
        
        i_list = []
        v_list = []

        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                #thresholded_a = (_a > self.percentile(_a, 100-sparsity))
                thresholded_a = (_a != 0)
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)
        return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3])).coalesce()



    def forward(self, x, a):
        '''
        a (b, roi, T, H, roi)
        x (b, roi, T, H, F)
        '''
        a = einops.rearrange(a, 'b roi1 T H roi2 -> b (T H) roi1 roi2')
        a = self._collate_adjacency_nosparsity(a)
        x_shape = x.shape
        x = einops.rearrange(x, 'b roi T H F -> (b T H roi) F')
        x = self.gnn(x, a)
        #x = einops.rearrange(x, '(b T H roi) F -> b roi T H F', b=x_shape[0], roi=x_shape[1], T=x_shape[2], H=x_shape[3])
        x = einops.rearrange(x, '(b T H roi) F -> (b H) roi T F', b=x_shape[0], roi=x_shape[1], T=x_shape[2], H=x_shape[3])
        x = F.relu(x)
        x = self.dp(x)
        f = []
        for i in range(self.rois):
            #RNN input -> (B T F)
            f.append(self.rnns[i](x[:,i,:,:])[0])
        x = torch.stack(f, dim=1) #((b H) roi T F)
        x = einops.rearrange(x, '(b H) roi T F -> b roi T H F', H=x_shape[3])
        return x

    
    def create_gnn(self, input_dim, output_dim, network):
        if network == 'GNN': return spGNN(input_dim, input_dim, output_dim)
        elif network == 'GFN': return spGNN(input_dim, input_dim, output_dim)
        elif network == 'GCN': return GCNConv(input_dim, output_dim)
        elif network == 'GAT': return GATConv(input_dim, output_dim)
        else: raise


class GNN_encoder(nn.Module):
    def __init__(self, input_dim, output_dim, layers, network='GNN', frq=False, H=3):
        super().__init__()
        self.layers = layers
        self.frequent_graph = (network=='GFN')

        self.gnns = nn.ModuleList([self.create_gnn(input_dim, input_dim, network) for _ in range(layers-1)]+[self.create_gnn(input_dim, output_dim, network)])

        #self.gnns = nn.ModuleList([GCNConv(input_dim, output_dim) for _ in range(layers)])
        '''self.gnns = nn.ModuleList()
        for _ in range(layers):
            self.gnns.append(spGNN(input_dim, input_dim, output_dim))'''

        self.bns = nn.ModuleList([nn.BatchNorm1d(output_dim) for _ in range(layers-1)])

        if frq: self.eps = [nn.Parameter(torch.Tensor([[0.0]])) for _ in range(H-1)]


    def create_gnn(self, input_dim, output_dim, network):
        if network == 'GNN': return spGNN(input_dim, input_dim, output_dim)
        elif network == 'GFN': return spGNN(input_dim, input_dim, output_dim)
        elif network == 'GCN': return GCNConv(input_dim, output_dim)
        elif network == 'GAT': return GATConv(input_dim, output_dim)
        else: raise




    def forward(self, x, a):
        '''
        a (b, roi, T, H, roi)
        x (b, roi, T, H, F)
        '''

        if self.frequent_graph:
            a = self.get_frequent_adj(a) #(b T H*roi, H*roi)
        else:
            a = einops.rearrange(a, 'b roi1 T H roi2 -> b (T H) roi1 roi2')


        a = self._collate_adjacency_nosparsity(a)

        x_shape = x.shape
        x = einops.rearrange(x, 'b roi T H F -> (b T H roi) F')

        for i in range(self.layers):
            x = self.gnns[i](x, a)
        
        x = einops.rearrange(x, '(b T H roi) F -> b roi T H F', b=x_shape[0], roi=x_shape[1], T=x_shape[2], H=x_shape[3])

        return x


    def get_frequent_adj(self, a):
        '''
        a (b roi1 T H roi2)
        ans_adj (b T (roi1*H) (roi2*H))
        '''
        b, roi1, T, H, roi2 = a.shape
        a = einops.rearrange(a, 'b roi1 T H roi2 -> b T H roi1 roi2')

        ans_adj = torch.zeros(b, T, roi1*H, roi1*H).cuda()

        ans_adj[:, :, 0:roi1, 0:roi1] = a[:, :, 0, :, :]
        for h in range(roi1, roi1*H, roi1):
            ans_adj[:, :, h:h+roi1, h:h+roi1] = a[:, :, h//roi1, :, :]

            ans_adj[:, :, h-roi1:h, h:h+roi1] = torch.eye(roi1) + torch.eye(roi1)*self.eps[h//roi1-1]
            ans_adj[:, :, h:h+roi1, h-roi1:h] = torch.eye(roi1) + torch.eye(roi1)*self.eps[h//roi1-1]

        
        return ans_adj



    def _collate_adjacency_nosparsity(self, a):
        '''
        a  ( b, T, roi,  roi)
        '''
        
        i_list = []
        v_list = []

        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                #thresholded_a = (_a > self.percentile(_a, 100-sparsity))
                thresholded_a = (_a != 0)
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)
        return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3])).coalesce()



class spGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                 nn.BatchNorm1d(hidden_dim), 
                                 nn.ELU(), 
                                 nn.Linear(hidden_dim, output_dim), 
                                 nn.BatchNorm1d(output_dim), 
                                 nn.ELU())
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0

        

    
    def forward(self, x, a):
        '''
        a (b*T*roi*H, b*T*roi*H)
        x (b*T*roi*H, F)
        '''

        #a = einops.rearrange(a, 'b roi1 T H roi2 -> b T (roi1 H) roi2')
        '''a = einops.rearrange(a, 'b roi1 T H roi2 -> b (T H) roi1 roi2')
        a = self._collate_adjacency_nosparsity(a)
        x = einops.rearrange(x, 'b roi T H F -> (b T roi H) F')'''

        v_aggregate = torch.sparse.mm(a, x)
        v_aggregate += self.epsilon * x # assumes that the adjacency matrix includes self-loop
        v_combine = self.mlp(v_aggregate)
        return v_combine


    

    def _collate_adjacency(self, a, sparsity):
        i_list = []
        v_list = []

        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > self.percentile(_a, 100-sparsity))
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)
        return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))




class FTDGNN(nn.Module):
    def __init__(self, config, transformer_input_dim=24, heads=2, embedding_dim=24, inputShape=(90,8,3,32), layers=4, drop_ratio=0.3, encoder_layers=1) -> None:
        super().__init__()
        self.inputShape = inputShape
        self.layers = layers
        self.config = config
        self.encoder_layers = encoder_layers
        self.rois, self.T, self.H, self.inputdim = self.inputShape
        self.drop_ratio = drop_ratio

        #Transformer Layers
        transformerlayer = TransformerEncoderLayer(d_model=transformer_input_dim,
                                                    nhead=heads,
                                                    batch_first=True,
                                                    dim_feedforward=embedding_dim,
                                                    dropout=drop_ratio)
        
        self.transformers = TransformerEncoder(transformerlayer, num_layers=layers)

        
        self.F = transformer_input_dim

        encoder_dims = [self.inputdim, self.inputdim]

        #Input Encoder
        self.init_input_encoder(encoder_dims, encoder_layers)
        
        #Batch Normalize
        self.encoder_bn = nn.BatchNorm1d(transformer_input_dim)

        #CLS
        if config['cls'] == 1:
            self.cls = torch.nn.Parameter(torch.zeros(1, 1, transformer_input_dim))

        #classification
        self.classification = nn.Sequential(nn.Linear(self.F, self.F),
                                            nn.ReLU(),
                                            nn.Dropout(drop_ratio),
                                            nn.Linear(self.F, 2))
        
        self.dps = nn.ModuleList([nn.Dropout(drop_ratio) for _ in range(encoder_layers)])
        
        
    
    def init_input_encoder(self, encoder_dims, encoder_layers):
        if self.config['input_encoder'] == 'MLP':
            self.encoder = nn.Sequential(nn.Linear(self.inputdim, encoder_dims[0]),
                                     nn.BatchNorm1d(encoder_dims[0]),
                                     nn.ReLU(),
                                     nn.Linear(encoder_dims[0], encoder_dims[1]),
                                     nn.BatchNorm1d(encoder_dims[1]),
                                     nn.ReLU(),
                                     nn.Linear(encoder_dims[1], self.F))
            
        elif self.config['input_encoder'] == 'AE':
            self.encoder = nn.Sequential(nn.Linear(self.inputdim, encoder_dims[0]),
                                     nn.BatchNorm1d(encoder_dims[0]),
                                     nn.ReLU(),
                                     nn.Linear(encoder_dims[0], encoder_dims[1]),
                                     nn.BatchNorm1d(encoder_dims[1]),
                                     nn.ReLU(),
                                     nn.Linear(encoder_dims[1], self.F))
            #load pretrain input encoder
            self.__load_pretrain_encoder()
        elif self.config['input_encoder'] == 'GRN':
            self.encoder = nn.ModuleList([GRN_encoder(input_dim=self.inputdim, 
                                                      output_dim=self.inputdim, 
                                                      network='GNN', 
                                                      rois=self.rois, 
                                                      dropout=self.drop_ratio) for _ in range(encoder_layers-1)]+[GRN_encoder(input_dim=self.inputdim, output_dim=self.F, network='GNN', rois=self.rois, dropout=self.drop_ratio)])
        elif self.config['input_encoder'] == 'GFRN':
            self.encoder = nn.ModuleList([GFRN_encoder(input_dim=self.inputdim, output_dim=self.F, network='GNN', H=self.H)])
        elif self.config['input_encoder'] in ['GNN', 'GCN', 'GFN', 'GAT']:  
            self.encoder = GNN_encoder(input_dim=self.inputdim, 
                                       output_dim=self.F, 
                                       layers=encoder_layers, 
                                       network=self.config['input_encoder'], 
                                       H=self.H, frq=(self.config['input_encoder'] == 'GFN'))
        elif self.config['input_encoder'] == 'Mixer':
            self.encoder = Mixer_encoder(input_dim=self.inputdim,
                                         output_dim=self.F,
                                         rois=self.rois)

        elif self.config['input_encoder'] == 'Conv1d':
            self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding='same'),
                                         nn.ReLU(),
                                         nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding='same'))
        else: raise


    def get_paramters_groups(self, lr):
        return [{"params": self.transformers.parameters(), 'lr':lr},
                {"params": self.classification.parameters(), 'lr':lr},
                {"params": self.cls, 'lr':lr * 0.1},
                {"params": self.SE, 'lr':lr},
                {"params": self.encoder_bn, 'lr':lr}]
        


    def __load_pretrain_encoder(self):
        encoder_param = torch.load('result/pretrain/model/pretrain_AR_dim24_e1000.pth', map_location=f'cuda:{torch.cuda.current_device()}')
        model_state = self.state_dict()
        
        pretrain_model_state = {k:v for k,v in encoder_param['model'].items() if k in model_state}
        model_state.update(pretrain_model_state)   
        self.load_state_dict(model_state)

        #freeze
        for param in self.encoder.parameters():
            param.requires_grad = False



    def TemporalEncoder(self):
        '''
        Shape of TE: (T, F)
        '''
        #rois, T, H, F = self.dataShape
        BE = self.BasedEncoder(self.T)
        return BE#self.TE(BE)

    def FreqentEncoder(self):
        '''
        Shape of TE: (H, F)
        '''
        #rois, T, H, F = self.dataShape
        BE = self.BasedEncoder(self.H)
        return self.FE(BE)

    def StructuralEncoder(self, U): 
        '''
        U (bs, rois, T, rois)
          or (bs, rois, T, H, rois)
        retutn (bs, rois, T, H, F)
        '''
        U_shape = U.shape
        if len(U_shape) == 6:
            U = U.view(-1, U_shape[2], U_shape[3], U_shape[4], U_shape[5])
        #bs, rois, T, _ = U.shape
        #U = U.view(bs, rois, T, 1, rois)
        return self.SE(U)

    def BasedEncoder(self, N):
        '''
        Shape of BE: (N, F)
        '''
        #rois, T, H, F = self.dataShape
        alpha = int(self.F ** 0.5)
        omega = torch.tensor([alpha**(-(i-1)/alpha) for i in range(32)], device='cuda')
        BE = torch.stack([n*omega for n in range(N)], dim=0)
        BE = torch.cos(BE)
        return BE
    
    def extract(self, x):
        '''
        Shape of x: (bs, 1+rois*T*H, F)
        '''
        if self.config['cls'] == 1:
            return x[:,0,:]
        else:
            return torch.mean(x, dim=1)
        
    #mask
    def get_mask(self):
        mask = torch.zeros(self.rois*self.T*self.H, self.rois*self.T*self.H).cuda()
        one = torch.ones(self.T*self.H, self.T*self.H)  * float('-inf')
        for i in range(0,self.T*self.H, self.H): one[i:i+self.H,i:i+self.H] = torch.zeros(self.H, self.H)
        
        for i in range(0, self.rois*self.T*self.H, self.T*self.H):
            for j in range(0, self.rois*self.T*self.H, self.T*self.H):
                mask[i:i+self.T*self.H, j:j+self.T*self.H] = one
            mask[i:i+self.T*self.H, i:i+self.T*self.H].fill_(0)
        
        if self.config['cls']:
            return_mask = torch.zeros(1+self.rois*self.T*self.H, 1+self.rois*self.T*self.H).cuda()
            return_mask[1:,1:] = mask
            return return_mask
        else:
            return mask


    def forward(self, data, need_atten=False):
        '''
        U (bs, rois, T, rois)
        x (bs, rois, T, H, F)
        '''

        # if self.config['node_feature'] == 'LaplacianEigenvectors':
        #     x = data['U'] #[:,:,:,1:2,:]#测试一个频率段
        # elif self.config['node_feature'] == 'Pearson':
        #     x = data['Pearson'] #[:,:,:,1:2,:]
        # elif self.config['node_feature'] == 'TimeSeries':
        #     x = data['x'] #[:,:,:,1:2,:]
        # else:
        #     raise
        x = data['x'][:,:,:self.T,:,:]

        x_shape = x.shape
        if len(x_shape) == 6:
            x = x.view(-1, x_shape[2], x_shape[3], x_shape[4], x_shape[5])
        #bs, rois, T, H, F = x.shape#(bs, 90, 8, 3, 32)
        bs = x.shape[0]
        

        #input encoder
        if self.config['input_encoder'] == 'Conv1d':
            x = einops.rearrange(x, 'bs rois T H F -> (bs rois T H) 1 F')
            x = self.encoder(x)
            x = einops.rearrange(x, '(bs rois T H) 1 F -> bs rois T H F', bs=bs, rois=self.rois, T=self.T, H=self.H)
        elif self.config['input_encoder'] == 'MLP':
            x = einops.rearrange(x, 'bs rois T H F -> (bs rois T H) F')
            x = self.encoder(x)
            x = self.encoder_bn(x)
            x = einops.rearrange(x, '(bs rois T H) F -> bs rois T H F', bs=bs, rois=self.rois, T=self.T, H=self.H)
        elif self.config['input_encoder'] in ['GNN', 'GFN', 'GCN', 'GAT', 'GRN', 'GFRN']:
            #b, roi, T, H, F = x_shape

            x = self.encoder[0](x, data['A'][:,:,:self.T,:,:])
            x = self.dps[0](x)
            for i in range(1, self.encoder_layers):
                x = self.encoder[i](x, data['A'][:,:,:self.T,:,:])
                x = self.dps[i](x)
            #(b, roi, T, H, F)    <-  ( (b, T, roi),  F)
            #x = einops.rearrange(x, '(b T roi H) F -> b roi T H F', b=bs, roi=x_shape[1], T=x_shape[2], H=x_shape[3])
        elif self.config['input_encoder'] == 'Mixer':
            x = self.encoder(x)
        else:
            raise
        
        x = einops.rearrange(x, 'bs rois T H F -> bs (rois T H) F')
        #x = x.flatten(1,3)#(bs, rois*T*H, F)

        #add CLS(1, 1, 32) -> (bs, 1, 32)
        if self.config['cls'] == 1:
            cls = repeat(self.cls, '() n d -> b n d', b = bs)
            x = torch.cat([cls, x], dim=1)

        #cls = self.cls.expand(bs, -1, -1)
        #x = torch.cat([cls, x], dim=1)
        if self.config['mask']:
            mask = self.get_mask()
        else:
            mask = None

        if self.layers:
            if need_atten:
                state = x
                x, atten_maps = self.transformers(x, mask=mask, need_atten=need_atten)#(bs, 1+rois*T*H, F)
            else:
                x = self.transformers(x, mask=mask)
        
        x = self.extract(x)#(bs, F)
        x = self.classification(x)#(bs, 2)

        if need_atten:
            return x, atten_maps, state
        else:
            return x


#Warm up
class WarmupLR:
    def __init__(self, optimizer, num_warm) -> None:
        self.optimizer = optimizer
        self.num_warm = num_warm
        self.lr = [group['lr'] for group in self.optimizer.param_groups]
        self.num_step = 0
 
    def __compute(self, lr) -> float:
        return lr * min(self.num_step ** (-0.5), self.num_step * self.num_warm ** (-1.5))
 
    def step(self) -> None:
        self.num_step += 1
        lr = [self.__compute(lr) for lr in self.lr]
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr[i]


#train with multi-GPU
def model_train_multiGPU(train_loader, model, opt, device, rank):
    model.train()
    #mean_loss = torch.zeros(1).to(device)
    mean_loss = 0
    opt.zero_grad()

    if rank == 0:
        train_loader = tqdm(train_loader)

    for step, sample in enumerate(train_loader):
        out = model(sample['x'].to(device), sample['U'].to(device))
        loss = nn.CrossEntropyLoss()(out, sample['label'].to(device).view(-1))
        loss.backward()
        loss = reduce_value(loss, average=True)
        #mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
        mean_loss += loss.item()
        opt.step()
        opt.zero_grad()

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss / len(train_loader)

#train
def model_train(train_loader, model, opt, scheduler=None):
    model.train()
    total_loss = 0
    for sample in tqdm(train_loader):
   
        sample = {k:sample[k].cuda() for k in sample.keys()}
        #sample = map(lambda x:x.cuda, sample)
        opt.zero_grad()
        #out = model(sample['x'].cuda(), sample['U'].cuda(), sample['A'].cuda())
        out = model(sample)
        loss = nn.CrossEntropyLoss()(out, sample['label'].view(-1))
        loss.backward()
        total_loss = total_loss + loss.item()
        opt.step()
        if scheduler is not None:
            scheduler.step()
    return total_loss / len(train_loader)

#validate with multi-GPU
def model_val_multiGPU(val_loader, model, device, rank):
    model.eval()
    mean_loss = torch.zeros(1).to(device)
    
    if rank == 0:
        print('val')
        val_loader = tqdm(val_loader)

    for step, sample in enumerate(val_loader):
        out = model(sample['x'].to(device), sample['U'].to(device), sample['A'].cuda())
        loss = nn.CrossEntropyLoss()(out, sample['label'].to(device).view(-1))
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
    
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    
    return mean_loss.item()

#validate
def model_val(val_loader, model):
    model.eval()
    total_loss = 0
    for sample in val_loader:
        #data = data.cuda()
        sample = {k:sample[k].cuda() for k in sample.keys()}
        out = model(sample)
        loss = nn.CrossEntropyLoss()(out, sample['label'].view(-1))
        total_loss = total_loss + loss.item()
    return total_loss / len(val_loader)


#test
def model_test(test_loader, model, need_atten=False):
    model.eval()
    t = 1e-8
    TP,TN,FP,FN = 0, 0, 0, 0
    gt = []
    pr = []
    atten_maps = []
    state_maps = []
    labels = []
    for sample in tqdm(test_loader):
        sample = {k:sample[k].cuda() for k in sample.keys()}
        if need_atten:
            out, atten_map, state_map = model(sample, need_atten=need_atten)
            atten_maps.append(atten_map.detach().cpu())
            state_maps.append(state_map.detach().cpu())
            labels.append(sample['label'])
        else:
            out = model(sample, need_atten=need_atten)
        
        out = torch.nn.functional.softmax(out, dim=1)
        y_pre = torch.argmax(out, dim=1)

        pr.append(out[:,1].detach().cpu().numpy())
        gt.append(sample['label'].detach().cpu().view(-1).numpy())

        for i in range(len(y_pre)):
            if y_pre[i]==1 and sample['label'].view(-1)[i]==1:
                TP = TP + 1
            elif y_pre[i]==0 and sample['label'].view(-1)[i]==0:
                TN = TN + 1
            elif y_pre[i]==0 and sample['label'].view(-1)[i]==1:
                FN = FN + 1
            elif y_pre[i]==1 and sample['label'].view(-1)[i]==0:
                FP = FP + 1
    acc = (TP+TN)/(TP+TN+FP+FN+t)
    spec = TN/(FP + TN+t)
    sen = TP/(TP + FN+t)
    if (TP+FP)==0:
        pre = 0
    else:
        pre = TP/(TP+FP+t)   
    if (pre+sen)==0:
        F1 = 0
    else:
        F1 = 2*pre*sen / (pre+sen+t)
    gt = np.concatenate(gt, axis=0)
    pr = np.concatenate(pr, axis=0)
    AUC = roc_auc_score(gt, pr)
    if need_atten:
        return {'acc':acc, 'spec':spec, 'sen':sen, 'pre':pre, 'AUC':AUC, 'F1':F1}, {'maps': torch.cat(atten_maps, dim=0), 'labels': torch.cat(labels, dim=0)}, {'maps': torch.cat(state_maps, dim=0), 'labels': torch.cat(labels, dim=0)}
    else:
        return {'acc':acc, 'spec':spec, 'sen':sen, 'pre':pre, 'AUC':AUC, 'F1':F1} 