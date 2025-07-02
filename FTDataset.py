import torch
from scipy.io import loadmat
import os
import pandas as pd
import numpy as np
from tqdm import *
import einops
from sklearn.model_selection import KFold, StratifiedKFold


def getListFromSplit(root,
                     partition='train', 
                     splitpath='split_ADNI_AD_NC_MOCO_ratio',
                     split=1, 
                     sources='hospital',
                     type='AD_NC'):
    datapath = os.path.join(root, 'raw')
    df = pd.read_csv(os.path.join('datasplit',splitpath, str(split), partition + '.csv'))
    
    if sources == 'hospital':
        sub_id, datatype = np.array(df['ID']), np.array(df['label'])
    elif sources == 'ADNI':
        sub_id, datatype = np.array(df['SUB_ID']), np.array(df['label'])

    if sources == 'hospital':
        if type == 'AD_NC':
            AD_dirs = [os.path.join(datapath, f'ROISignals_Sub_{sub}.mat') for sub in sub_id[datatype == "AD"] if
                    os.path.exists(os.path.join(datapath, f'ROISignals_Sub_{sub}.mat'))]
            NC_dirs = [os.path.join(datapath, f'ROISignals_Sub_{sub}.mat') for sub in sub_id[datatype == "NC"] if
                    os.path.exists(os.path.join(datapath, f'ROISignals_Sub_{sub}.mat'))]
            return AD_dirs+NC_dirs, [1 for i in range(len(AD_dirs))]+[0 for i in range(len(NC_dirs))]
        
        elif type == 'AD_NC_4D':
            AD_dirs = [os.path.join(datapath, f'Sub_{sub:03d}','Filtered_4DVolume.npy') for sub in sub_id[datatype == "AD"] if
                    os.path.exists(os.path.join(datapath, f'Sub_{sub:03d}'))]
            NC_dirs = [os.path.join(datapath, f'Sub_{sub:03d}','Filtered_4DVolume.npy') for sub in sub_id[datatype == "NC"] if
                    os.path.exists(os.path.join(datapath, f'Sub_{sub:03d}'))]
            return AD_dirs+NC_dirs, [1 for i in range(len(AD_dirs))]+[0 for i in range(len(NC_dirs))]
        
        elif type == 'AD_MCI':
            AD_dirs = [os.path.join(datapath, f'ROISignals_Sub_{sub}.mat') for sub in sub_id[datatype == "AD"] if
                    os.path.exists(os.path.join(datapath, f'ROISignals_Sub_{sub}.mat'))]
            MCI_dirs = [os.path.join(datapath, f'ROISignals_Sub_{sub}.mat') for sub in sub_id[datatype == "MCI"] if
                    os.path.exists(os.path.join(datapath, f'ROISignals_Sub_{sub}.mat'))]
            return AD_dirs+MCI_dirs, [1 for i in range(len(AD_dirs))]+[0 for i in range(len(MCI_dirs))]
        elif type == 'NC_MCI':
            NC_dirs = [os.path.join(datapath, f'ROISignals_Sub_{sub}.mat') for sub in sub_id[datatype == "NC"] if
                    os.path.exists(os.path.join(datapath, f'ROISignals_Sub_{sub}.mat'))]
            MCI_dirs = [os.path.join(datapath, f'ROISignals_Sub_{sub}.mat') for sub in sub_id[datatype == "MCI"] if
                    os.path.exists(os.path.join(datapath, f'ROISignals_Sub_{sub}.mat'))]
            return NC_dirs+MCI_dirs, [1 for i in range(len(NC_dirs))]+[0 for i in range(len(MCI_dirs))]


    elif sources == 'ADNI':
        if type == 'AD_NC':
            AD_dirs = [os.path.join(datapath, f'sub_{sub}.pt') for sub in sub_id[datatype == "AD"] if
                    os.path.exists(os.path.join(datapath, f'sub_{sub}.pt'))]
            NC_dirs = [os.path.join(datapath, f'sub_{sub}.pt') for sub in sub_id[datatype == "NC"] if
                    os.path.exists(os.path.join(datapath, f'sub_{sub}.pt'))]
            return AD_dirs+NC_dirs, [1 for i in range(len(AD_dirs))]+[0 for i in range(len(NC_dirs))]
        
        elif type == 'AD_MCI':
            AD_dirs = [os.path.join(datapath, f'sub_{sub}.pt') for sub in sub_id[datatype == "AD"] if
                    os.path.exists(os.path.join(datapath, f'sub_{sub}.pt'))]
            MCI_dirs = [os.path.join(datapath, f'sub_{sub}.pt') for sub in sub_id[datatype == "MCI"] if
                    os.path.exists(os.path.join(datapath, f'sub_{sub}.pt'))]
            return AD_dirs+MCI_dirs, [1 for i in range(len(AD_dirs))]+[0 for i in range(len(MCI_dirs))]
        
        elif type == 'NC_MCI':
            NC_dirs = [os.path.join(datapath, f'sub_{sub}.pt') for sub in sub_id[datatype == "NC"] if
                    os.path.exists(os.path.join(datapath, f'sub_{sub}.pt'))]
            MCI_dirs = [os.path.join(datapath, f'sub_{sub}.pt') for sub in sub_id[datatype == "MCI"] if
                    os.path.exists(os.path.join(datapath, f'sub_{sub}.pt'))]
            return NC_dirs+MCI_dirs, [1 for i in range(len(NC_dirs))]+[0 for i in range(len(MCI_dirs))]
        

def maskA(A, percent=30):
        roi = A.shape[0]
        A = A.to(torch.float32)
        k = int(roi * ( percent / 100 ))
        v, i = torch.topk(A, k, dim=0, sorted=False)
        mA = torch.zeros(roi, roi)
        for c in range(roi):
            #mA[i[:,c],c] = A[i[:,c],c]
            mA[c, i[:,c]] = A[c, i[:,c]]
            mA[i[:,c], c] = A[c, i[:,c]]      
        return mA


def get_adj(x):
    A = torch.corrcoef(x)
    A[torch.isnan(A)] = 0
    return maskA(A, 30)

class FTDataset(torch.utils.data.Dataset):
    def __init__(self, root, partition='train', sources='hospital', type='AD_NC', splitpath='split_hospital_AD_NC', split=1, roi_num=90, signal_len=200, reprocess=0, node_feature='one-hot'):
        self.data_sources = sources
        self.node_feature = node_feature

        if sources == 'hospital':
            self.__init__noaug(root=root, partition=partition, type=type, splitpath=splitpath, split=split, roi_num=roi_num, signal_len=signal_len, reprocess=reprocess)
        elif sources == 'ADNI':
            self.__init__ADNI(root=root, partition=partition, type=type, splitpath=splitpath, split=split, roi_num=roi_num, signal_len=signal_len, reprocess=reprocess)
        elif sources == 'ABIDE':
            self.__init__ABIDE(root=root, partition=partition, type=type, splitpath=splitpath, split=split, roi_num=roi_num, signal_len=signal_len, reprocess=reprocess)
        else:
            raise


    def __init__ABIDE(self, root, partition='train', type='ASD_HC', splitpath='split_ABIDE', split=1, roi_num=90, signal_len=146, reprocess=0):
        super().__init__()
        self.signal_len = signal_len
        self.root = root
        self.roi_num = roi_num
        strike = 16

        self.all_raw_files = os.listdir(os.path.join(root, 'raw'))
        self.all_processed_files = os.listdir(os.path.join(root, 'processed'))

        if reprocess:
            self.all_processed_files = []
            self.all_labels = []
            meta = pd.read_csv(os.path.join(root, 'Phenotypic_V1_0b_preprocessed1.csv'))
            for sub in tqdm(self.all_raw_files):
                data = np.loadtxt(os.path.join(root, 'raw', sub))#(xx, 116)
                data = torch.from_numpy(data.T)[:roi_num,:] #(roi, xx)
                

                #normalization
                data = self.roi_normalization(data)
                

                sid = int(sub.split('_')[-3])
                center = sub.split('_')[0]
                label = meta[meta['SUB_ID']==sid]['DX_GROUP'].values
                if label == 2: label = np.array([0])

                if data.shape[1] > signal_len:

                    datas = [data[:,:signal_len], data[:,-signal_len:]]
                    for i,d in enumerate(datas):
                        #origin_A = get_adj(d)
                        origin_A=None

                        featureMap = self.meanfilter3seg(d)
                        featureMap = self.temporal_split_window(featureMap, 64, strike)
                        tfA_set, tfU_set, A = self.getTemporalFrequenceStructInfo(featureMap)
                        one_hot = torch.eye(roi_num)
                        one_hot = einops.repeat(one_hot, 'roi1 roi2 -> roi1 T H roi2', T=tfU_set.shape[1], H=tfU_set.shape[2])
                        freq_one_hot = torch.eye(roi_num * tfU_set.shape[2]) #(H*roi, H*roi)
                        freq_one_hot = einops.rearrange(freq_one_hot, '(roi H) Hroi -> roi H Hroi', H=tfU_set.shape[2]) 
                        freq_one_hot = einops.repeat(freq_one_hot, 'roi H Hroi -> roi T H Hroi', T=tfU_set.shape[1])
                        torch.save({'data': featureMap, 'tA_set':tfA_set, 'tU_set':tfU_set, 'A_fullcon':A, 'one-hot':one_hot, 'f-one-hot':freq_one_hot, 'origin_A':origin_A, 'origin_signal':d, 'label': label}, 
                                os.path.join(root, 'processed', f'{center}_{sid}_{i}.pt'))
                        self.all_processed_files.append(f'{center}_{sid}_{i}.pt')
                        self.all_labels.append(label)

                else:
                    p = ''
                    if data.shape[1] < signal_len:#padding
                        pad = torch.zeros(roi_num, signal_len-data.shape[1])
                        data = torch.cat([data, pad], dim=1)
                        p += '_p'

                    origin_A = None

                    featureMap = self.meanfilter3seg(data)
                    featureMap = self.temporal_split_window(featureMap, 64, strike)
                    tfA_set, tfU_set, A = self.getTemporalFrequenceStructInfo(featureMap)
                    one_hot = torch.eye(roi_num)
                    one_hot = einops.repeat(one_hot, 'roi1 roi2 -> roi1 T H roi2', T=tfU_set.shape[1], H=tfU_set.shape[2])
                    freq_one_hot = torch.eye(roi_num * tfU_set.shape[2]) #(H*roi, H*roi)
                    freq_one_hot = einops.rearrange(freq_one_hot, '(roi H) Hroi -> roi H Hroi', H=tfU_set.shape[2]) 
                    freq_one_hot = einops.repeat(freq_one_hot, 'roi H Hroi -> roi T H Hroi', T=tfU_set.shape[1])
                    torch.save({'data': featureMap, 'tA_set':tfA_set, 'tU_set':tfU_set, 'A_fullcon':A, 'one-hot':one_hot, 'f-one-hot':freq_one_hot, 'origin_A':origin_A, 'origin_signal':data, 'label': label}, 
                            os.path.join(root, 'processed', f'{center}_{sid}{p}.pt'))
                    self.all_processed_files.append(f'{center}_{sid}{p}.pt')
                    self.all_labels.append(label)
            torch.save(self.all_labels, os.path.join(root, 'processed_labels.pt'))
        else:
            self.all_labels = torch.load(os.path.join(root, 'processed_labels.pt'), weights_only=False)

        #k-folds
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(skf.split(self.all_processed_files, self.all_labels)):
            if fold == split - 1:
                if partition == 'train':
                    self.filepath_list = np.array(self.all_processed_files)[train_index][len(test_index)//2:]
                elif partition == 'test':
                    self.filepath_list = np.array(self.all_processed_files)[test_index]
                elif partition == 'val':
                    self.filepath_list = np.array(self.all_processed_files)[train_index][:len(test_index)//2]
                else:
                    raise
                break
    

        

    def __init__ADNI(self, root, partition='train', type='AD_NC', splitpath='split_hospital_AD_NC', split=1, roi_num=90, signal_len=140, reprocess=0):
        super().__init__()

        self.signal_len = signal_len
        self.root = root
        self.roi_num = roi_num

        self.filepath_list, self.type_list = getListFromSplit(root=root, 
                                                              partition=partition, 
                                                              split=split, 
                                                              sources='ADNI',
                                                              splitpath=splitpath,
                                                              type=type)

        if not os.path.exists(os.path.join(root, 'processed')):
            os.mkdir(os.path.join(root, 'processed'))

        if reprocess == 1:
            for n, fpath in tqdm(enumerate(self.filepath_list)):
                f = fpath.split('/')[-1]
                p = os.path.join(root, 'processed', f)

                data = torch.load(fpath)

                data = self.roi_normalization(data['data'][:,:140])

                #origin_A = get_adj(data)
                origin_A = None

                featureMap = self.meanfilter3seg(data)
                
                featureMap = self.temporal_split_window(featureMap, 64, 16)

                tfA_set, tfU_set, A = self.getTemporalFrequenceStructInfo(featureMap)
                #assert torch.isnan(tfA_set).any() == False, f'样本{f}有nan'
                if torch.isnan(tfA_set).any() == True:
                    print(f'样本{f}有nan')

                one_hot = torch.eye(roi_num)
                one_hot = einops.repeat(one_hot, 'roi1 roi2 -> roi1 T H roi2', T=tfU_set.shape[1], H=tfU_set.shape[2])  
                
                torch.save({'data': featureMap, 'tA_set':tfA_set, 'tU_set':tfU_set, 'A_fullcon':A, 'one-hot':one_hot, 'origin_signal': data,'origin_A':origin_A, 'label': self.type_list[n]}, p)

        print(f'{partition} set finish loadding')


    def __init__noaug(self, root, partition='train', type='AD_NC', splitpath='split_hospital_AD_NC', split=1, roi_num=90, signal_len=200, reprocess=0):
        super().__init__()

        #37 38 39 40 41 42
        self.signal_len = signal_len
        self.root = root
        self.roi_num = roi_num

        self.filepath_list, self.type_list = getListFromSplit(root=root, 
                                                              partition=partition, 
                                                              split=split, 
                                                              splitpath=splitpath,
                                                              type=type)
        #Create processed dir
        if not os.path.exists(os.path.join(root, 'processed')):
            os.mkdir(os.path.join(root, 'processed'))
        for n, fpath in tqdm(enumerate(self.filepath_list)):
            f = fpath.split('/')[-1]
            id = f.split('_')[-1]
            id = id.split('.')[0]
            p = os.path.join(root,'processed',f'Sub_{id}.pt')     
            if not os.path.exists(p) or reprocess==1:
                origin_signal = loadmat(fpath)['ROISignals']#现在只取某一脑区
                origin_signal = torch.from_numpy(origin_signal.T)[:roi_num,:]#(roi, 200)
                #normalization
                origin_signal = self.roi_normalization(origin_signal) #(roi, 200)
                origin_A = get_adj(origin_signal)

                featureMap = self.meanfilter3seg(origin_signal)
                featureMap = self.temporal_split_window(featureMap, 64, 16)#(roi,H,F)->(roi,T,H,F)

                tfA_set, tfU_set, A = self.getTemporalFrequenceStructInfo(featureMap)

                one_hot = torch.eye(roi_num)
                one_hot = einops.repeat(one_hot, 'roi1 roi2 -> roi1 T H roi2', T=tfU_set.shape[1], H=tfU_set.shape[2])

                freq_one_hot = torch.eye(roi_num * tfU_set.shape[2]) #(H*roi, H*roi)
                freq_one_hot = einops.rearrange(freq_one_hot, '(roi H) Hroi -> roi H Hroi', H=tfU_set.shape[2]) 
                freq_one_hot = einops.repeat(freq_one_hot, 'roi H Hroi -> roi T H Hroi', T=tfU_set.shape[1])


                torch.save({'data': featureMap, 'tA_set':tfA_set, 'tU_set':tfU_set, 'A_fullcon':A, 'one-hot':one_hot, 'f-one-hot':freq_one_hot, 'origin_signal':origin_signal,'origin_A':origin_A ,'label': self.type_list[n]}, 
                           os.path.join(root, 'processed', f'Sub_{id}.pt'))
                
        print(f'{partition} set finish loadding')



    def roi_normalization(self, x):
        return (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8) #normalize


    def global_normalization(self, origin_signal):
        '''
        origin_signal: (roi, 200)
        '''
        m = torch.mean(origin_signal)
        std = torch.std(origin_signal)

        return (origin_signal - m) / (std + 1e-8)



    def __getitem__(self, idx):
        if self.data_sources == 'hospital':
            return self.__getitem__noaug(idx)
        elif self.data_sources == 'ADNI':
            return self.__getitem__ADNI(idx)
        elif self.data_sources == 'ABIDE':
            return self.__getitem__ABIDE(idx)
        else:
            raise

        
    def __getitem__ABIDE(self, idx):
        fpath = os.path.join(self.root, 'processed', self.filepath_list[idx])
        data = torch.load(fpath, weights_only=False)
        label = data['label']
        if self.node_feature == 'one-hot':
            #return {'x': data['one-hot'][:,:,0:1,:], 'A': data['tA_set'][:,:,0:1,:], 'label':label}
            return {'x': data['one-hot'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'f-one-hot':
            return {'x': data['f-one-hot'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'Pearson':
            return {'x': data['A_fullcon'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'LaplacianEigenvectors':
            return {'x': data['tU_set'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'Adj':
            return {'x': data['tA_set'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'TimeSeries':
            return {'x': data['data'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'baseline':
            return {'x': data['origin_signal'], 'A':data['origin_A'], 'label':label}
        else:
            raise
    
        
    def __getitem__ADNI(self, idx):
        fpath = self.filepath_list[idx]
        f = fpath.split('/')[-1]
        data = torch.load(os.path.join(self.root, 'processed', f))
        label = data['label']

        if self.node_feature == 'one-hot':
            return {'x': data['one-hot'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'f-one-hot':
            return {'x': data['f-one-hot'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'Pearson':
            return {'x': data['Pearson'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'LaplacianEigenvectors':
            return {'x': data['tU_set'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'Adj':
            return {'x': data['tA_set'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'baseline':
            return {'x': data['origin_signal'], 'A':data['origin_A'], 'label':label}
        else:
            raise


    def __getitem__noaug(self, idx):
        fpath = self.filepath_list[idx]
        f = fpath.split('/')[-1]
        id = f.split('_')[-1]
        id = id.split('.')[0]
        data = torch.load(os.path.join(self.root, 'processed', f'Sub_{id}.pt'))
        label = data['label']
        '''if label == 'AD':
            label = 1
        else:
            label = 0'''
        #return data['data'], label
        if self.node_feature == 'one-hot':
            return {'x': data['one-hot'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'f-one-hot':
            return {'x': data['f-one-hot'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'Pearson':
            return {'x': data['Pearson'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'LaplacianEigenvectors':
            return {'x': data['tU_set'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'Adj':
            return {'x': data['tA_set'], 'A': data['tA_set'], 'label':label}
        elif self.node_feature == 'baseline':
            return {'x': data['origin_signal'], 'A':data['origin_A'], 'label':label}
        else:
            raise



    def __len__(self):
        return len(self.filepath_list)
    

    def getTemporalFrequenceStructInfo(self, featureMap_all):
        '''
        Shape of featureMap_all: (roi, T, H, F) or (b, roi, T, H, F))
        '''
        if len(featureMap_all.shape) == 4:
            rois, T , H, F = featureMap_all.shape
            A_set = torch.zeros(rois, T, H ,rois)
            A_set_origin = torch.zeros(rois, T, H ,rois)
            U_set = torch.zeros(rois, T, H ,rois)
            for t in range(T):
                for h in range(H):
                    x = featureMap_all[:,t,h,:]
                    A = torch.corrcoef(x)
                    A[torch.isnan(A)] = 0
                    A_set_origin[:, t, h, :] = A
                    #top 30%
                    A_set[:,t,h,:] = self.maskA(A)
                    D = self.degreeA(A_set[:,t,h,:].clone())
                    #D = self.degreeA(A)
                    D = D**(-1/2)
                    D = torch.diag(D)
                    deta = torch.eye(rois) - D @ A_set[:,t,h,:] @ D
                    deta[torch.isnan(deta)] = 0
                    #deta = torch.eye(rois) - D @ A @ D
                    _, U = torch.linalg.eig(deta)
                    U_set[:,t,h,:] = U.real
        else:
            B, rois, T , H, F = featureMap_all.shape
            A_set =  torch.zeros(B, rois, T, H ,rois)
            A_set_origin = torch.zeros(B, rois, T, H, rois)
            U_set =  torch.zeros(B, rois, T, H ,rois)
            for b in range(B):
                for t in range(T):
                    for h in range(H):
                        x = featureMap_all[b, :, t, h, :]
                        A = torch.corrcoef(x)
                        A_set_origin[b, :, t, h, :] = A
                        #top 30%
                        A_set[b, :, t, h, :] = self.maskA(A)
                        #if torch.isnan(A).any():
                        #    print(self.filepath_list[idx])
                        
                        D = self.degreeA(A)
                        D = D**(-1/2)
                        D = torch.diag(D)
                        deta = torch.eye(rois) - D @ A @ D
                        _, U = torch.linalg.eig(deta)
                        U_set[b, :, t, h, :] = U.real
        return A_set, U_set, A_set_origin


    def stft(self, signals):
        '''
        Shape of sp: (H, Ts)
        '''
        n = 64#128
        sp = torch.stft(input=torch.from_numpy(signals), 
                        window=torch.hann_window(n), 
                        n_fft=n, 
                        hop_length=1,#int(0.25*n),#int(0.25*128), 
                        win_length = n, 
                        center=True, 
                        return_complex=True)
        #return signal.stft(signals, fs=0.5, window='hann',nperseg=16,noverlap=None,nfft=None,detrend=False,return_onesided=True,boundary='zeros',padded=True,axis=-1)
        return sp
    
    #脑区平均滤波成三个频段
    def meanfilter3seg(self, x, start=0.01, seg1=0.027, seg2=0.073, end=0.1):
        '''
        x: (90, 200)
        res: (90, 3, 200)
        '''
        rois, f_dim = x.shape
        fs = 0.5

        spectrum = torch.fft.fft(x)
        freqs = torch.fft.fftfreq(f_dim, d=1/fs)

        filter1 = torch.logical_and(freqs >= start, freqs < seg1).float()
        filter2 = torch.logical_and(freqs >= seg1, freqs < seg2).float()
        filter3 = torch.logical_and(freqs >= seg2, freqs <= end).float()
        
        filtered_spectrum1 = spectrum * filter1
        filtered_spectrum2 = spectrum * filter2
        filtered_spectrum3 = spectrum * filter3

        filtered_signal1 = torch.fft.ifft(filtered_spectrum1).real
        filtered_signal2 = torch.fft.ifft(filtered_spectrum2).real
        filtered_signal3 = torch.fft.ifft(filtered_spectrum3).real

        res = torch.zeros(rois, 3, f_dim)
    
        res[:,0,:] = filtered_signal1
        res[:,1,:] = filtered_signal2
        res[:,2,:] = filtered_signal3
        return res


    #4D滤波成三个频段
    def filter3seg(self, x, start=0.01, seg1=0.027, seg2=0.073, end=0.1):
        '''
        x: (61, 73, 61, 200)
        res: (61, 73, 61, 3, 200)
        '''
        X,Y,Z = 61,73,61
        f_dim = x.shape[-1]

        fs = 0.5
        x = torch.flatten(x, start_dim=0, end_dim=2) #(xxx, 200)
        x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8) #normalize

        spectrum = torch.fft.fft(x)
        freqs = torch.fft.fftfreq(f_dim, d=1/fs)

        filter1 = torch.logical_and(freqs >= start, freqs < seg1).float()
        filter2 = torch.logical_and(freqs >= seg1, freqs < seg2).float()
        filter3 = torch.logical_and(freqs >= seg2, freqs <= end).float()

        filtered_spectrum1 = spectrum * filter1
        filtered_spectrum2 = spectrum * filter2
        filtered_spectrum3 = spectrum * filter3

        filtered_signal1 = torch.fft.ifft(filtered_spectrum1).real
        filtered_signal2 = torch.fft.ifft(filtered_spectrum2).real
        filtered_signal3 = torch.fft.ifft(filtered_spectrum3).real

        res = torch.zeros(X, Y, Z, 3, f_dim)
        res[:,:,:,0,:] = filtered_signal1.view(X,Y,Z,f_dim)
        res[:,:,:,1,:] = filtered_signal2.view(X,Y,Z,f_dim)
        res[:,:,:,2,:] = filtered_signal3.view(X,Y,Z,f_dim)

        return res
    
    def process_4Dstruct(self):
        if not os.path.exists(os.path.join(self.root, 'struct_processed')):
            os.mkdir(os.path.join(self.root, 'struct_processed'))
        
        for n, fpath in tqdm(enumerate(self.filepath_list)):
            f = fpath.split('/')[-2]
            id = f.split('_')[-1]
            #id = id.split('.')[0]
            p = os.path.join(self.root, 'struct_processed', f'Sub_{int(id)}.pt')
            origin_4Dsignal = np.load(fpath, allow_pickle=True)
            origin_4Dsignal = torch.from_numpy(origin_4Dsignal)
            filter_4Dsignal = self.filter3seg(origin_4Dsignal) #(X, Y, Z, 3, 200)
            struct_data = [self.find_voxels(roi, filter_4Dsignal) for roi in range(1, self.roi_num+1)]
            torch.save({'data': struct_data, 'label': self.type_list[n]},  p)
            

    def find_voxels(self, roi, brain_matrix, shuffle=True):
        '''
        brain_matrix: (X, Y, Z, H, F)
        voxels: (N, H, F)
        '''
        X, Y, Z, H, F = brain_matrix.shape
        #mask = self.mask.view(X,Y,Z,1,1)
        flag_mask = (self.mask == roi).to(torch.int)
        idx = torch.nonzero(flag_mask)
        voxels = brain_matrix[idx[:,0], idx[:,1], idx[:,2], :, :]
        if shuffle:
            N = voxels.shape[0]
            shuffle_idx = torch.randperm(N)
            voxels = voxels[shuffle_idx,:,:] 
        return voxels


    def process_4Dfilter(self):
        if not os.path.exists(os.path.join(self.root, '4Dprocessed')):
            os.mkdir(os.path.join(self.root, '4Dprocessed'))

        for n, fpath in tqdm(enumerate(self.filepath_list)):
            f = fpath.split('/')[-2]
            id = f.split('_')[-1]
            #id = id.split('.')[0]

            p = os.path.join(self.root, '4Dprocessed', f'Sub_{int(id)}.pt') 
            if not os.path.exists(p):
                origin_4Dsignal = np.load(fpath, allow_pickle=True)
                origin_4Dsignal = torch.from_numpy(origin_4Dsignal)
                assert ~(torch.isnan(origin_4Dsignal).any()),f'样本{id}含有Nan值'
                filter_4Dsignal = self.filter3seg(origin_4Dsignal) #(X, Y, Z, 3, 200)
                assert (torch.isnan(filter_4Dsignal).any()) == False, f'样本{id}含有Nan值'
                torch.save({'data': filter_4Dsignal, 'label': self.type_list[n]}, 
                            os.path.join(self.root, '4Dprocessed', f'Sub_{int(id)}.pt'))


    def degreeA(self, A):
        D = A.clone()
        D[D!=0] = 1
        D = torch.sum(D, dim=0)
        return D

    
    def maskA(self, A, percent=55):
        k = int(self.roi_num * ( percent / 100 ))
        v, i = torch.topk(A, k, dim=0, sorted=False)
        mA = torch.zeros(self.roi_num, self.roi_num)
        for c in range(self.roi_num):
            #mA[i[:,c],c] = A[i[:,c],c]
            mA[c, i[:,c]] = A[c, i[:,c]].to(torch.float32)
            mA[i[:,c], c] = A[c, i[:,c]].to(torch.float32)    
        return mA

    
    
    def temporal_split_window(self, feature_map, window_len=64, stride=16):
        '''
        feature_map: (roi, H, orign_F)
        Shape of sp_sigal: (roi, T, H, F)
        '''
        sp_signal = [feature_map[:, :, i:i+window_len] for i in range(0, feature_map.shape[-1]-window_len, stride)]
        sp_signal = torch.stack(sp_signal, dim=1)
        return sp_signal



    def zeropad(self, signal):
        s = len(signal)
        if s >= self.signal_len:
            pad_signal = signal
        else:
            pad_signal = np.zeros(self.signal_len)
            pad_signal[0:s] = signal
        return pad_signal


