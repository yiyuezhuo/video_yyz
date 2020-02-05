print("notebook_utils 0.1")

from video_yyz.utils import accuracy
from video_yyz.frozen_utils import get_pipeline, get_model

from IPython.display import display, HTML
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np


class EvalSuit:
    def __init__(self, root, model_name, dataset_config, checkpoint_path, device='cuda'):
        self.root = root
        self.model_name = model_name
        self.dataset_config = dataset_config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        
    def load(self):
        root = self.root
        device = self.device
        
        model_name, dataset_config, checkpoint_path = self.model_name, self.dataset_config, self.checkpoint_path
        config = dataset_config
        
        dataset_test, transform_test, collate_test, data_loader_test = get_pipeline(root, *config)
        model = get_model(model_name)
        model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        
        self.dataset_test = dataset_test
        self.transform_test = transform_test
        self.collate_test = collate_test
        self.data_loader_test = data_loader_test
        self.model = model
        self.checkpoint = checkpoint
        
    def compute(self):
        device = self.device
        model = self.model
        data_loader_test = self.data_loader_test
        
        acc1_list = []
        batch_size_list = []
        output_list = []
        video_idx_list = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader_test):
                video, target = batch['video'], batch['target']

                video = video.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(video)

                acc1, = accuracy(output, target, topk=(1, ))
                batch_size = video.shape[0]

                acc1_list.append(acc1.item())
                batch_size_list.append(batch_size)
                output_list.append(output.cpu())
                video_idx_list.append(batch['video_idx'])
        
        self.acc1_list = acc1_list
        self.batch_size_list = batch_size_list
        self.output_list = output_list
        self.video_idx_list = video_idx_list
    
    def summary(self):
        output_list = self.output_list
        video_idx_list = self.video_idx_list
        dataset_test = self.dataset_test
        
        output = torch.cat(output_list, 0)
        prob = output.softmax(1)
        
        video_idx = torch.cat(video_idx_list, 0)
        df = pd.DataFrame(prob.numpy())
        df['video_index'] = video_idx
        df_prob_mean = df.groupby('video_index').mean()
        
        target_ser = df_prob_mean.index.map(lambda idx: dataset_test.samples[idx][1])
        pred = df_prob_mean.idxmax('columns')
        
        self.output = output
        self.prob = prob
        self.df = df
        self.df_prob_mean = df_prob_mean
        self.target_ser = target_ser
        # self.pred = pred
        
    def prepare_display(self, adjust_threshold=0.7):
        self.adjust_threshold = adjust_threshold

        df_prob_mean = self.df_prob_mean
        target_ser = self.target_ser

        pred = df_prob_mean.idxmax('columns')
        
        acc = (pred == target_ser).mean()
        
        ct = pd.crosstab(target_ser, pred, rownames=['target'], colnames=['pred'])
        df_pred_target = pd.DataFrame(dict(pred=pred, target=target_ser))
        df_merged = pd.merge(df_prob_mean, df_pred_target, left_index=True, right_index=True)
        
        df_prob_mean_adjusted = df_prob_mean.copy()
        df_prob_mean_adjusted[2] = (df_prob_mean_adjusted[2] > adjust_threshold) * df_prob_mean_adjusted[2]
        pred_adjusted = df_prob_mean_adjusted.idxmax(1)
        ct_adjusted = pd.crosstab(target_ser, pred_adjusted, rownames=['target'], colnames=['pred_adjusted'])

        acc_adjusted = (pred_adjusted == target_ser).mean()
        
        self.pred = pred
        self.ct = ct
        self.ct_adjusted = ct_adjusted
        self.acc = acc
        self.acc_adjusted = acc_adjusted
        self.df_merged = df_merged
        
    def display(self):
        if hasattr(self, 'acc1_list') and hasattr(self, 'batch_size_list'):
            acc1_list = getattr(self, 'acc1_list')
            batch_size_list = getattr(self, 'batch_size_list')
            acc1_arr = np.array(acc1_list)
            batch_size_arr = np.array(batch_size_list)
            acc_frame = np.sum(acc1_arr * batch_size_arr)/batch_size_arr.sum()
            print(f'acc (frame): {acc_frame}')

        print(f"acc: {self.acc}")
        print("confuse matrix:")
        display(self.ct)
        print("Wrong example show:")
        display(self.df_merged[self.target_ser != self.pred])
        print(f"Adjustment threshold={self.adjust_threshold}")       
        print(f"acc (adjusted): {self.acc_adjusted}")
        print("confuse matrix (adjusted):")
        display(self.ct_adjusted)

    def do_work(self, adjust_threshold=0.7):
        self.load()
        self.compute()
        self.summary()
        self.prepare_display(adjust_threshold=adjust_threshold)


class DisplaySuit(EvalSuit):
    '''
    Used to call prepare_display, display without extra parameters
    '''
    def __init__(self, *, df_prob_mean, target_ser):
        self.df_prob_mean = df_prob_mean
        self.target_ser = target_ser
