import torch
from torch.utils.data import Dataset
import numpy as np

class price_high_low_variance(Dataset):
    '''
    A dataset used to predict the high, low, and variance of a stock over a given horizon
    
    Params:
    --------
    df: pd.DataFrame
        The dataframe containing the stock data
    in_winsize: int
        The number of timesteps to use as input
    in_keys: list
        The keys of the columns of DF to use as input
    out_winsize: int
        The number of timesteps to aggregate to get the output
    out_key: str
        The key of the column to predict
        
    Returns:
    --------
    x: torch.Tensor
        The input tensor containing the in_keys over the in_winsize
    y: torch.Tensor
        The output tensor containing the max, min, and std of the out_key over the out_winsize
    '''
    def __init__(self, df, in_winsize, in_keys, out_winsize, out_key):
        self.df = df
        self.in_winsize = in_winsize
        self.in_keys = in_keys
        self.out_winsize = out_winsize
        self.out_key = out_key

    def __len__(self):
        return len(self.df) - self.in_winsize - self.out_winsize

    def __getitem__(self, idx):
        x = torch.tensor(self.df.iloc[idx:idx+self.in_winsize][self.in_keys].values)
        _y = torch.tensor(self.df.iloc[idx+self.in_winsize:idx+self.in_winsize+self.out_winsize][self.out_key].values)
        y = torch.tensor([_y.max(), _y.min(), _y.std()])
        return x, y
    
class quartile_dataset(Dataset):
    '''
    A dataset used to predict the data quartiles over a given horizon
    
    Params:
    --------
    df: pd.DataFrame
        The dataframe containing the stock data
    in_winsize: int
        The number of timesteps to use as input
    in_keys: list
        The keys of the columns of DF to use as input
    out_winsize: int
        The number of timesteps to aggregate to get the output
    out_key: str
        The key of the column to predict
    num_quartiles: int
        The number of quartiles to predict
    include_max_min: bool
        If True, includes the max and min of the output in the output tensor along with the quartiles. 
        They are placed in order (i.e. min, q1, q2, ... qn, max)
    normalize: bool
        If True, normalizes the returned data
    debug: bool
        If True, returns the actual output segment as well as the quartiles

        
    Returns:
    --------
    x: torch.Tensor
        The input tensor containing the in_keys over the in_winsize
        of shape (len(in_keys), in_winsize)
    y: torch.Tensor
        The output tensor containing the max, min, and std of the out_key over the out_winsize
    '''
    def __init__(self, df, in_winsize, out_winsize, out_key, in_keys=None, num_quartiles=5, include_max_min=True, normalize=True, debug=False):
        self.df = df
        self.in_winsize = in_winsize
        if in_keys is None:
            in_keys = df.columns
        self.in_keys = in_keys
        self.out_winsize = out_winsize
        self.out_key = out_key
        self.num_quartiles = num_quartiles
        self.debug = debug
        self.normalize = normalize
        self.out_idx = df.columns.get_loc(out_key)
        self.include_max_min = include_max_min

    def __len__(self):
        return len(self.df) - self.in_winsize - self.out_winsize

    def __getitem__(self, idx):
        x = torch.tensor(self.df.iloc[idx:idx+self.in_winsize][self.in_keys].values).T
        _y = torch.tensor(self.df.iloc[idx+self.in_winsize:idx+self.in_winsize+self.out_winsize][self.out_key].values)
        
        if self.normalize:
            _y = (_y - x[self.out_idx].mean()) / x[self.out_idx].std() #normalize y by
            x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
        
        quartiles = torch.tensor([_y.quantile(i/self.num_quartiles) for i in range(1, self.num_quartiles)])
        if self.include_max_min:
            quartiles = torch.cat([torch.tensor([_y.min()]), quartiles, torch.tensor([_y.max()])])
        if self.debug:
            return x, quartiles, _y
        else:
            return x, quartiles
    
class Multichannel_1d_to_2d_wrapper(Dataset):
    '''
    A dataset wrapper that converts a 1D dataset to a 2D dataset
    
    Params:
    --------
    dataset: Dataset
        The dataset to wrap. Assumes that the dataset returns a 2D tensor of shape (num_channels, num_timesteps)
    window_size: int
        The number of timesteps to use as input
    stride: int
        The number of timesteps to skip between each input
        
    Returns:
    --------
    x: torch.Tensor
        The input tensor reshaped by stacking the input tensors over the window size with a stride
    y: torch.Tensor
        The unchanged output tensor
    '''
    def __init__(self, dataset, window_size, stride):
        self.dataset = dataset
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = x.unfold(1, self.window_size, self.stride) #.permute(0, 2, 1)
        return x, y
    
class RandomDownsampleDataset(Dataset):
    '''
    A dataset that downsamples another dataset.
    Useful if the dataset is larger than desired for a single epoch.
    downsampling can be in order or rendom permutation
    Params
    ---------
    dataset: Dataset
        The dataset to downsample
    out_size: int
        The size of the downsampled dataset
    method: str
        The downsampling method. Can be 'ordered' or 'random'
    '''
    def __init__(self, dataset, out_size, method='random'):
        self.dataset = dataset
        self.method = method
        self.out_size = out_size
        self.downsampled_idxs = [0]
        self.next_indecies()
        
    def __len__(self):
        return self.out_size
    
    def __getitem__(self, idx):
        _idx = self.downsampled_idxs[idx]
        return self.dataset[_idx]
    
    def next_indecies(self):
        if self.method == 'ordered':
            last_idx = self.downsampled_idxs[-1]
            self.downsampled_idxs = list(range(last_idx, min(last_idx + self.out_size, len(self.dataset))))
            if len(self.downsampled_idxs) < self.out_size:
                self.downsampled_idxs += list(range(self.out_size - len(self.downsampled_idxs))) #fill in the rest by wrapping around
        elif self.method == 'random':
            self.downsampled_idxs = np.random.choice(len(self.dataset), self.out_size, replace=False)
