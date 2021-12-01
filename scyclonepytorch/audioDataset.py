import os
import torch
from torch import Tensor, save, load
from torchaudio import load as load_wav
from torchaudio.transforms import Spectrogram, Resample
from torch.utils.data import Dataset
import glob

def pad_last_dim(d: Tensor, length_min: int = 160) -> Tensor:
    """
    Pad last dimension with 0 if length is not enough.
    If input is longer than `length_min`, nothing happens.
    [..., L<160] => [..., L==160]
    """
    shape = d.size()
    length_d = shape[-1]
    if length_d < length_min:
        a = torch.zeros([*shape[:-1], length_min - length_d])
        return torch.cat((d, a), -1)
    else:
        return d


def slice_last_dim(d: Tensor, length: int = 160) -> Tensor:
    """
    Slice last dimention if length is too much.
    If input is shorter than `length`, error is thrown.
    [..., L>160] => [..., L==160]
    """
    start = torch.randint(0, d.size()[-1] - (length - 1), (1,)).item()
    return torch.narrow(d, -1, start, length)


def pad_clip(d: Tensor) -> Tensor:
    return slice_last_dim(pad_last_dim(d))

class audioDataset(Dataset):
    def __init__(self, path=None, train=True):
        self.path = path
        self.all_data = []
        path_list = glob.glob(self.path + "/*") #glob,os,pathlibなどのモジュールを使うことが考えられる。その際、cls_dirを用いるとよい
        for path in path_list:
            waveform, _sr_orig = load_wav(path)
            if _sr_orig != 16000:
                channel = 0
                waveform = Resample(_sr_orig, 16000)(waveform[channel, :].view(1, -1))
            waveform: Tensor = waveform[0, :]
            spec: Tensor = Spectrogram(254)(waveform)
            if train == True:
                spec = pad_clip(spec)
                self.all_data.append(spec)
            else :
                num = spec.size()[1] // 160 + 1
                print("num is {}".format(num))
                spec = torch.split(spec, 161, dim=1)
                for i in range(num):
                    if i + 1 == num:
                        self.all_data.append(pad_clip(spec[i]))
                    else:
                        self.all_data.append(spec[i][:, :160])
                    print(spec[i].size())

        print("specsize is {}".format(self.all_data[0].size()))

    def __len__(self):
        #データセットの数を返す関数
        return len(self.all_data)
    
    def __getitem__(self, idx):
        # TODO 画像とラベルの読み取り
        #self.all_dataを用いる
        return self.all_data[idx]

if __name__ == "__main__":
    audiodata = audioDataset("scyclonepytorch/data/dataset")
    print(len(audiodata))