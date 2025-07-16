import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class MockArgs:
    def __init__(self):
        self.augmentation_ratio = 0
        self.num_workers = 0  # DataLoader用


def run_augmentation_single(data_x, data_y, args):
    # 拡張処理のモック（実際の処理は省略）
    return data_x, data_y, None


def get_data_loader(args, root_path, flag='train', size=None,
                   features='S', data_path='ETTh1.csv',
                   target='OT', scale=True, timeenc=0, freq='h',
                   batch_size=32, shuffle_flag=True, drop_last=True):
    """
    データセットとDataLoaderを返す関数
    
    Args:
        args: 引数オブジェクト
        root_path: データのルートパス
        flag: 'train', 'val', 'test'のいずれか
        size: [seq_len, label_len, pred_len]のサイズ
        features: 'S' (単一特徴量) または 'M' (全特徴量)
        data_path: データファイルのパス
        target: ターゲット列名
        scale: スケーリングするかどうか
        timeenc: 時間エンコーディングのタイプ
        freq: 頻度
        batch_size: バッチサイズ
        shuffle_flag: シャッフルするかどうか
        drop_last: 最後のバッチを削除するかどうか
    
    Returns:
        data_set: データセット
        data_loader: DataLoader
    """
    data_set = Dataset_ETT_hour(
        args=args,
        root_path=root_path,
        flag=flag,
        size=size,
        features=features,
        data_path=data_path,
        target=target,
        scale=scale,
        timeenc=timeenc,
        freq=freq
    )
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    
    return data_set, data_loader


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        # 時間特徴量の処理を簡素化
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
        data_stamp = df_stamp.drop(['date'], axis=1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) 