import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import sys

class MockArgs:
    def __init__(self):
        self.augmentation_ratio = 0
        self.num_workers = 0  # DataLoader用


def run_augmentation_single(data_x, data_y, args):
    # 拡張処理のモック（実際の処理は省略）
    return data_x, data_y, None


def get_ppg_data_loader(args, csv_file_path='all.csv', flag='train', size=None,
                       target_column='arm_ppg', scale=True, stride=None,
                       batch_size=32, shuffle_flag=True, drop_last=True):
    """
    PPGデータセットとDataLoaderを返す関数
    
    Args:
        args: 引数オブジェクト
        csv_file_path: all.csvファイルのパス
        flag: 'train', 'val', 'test'のいずれか
        size: [seq_len, label_len, pred_len]のサイズ
        target_column: 使用するターゲット列名（'arm_ppg'など）
        scale: スケーリングするかどうか
        stride: ストライド幅（Noneの場合はseq_lenと同じ値）
        batch_size: バッチサイズ
        shuffle_flag: シャッフルするかどうか
        drop_last: 最後のバッチを削除するかどうか
    
    Returns:
        data_set: データセット
        data_loader: DataLoader
    """
    data_set = Dataset_PPG(
        args=args,
        csv_file_path=csv_file_path,
        flag=flag,
        size=size,
        target_column=target_column,
        scale=scale,
        stride=stride
    )
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    
    return data_set, data_loader


class Dataset_PPG(Dataset):
    def __init__(self, args, csv_file_path='all.csv', flag='train', size=None,
                 target_column='arm_ppg', scale=True, stride=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        
        # デフォルトサイズ設定
        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # ストライド設定（Noneの場合はseq_lenと同じ値）
        if stride is None:
            self.stride = self.seq_len
        else:
            self.stride = stride
        
        # データセット分割設定
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.csv_file_path = csv_file_path
        self.target_column = target_column
        self.scale = scale
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # all.csvを読み込み
        df_meta = pd.read_csv(self.csv_file_path)
        
        # 各人のシーケンスを個別に処理
        all_person_sequences = []  # 各人のシーケンスを格納
        all_person_time_sequences = []  # 各人の時間特徴量シーケンスを格納
        all_person_labels = []  # 各人のラベルを格納
        
        for idx, row in df_meta.iterrows():
            file_path = row['path'] + '/final/1KHz/FINGER_TIP_PPG_final1.csv'
            label = row['label']
            
            if os.path.exists(file_path):
                print(f"Processing: {file_path}")
                df_ppg = pd.read_csv(file_path)

                
                # arm_ppgデータを取得
                ppg_data = df_ppg[self.target_column].values
                # print(self.target_column)
                # print(ppg_data)
                # sys.exit()
                print(f"  Original samples長さ: {len(ppg_data)}")
                
                # タイムスタンプデータを取得（一列目）
                timestamps = df_ppg.iloc[:, 0].values
                
                # このファイル内でのタイムスタンプを正規化
                relative_timestamps = timestamps - timestamps[0]
                
                # 分、秒、ミリ秒に分解
                minutes = (relative_timestamps // (1000 * 60)) % 60
                seconds = (relative_timestamps // 1000) % 60
                milliseconds = relative_timestamps % 1000
                
                # 正規化
                minutes_norm = minutes / 60.0
                seconds_norm = seconds / 60.0
                milliseconds_norm = milliseconds / 1000.0
                
                # このファイルの時間特徴量を作成
                file_time_features = np.column_stack([
                    minutes_norm, seconds_norm, milliseconds_norm
                ])
                
                # データを1次元から2次元に変換
                file_data = ppg_data.reshape(-1, 1)

                # この人のデータからシーケンスを作成
                OL = len(file_data)  # Original Length
                SL = self.stride     # Stride length
                SPL = self.seq_len + self.pred_len  # seq_len + pred_len

                if OL >= SPL:
                    # ユーザーの公式: N = [(OL-SPL)/SL] + 1
                    N = ((OL - SPL) // SL) + 1  # 切り出し可能な個数
                    
                    if N > 0:
                        print(f"  Available sequences: {N}個数")
                        
                        # この人のシーケンスを作成
                        person_sequences = []
                        person_time_sequences = []
                        person_labels = []
                        
                        for seq_idx in range(N):
                            # シーケンスの開始位置
                            s_begin = seq_idx * SL
                            s_end = s_begin + self.seq_len
                            r_begin = s_end - self.label_len
                            r_end = r_begin + self.label_len + self.pred_len
                            
                            # データとラベル用のシーケンスを作成
                            seq_x = file_data[s_begin:s_end]
                            seq_y = file_data[r_begin:r_end]
                            seq_x_mark = file_time_features[s_begin:s_end]
                            seq_y_mark = file_time_features[r_begin:r_end]
                            
                            person_sequences.append((seq_x, seq_y))
                            person_time_sequences.append((seq_x_mark, seq_y_mark))
                            person_labels.append(label)
                        
                        # この人のシーケンスを全体リストに追加
                        all_person_sequences.extend(person_sequences)
                        all_person_time_sequences.extend(person_time_sequences)
                        all_person_labels.extend(person_labels)
                        
                        print(f"  Created sequences: {len(person_sequences)}")
                        print(f"  Duration: {(timestamps[-1] - timestamps[0])/1000:.2f}s")
                    else:
                        print(f"  Skipped: Not enough data for even 1 sequence")
                else:
                    print(f"  Skipped: File too short ({len(file_data)} < {SPL})")
            else:
                print(f"Warning: File not found: {file_path}")
        
        if not all_person_sequences:
            raise ValueError("No valid sequences found!")
        
        print(f"\nTotal sequences created: {len(all_person_sequences)}")
        
        # シーケンスをnumpy配列に変換
        all_seq_x = []
        all_seq_y = []
        all_seq_x_mark = []
        all_seq_y_mark = []
        
        for (seq_x, seq_y), (seq_x_mark, seq_y_mark) in zip(all_person_sequences, all_person_time_sequences):
            all_seq_x.append(seq_x)
            all_seq_y.append(seq_y)
            all_seq_x_mark.append(seq_x_mark)
            all_seq_y_mark.append(seq_y_mark)
        
        # numpy配列に変換
        all_seq_x = np.array(all_seq_x)  # (total_sequences, seq_len, 1)
        all_seq_y = np.array(all_seq_y)  # (total_sequences, label_len+pred_len, 1)
        all_seq_x_mark = np.array(all_seq_x_mark)  # (total_sequences, seq_len, 3)
        all_seq_y_mark = np.array(all_seq_y_mark)  # (total_sequences, label_len+pred_len, 3)
        
        # データセット分割（train: 100%, val: 0%, test: 0%）← 全データ確認用
        total_sequences = len(all_seq_x)
        train_len = int(total_sequences * 1.0)  # 100%
        val_len = int(total_sequences * 0.0)    # 0%
        
        border1s = [0, train_len, train_len + val_len]
        border2s = [train_len, train_len + val_len, total_sequences]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 該当するセットのデータを取得
        selected_seq_x = all_seq_x[border1:border2]
        selected_seq_y = all_seq_y[border1:border2]
        selected_seq_x_mark = all_seq_x_mark[border1:border2]
        selected_seq_y_mark = all_seq_y_mark[border1:border2]
        
        # スケーリング（trainデータでfitし、全体に適用）
        if self.scale:
            # スケーリング用にtrainデータを2D形状に変換
            train_seq_x = all_seq_x[border1s[0]:border2s[0]]  # (train_sequences, seq_len, 1)
            train_data_for_scaling = train_seq_x.reshape(-1, 1)  # (train_sequences * seq_len, 1)
            
            self.scaler.fit(train_data_for_scaling)
            
            # 選択されたデータをスケーリング
            selected_seq_x_scaled = []
            selected_seq_y_scaled = []
            
            for seq_x, seq_y in zip(selected_seq_x, selected_seq_y):
                seq_x_scaled = self.scaler.transform(seq_x)
                seq_y_scaled = self.scaler.transform(seq_y)
                selected_seq_x_scaled.append(seq_x_scaled)
                selected_seq_y_scaled.append(seq_y_scaled)
            
            selected_seq_x = np.array(selected_seq_x_scaled)
            selected_seq_y = np.array(selected_seq_y_scaled)
        
        # データセット用のデータを設定
        self.sequences_x = selected_seq_x
        self.sequences_y = selected_seq_y
        self.sequences_x_mark = selected_seq_x_mark
        self.sequences_y_mark = selected_seq_y_mark
        
        print(f"\nDataset {['train', 'val', 'test'][self.set_type]} prepared:")
        print(f"  Total sequences: {len(self.sequences_x)}")
        print(f"  seq_x shape: {self.sequences_x.shape}")
        print(f"  seq_y shape: {self.sequences_y.shape}")
        print(f"  seq_x_mark shape: {self.sequences_x_mark.shape}")
        print(f"  seq_y_mark shape: {self.sequences_y_mark.shape}")

    def __getitem__(self, index):
        # 既に作成済みのシーケンスを直接返す
        seq_x = self.sequences_x[index]
        seq_y = self.sequences_y[index]
        seq_x_mark = self.sequences_x_mark[index]
        seq_y_mark = self.sequences_y_mark[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # 作成済みのシーケンス数を返す
        return len(self.sequences_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) 