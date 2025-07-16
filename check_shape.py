import argparse
from dataset_ett import Dataset_ETT_hour, MockArgs as ETTMockArgs, get_data_loader
from dataset_ppg import Dataset_PPG, MockArgs as PPGMockArgs, get_ppg_data_loader
import numpy as np


def test_ett_dataset_shapes(custom_size=None, show_dataloader=False, show_all_data=False):
    """ETTデータセットのshape確認（DataLoader確認も統合）"""
    
    if custom_size is None:
        custom_size = [96, 48, 96]  # デフォルトサイズ
    
    print("=== ETTデータセットのshape確認 ===")
    print(f"カスタムサイズ: seq_len={custom_size[0]}, label_len={custom_size[1]}, pred_len={custom_size[2]}")
    print(f"DataLoader確認: {show_dataloader}")
    print(f"全データ表示: {show_all_data}")
    print()
    
    # パラメータ設定
    args = ETTMockArgs()
    root_path = '/home/nonat/Desktop/tanaka2/20250715/PUC01/dataset/ETT-small'
    data_path = 'ETTh1.csv'
    
    # 各フラグ（train, val, test）で確認
    for flag in ['train', 'val', 'test']:
        print(f"--- {flag.upper()} データセット ---")
        
        try:
            if show_dataloader and flag == 'train':
                # DataLoader確認の場合はget_data_loaderを使用
                data_set, data_loader = get_data_loader(
                    args=args,
                    root_path=root_path,
                    flag=flag,
                    size=custom_size,
                    features='S',
                    data_path=data_path,
                    target='OT',
                    scale=True,
                    batch_size=32,
                    shuffle_flag=True,
                    drop_last=True
                )
                dataset_ett = data_set
                
                print(f"ETT {data_path} ({flag}):")
                print(f"  data_x shape: {dataset_ett.data_x.shape}")
                print(f"  data_y shape: {dataset_ett.data_y.shape}")
                print(f"  data_stamp shape: {dataset_ett.data_stamp.shape}")
                print(f"  データセット長: {len(dataset_ett)}")
                print(f"  seq_len: {dataset_ett.seq_len}, label_len: {dataset_ett.label_len}, pred_len: {dataset_ett.pred_len}")
                
                # DataLoader情報
                print(f"\nDataLoader情報:")
                print(f"  Dataset長: {len(data_set)}")
                print(f"  バッチ数: {len(data_loader)}")
                
                # 最初のバッチを取得
                for batch_data in data_loader:
                    seq_x, seq_y, seq_x_mark, seq_y_mark = batch_data
                    print(f"  バッチshape:")
                    print(f"    seq_x: {seq_x.shape}")
                    print(f"    seq_y: {seq_y.shape}")
                    print(f"    seq_x_mark: {seq_x_mark.shape}")
                    print(f"    seq_y_mark: {seq_y_mark.shape}")
                    break
                
                # 全データ表示が要求されている場合
                if show_all_data:
                    print("\n" + "="*40)
                    print("全シーケンス詳細表示:")
                    print("="*40)
                    show_all_dataset_data(data_set, dataset_type='ett', max_sequences=10)
                
            else:
                # 通常のデータセット確認
                dataset_ett = Dataset_ETT_hour(
                    args=args,
                    root_path=root_path,
                    flag=flag,
                    features='S',
                    data_path=data_path,
                    target='OT',
                    size=custom_size
                )
                
                print(f"ETT {data_path} ({flag}):")
                print(f"  data_x shape: {dataset_ett.data_x.shape}")
                print(f"  data_y shape: {dataset_ett.data_y.shape}")
                print(f"  data_stamp shape: {dataset_ett.data_stamp.shape}")
                print(f"  データセット長: {len(dataset_ett)}")
                print(f"  seq_len: {dataset_ett.seq_len}, label_len: {dataset_ett.label_len}, pred_len: {dataset_ett.pred_len}")
                
                # show_all_dataが要求されていて、dataloaderが実行されていない場合
                if show_all_data and not show_dataloader and flag == 'train':
                    print("\n" + "="*40)
                    print("全シーケンス詳細表示:")
                    print("="*40)
                    show_all_dataset_data(dataset_ett, dataset_type='ett', max_sequences=10)
            
            # サンプルデータの取得（共通）
            if len(dataset_ett) > 0:
                sample_data = dataset_ett[0]
                print(f"  サンプルデータのshape:")
                print(f"    seq_x: {sample_data[0].shape}")
                print(f"    seq_y: {sample_data[1].shape}")
                print(f"    seq_x_mark: {sample_data[2].shape}")
                print(f"    seq_y_mark: {sample_data[3].shape}")
            else:
                print("  データセットが空です")
                
        except Exception as e:
            print(f"  エラー: {e}")
        
        print()


def test_ppg_dataset_shapes(csv_file='test_all.csv', custom_size=None, target_column='arm_ppg', scale=True, stride=None, show_dataloader=False, show_all_data=False):
    """PPGデータセットのshape確認（DataLoader確認も統合）"""
    
    if custom_size is None:
        custom_size = [4, 2, 4]  # デフォルトはテスト用サイズ
    
    print("=== PPGデータセットのshape確認 ===")
    print(f"カスタムサイズ: seq_len={custom_size[0]}, label_len={custom_size[1]}, pred_len={custom_size[2]}")
    print(f"CSVファイル: {csv_file}")
    print(f"ターゲット列: {target_column}")
    print(f"スケーリング: {scale}")
    print(f"ストライド: {stride}")
    print(f"DataLoader確認: {show_dataloader}")
    print(f"全データ表示: {show_all_data}")
    print()
    
    # 各フラグ（train, val, test）で確認
    for flag in ['train', 'val', 'test']:
        print(f"--- {flag.upper()} データセット ---")
        
        args = PPGMockArgs()
        
        try:
            if show_dataloader and flag == 'train':
                # DataLoader確認の場合はget_ppg_data_loaderを使用
                data_set, data_loader = get_ppg_data_loader(
                    args=args,
                    csv_file_path=csv_file,
                    flag=flag,
                    size=custom_size,
                    target_column=target_column,
                    scale=scale,
                    stride=stride,
                    batch_size=2,
                    shuffle_flag=True,
                    drop_last=False
                )
                dataset_ppg = data_set
                
                print(f"PPG {target_column} (1次元):")
                print(f"  sequences_x shape: {dataset_ppg.sequences_x.shape}")
                print(f"  sequences_y shape: {dataset_ppg.sequences_y.shape}")
                print(f"  sequences_x_mark shape: {dataset_ppg.sequences_x_mark.shape}")
                print(f"  sequences_y_mark shape: {dataset_ppg.sequences_y_mark.shape}")
                print(f"  データセット長: {len(dataset_ppg)}")
                print(f"  seq_len: {dataset_ppg.seq_len}, label_len: {dataset_ppg.label_len}, pred_len: {dataset_ppg.pred_len}")
                print(f"  stride: {dataset_ppg.stride}")
                
                # DataLoader情報
                print(f"\nDataLoader情報:")
                print(f"  Dataset長: {len(data_set)}")
                print(f"  バッチ数: {len(data_loader)}")
                
                # 最初のバッチを取得
                for batch_data in data_loader:
                    seq_x, seq_y, seq_x_mark, seq_y_mark = batch_data
                    print(f"  バッチshape:")
                    print(f"    seq_x: {seq_x.shape}")
                    print(f"    seq_y: {seq_y.shape}")
                    print(f"    seq_x_mark: {seq_x_mark.shape}")
                    print(f"    seq_y_mark: {seq_y_mark.shape}")
                    break
                
                # 全データ表示が要求されている場合
                if show_all_data:
                    print("\n" + "="*40)
                    print("全シーケンス詳細表示:")
                    print("="*40)
                    show_all_dataset_data(data_set, dataset_type='ppg', max_sequences=None)
                
            else:
                # 通常のデータセット確認
                dataset_ppg = Dataset_PPG(
                    args=args,
                    csv_file_path=csv_file,
                    flag=flag,
                    size=custom_size,
                    target_column=target_column,
                    scale=False,
                    # scale=scale,
                    stride=stride
                )
                
                print(f"PPG {target_column} (1次元):")
                print(f"  sequences_x shape: {dataset_ppg.sequences_x.shape}")
                print(f"  sequences_y shape: {dataset_ppg.sequences_y.shape}")
                print(f"  sequences_x_mark shape: {dataset_ppg.sequences_x_mark.shape}")
                print(f"  sequences_y_mark shape: {dataset_ppg.sequences_y_mark.shape}")
                print(f"  データセット長: {len(dataset_ppg)}")
                print(f"  seq_len: {dataset_ppg.seq_len}, label_len: {dataset_ppg.label_len}, pred_len: {dataset_ppg.pred_len}")
                print(f"  stride: {dataset_ppg.stride}")
                
                # show_all_dataが要求されていて、dataloaderが実行されていない場合
                if show_all_data and not show_dataloader and flag == 'train':
                    print("\n" + "="*40)
                    print("全シーケンス詳細表示:")
                    print("="*40)
                    show_all_dataset_data(dataset_ppg, dataset_type='ppg', max_sequences=None)
            
            # サンプルデータの取得（共通）
            if len(dataset_ppg) > 0:
                sample_data = dataset_ppg[0]
                print(f"  サンプルデータのshape:")
                print(f"    seq_x: {sample_data[0].shape}")
                print(f"    seq_y: {sample_data[1].shape}")
                print(f"    seq_x_mark: {sample_data[2].shape}")
                print(f"    seq_y_mark: {sample_data[3].shape}")
                
                # 詳細データ確認
                print(f"  サンプルデータの内容:")
                print(f"    seq_x data: {sample_data[0].flatten()}")
                print(f"    seq_y data: {sample_data[1].flatten()}")
            else:
                print("  データセットが空です")
                
        except Exception as e:
            print(f"  エラー: {e}")
        
        print("\n" + "="*50 + "\n")


def show_all_dataset_data(dataset, dataset_type='ppg', max_sequences=None):
    """
    データセットの全データを詳細表示する関数
    
    Args:
        dataset: データセットオブジェクト
        dataset_type: データセットの種類 ('ett' または 'ppg')
        max_sequences: 表示する最大シーケンス数 (Noneの場合は全て)
    """
    print(f"\n=== {dataset_type.upper()}データセット全データ詳細表示 ===")
    
    total_sequences = len(dataset)
    display_sequences = min(total_sequences, max_sequences) if max_sequences else total_sequences
    
    print(f"総シーケンス数: {total_sequences}")
    print(f"表示シーケンス数: {display_sequences}")
    print("="*60)
    
    for i in range(display_sequences):
        sample_data = dataset[i]
        seq_x = sample_data[0]
        seq_y = sample_data[1]
        seq_x_mark = sample_data[2]
        seq_y_mark = sample_data[3]
        
        print(f"\n【シーケンス {i}】")
        print(f"Shape情報:")
        print(f"  seq_x: {seq_x.shape}")
        print(f"  seq_y: {seq_y.shape}")
        print(f"  seq_x_mark: {seq_x_mark.shape}")
        print(f"  seq_y_mark: {seq_y_mark.shape}")
        
        print(f"\nデータ内容:")
        print(f"  seq_x data: {seq_x.flatten()}")
        print(f"  seq_y data: {seq_y.flatten()}")
        
        # label_lenの重複確認
        if hasattr(dataset, 'label_len'):
            label_len = dataset.label_len
            if label_len > 0:
                seq_x_end = seq_x.flatten()[-label_len:]
                seq_y_start = seq_y.flatten()[:label_len]
                overlap_match = np.allclose(seq_x_end, seq_y_start, rtol=1e-10, atol=1e-10)
                print(f"  重複部分確認:")
                print(f"    seq_x末尾{label_len}個: {seq_x_end}")
                print(f"    seq_y先頭{label_len}個: {seq_y_start}")
                print(f"    重複一致: {'✓' if overlap_match else '✗'}")
            
            # 予測部分の表示
            if hasattr(dataset, 'pred_len'):
                pred_len = dataset.pred_len
                pred_start = seq_y.flatten()[label_len:label_len+pred_len]
                print(f"  予測部分 (label_len={label_len}以降{pred_len}個): {pred_start}")
        
        # 時間特徴量の確認
        print(f"  時間特徴量:")
        print(f"    seq_x_mark[0]: {seq_x_mark[0] if len(seq_x_mark) > 0 else 'なし'}")
        print(f"    seq_y_mark[0]: {seq_y_mark[0] if len(seq_y_mark) > 0 else 'なし'}")
        
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='データセットのshape確認プログラム')
    parser.add_argument('--dataset', type=str, choices=['ett', 'ppg', 'both'], default='ppg',
                        help='確認するデータセットの種類 (ett, ppg, both)')
    parser.add_argument('--flag', type=str, choices=['train', 'val', 'test'], default='train',
                        help='確認するデータセットのフラグ')
    parser.add_argument('--seq_len', type=int, default=4,
                        help='シーケンス長')
    parser.add_argument('--label_len', type=int, default=2,
                        help='ラベル長')
    parser.add_argument('--pred_len', type=int, default=4,
                        help='予測長')
    parser.add_argument('--test_dataloader', action='store_true',
                        help='DataLoaderの動作確認も実行する')
    parser.add_argument('--csv_file', type=str, default='test_all.csv',
                        help='PPGデータセット用のCSVファイル名')
    parser.add_argument('--target_column', type=str, default='arm_ppg',
                        help='PPGデータセットのターゲット列名')
    parser.add_argument('--scale', action='store_true', default=True,
                        help='データをスケーリングするかどうか')
    parser.add_argument('--stride', type=int, default=None,
                        help='ストライド幅（Noneの場合はseq_lenと同じ値）')
    parser.add_argument('--show_all_data', action='store_true',
                        help='生成されたすべてのデータを詳細表示する')
    
    args = parser.parse_args()
    
    # カスタムサイズ設定
    custom_size = [args.seq_len, args.label_len, args.pred_len]
    
    print(f"実行設定:")
    print(f"  データセット: {args.dataset}")
    print(f"  フラグ: {args.flag}")
    print(f"  サイズ: seq_len={args.seq_len}, label_len={args.label_len}, pred_len={args.pred_len}")
    print(f"  CSVファイル: {args.csv_file}")
    print(f"  ターゲット列: {args.target_column}")
    print(f"  スケーリング: {args.scale}")
    print(f"  ストライド: {args.stride}")
    print(f"  DataLoader確認: {args.test_dataloader}")
    print(f"  全データ表示: {args.show_all_data}")
    print("="*60)
    
    # 通常の確認モード
    if args.dataset in ['ett', 'both']:
        test_ett_dataset_shapes(custom_size=custom_size, show_dataloader=args.test_dataloader, show_all_data=args.show_all_data)
    
    if args.dataset in ['ppg', 'both']:
        if args.dataset == 'both':
            print("\n" + "="*60)
        test_ppg_dataset_shapes(csv_file=args.csv_file, custom_size=custom_size, target_column=args.target_column, scale=args.scale, stride=args.stride, show_dataloader=args.test_dataloader, show_all_data=args.show_all_data) 