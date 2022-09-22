import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.io.arff import loadarff
from base import BaseDataLoader

class DataLoader(BaseDataLoader):

    # config 들어왔을 때 dataset을 뽑아서 base로 넘겨주기
    # --------------------------------------------------------------------------
    def __init__(self, verbose, dataname, vl_ratio, ts_ratio, shuffle, normalize,
                i_stt, i_end, X_length, y_length, y_offset, stride_length):

        self.normalize = normalize

        # (i)dataname이 ('FordA','FordB','Wafer')가 아닌 경우 Error 반환하기
        # --------------------------------------------------------------------------
        assert dataname in os.listdir(os.path.join(os.getcwd(),'data')), 'Dataname Error'

        # 먼저, 2개의 arff 파일로 나뉘어져 있는 data 합치기
        # --------------------------------------------------------------------------
        data1, meta1 = loadarff(os.path.join(os.getcwd(),'data',dataname,dataname+'_TRAIN.arff'))
        data2, meta2 = loadarff(os.path.join(os.getcwd(),'data',dataname,dataname+'_TEST.arff' ))
        dataset1 = pd.DataFrame(data1, columns=meta1.names())
        dataset2 = pd.DataFrame(data2, columns=meta2.names())
        dataset  = pd.concat([dataset1, dataset2], axis=0, ignore_index=True)

        # target column값 -1,+1 --> 0,1로 조정하기
        # --------------------------------------------------------------------------
        dataset['target']= dataset['target'].astype(float)
        dataset['target'].replace(-1, 0, inplace=True)

        # (i)ratio+shuffle에 따라, data 나누기 (tr: train, vl: validation, ts: test)
        # --------------------------------------------------------------------------
        split_ratio = [int((1-vl_ratio-ts_ratio)*len(dataset)), int((1-ts_ratio)*len(dataset))]
        dataset = dataset.sample(frac=1, axis=0) if shuffle else dataset
        tr_dataset, vl_dataset, ts_dataset = np.split(dataset, split_ratio)

        # index 초기화하기
        # --------------------------------------------------------------------------
        tr_dataset.reset_index(drop=True, inplace=True)
        vl_dataset.reset_index(drop=True, inplace=True)
        ts_dataset.reset_index(drop=True, inplace=True)

        # X 정규화하기
        if self.normalize:
            tr_dataset = self._X_normalize(tr_dataset)
            vl_dataset = self._X_normalize(vl_dataset) if vl_ratio!=0 else vl_dataset
            ts_dataset = self._X_normalize(ts_dataset) if ts_ratio!=0 else ts_dataset

        # 넘겨줄 dict 생성하기
        # --------------------------------------------------------------------------
        self.dataset_dict = (tr_dataset, vl_dataset, ts_dataset)

        super().__init__(self.dataset_dict, verbose, i_stt, i_end, X_length, y_length, y_offset, stride_length)


    # X 정규화 함수 선언하기
    # --------------------------------------------------------------------------
    def _X_normalize(self, dataset):
        ss = StandardScaler()
        X=pd.DataFrame(ss.fit_transform(dataset.iloc[:,:-1]))
        y=dataset.iloc[:,-1]
        dataset = pd.concat([X,y], axis=1, ignore_index=False)
        return dataset