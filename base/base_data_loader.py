import torch

class BaseDataLoader():

    def __init__(self, dataset_dict, verbose, i_stt, i_end, X_length, y_length, y_offset, stride_length):

        self.dataset_dict = dataset_dict # (tr_dataset, vl_dataset, ts_dataset)
        self.verbose = verbose
        self.i_stt = i_stt
        self.i_end = i_end
        self.X_len = X_length
        self.y_len = y_length
        self.y_off = y_offset
        self.b_len = max(X_length, y_length + y_offset)
        self.s_len = stride_length

    def get_data(self):

        phase_list = ['tr','vl','ts']
        dataloader_list = []

        # (i)dataset_dict 각각 진행하기
        # --------------------------------------------------------------------------
        for _ in range(len(self.dataset_dict)):
            if self.verbose:
                print(f'{phase_list[_].upper()} Data: ','-'*80)
            dataset = self.dataset_dict[_]

            # 빈 phase일 경우 0 반환하기
            # --------------------------------------------------------------------------
            if dataset.empty:
                blank = torch.as_tensor([])
                dataloader_list.append((blank, blank))
                if self.verbose:
                    print('(Empty)')
            else:
                # (i)index에 따라, dataset 잘라주기
                # --------------------------------------------------------------------------
                i_stt_ = 0              if self.i_stt ==None else self.i_stt
                i_end_ = len(dataset)-1 if self.i_end ==None else self.i_end
                dataset = dataset[i_stt_ : i_end_+1].reset_index(drop=True)

                # batch_count 설정하기
                # --------------------------------------------------------------------------
                batch_count = int((len(dataset)-self.b_len)/self.s_len) +1
                if self.verbose:
                    print(f'Index Range: {i_stt_:>d} ~ {i_end_:>4d}, Batch Count: {batch_count},',
                        f'Batch Length(X/y(+offset)): {self.X_len}/{self.y_len}(+{self.y_off}), Stride Length: {self.s_len}')

                # (i)length+offset에 따라, dataset 분할하기
                # --------------------------------------------------------------------------
                dataset_X_list = []
                dataset_y_list = []
                for i in range(batch_count):
                    dataset_X_list.append(dataset.iloc[i*self.s_len           :i*self.s_len+self.X_len           ,:-1].values.tolist())
                    dataset_y_list.append(dataset.iloc[i*self.s_len+self.y_off:i*self.s_len+self.y_len+self.y_off, -1].values.tolist())
                dataloader_X = torch.as_tensor(dataset_X_list).float()
                dataloader_y = torch.as_tensor(dataset_y_list).long()
                if self.verbose:
                    print(f'X Tensor Shape: {dataloader_X.shape}, y Tensor Shape: {dataloader_y.shape}')

                # 결과 tensor 저장하기
                dataloader_list.append((dataloader_X, dataloader_y))

        # 출력할 dict 생성하기
        # --------------------------------------------------------------------------
        dataloader_dict = dict(zip(phase_list, dataloader_list))
        if self.verbose:
            print('='*90)
        print('='*35,'Data Load Complete','='*35)

        return dataloader_dict