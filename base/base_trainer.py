import os
import time
import torch
import pandas as pd
from util import *
from abc import abstractmethod
import index.early_stopping as module_elst
import matplotlib.pyplot as plt

class BaseTrainer:
    def __init__(self, config, model, training):
        self.model = model
        self.training = training
        self.config = config
        self.epochs = config['trainer']['epochs'] if training else 1
        self.dataname = config['data_loader']['args']['dataname']
        self.filename = config['model'] + '_' + config['tuning_version']

    @abstractmethod
    def _one_tr_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def _one_ts_epoch(self, phase):
        raise NotImplementedError

    def train(self):
        dataname = self.dataname
        filename = f'{self.filename}.pt'
        result_list = pd.DataFrame()
        stt_time = time.time()

        # Train/Validation 상황
        # --------------------------------------------------------------------------
        if self.training:

            epoch_list   = []
            time_list    = []
            tr_loss_list = []
            tr_accu_list = []
            vl_loss_list = []
            vl_accu_list = []

            best_vl_loss = float('inf')
            best_vl_accu = 0.0
            es = init_obj(self.config, 'early_stopping', module_elst, dataname, filename)

            for epoch in range(self.epochs):

                # One Epoch 진행
                # --------------------------------------------------------------------------
                tr_loss, tr_accu, vl_loss, vl_accu = self._one_tr_epoch()

                # One Epoch Time 체크
                epoch_time = time.time()
                r_min, r_sec = time_remained(stt_time, epoch_time, epoch+1, self.epochs)

                # Loss/Accuracy 출력
                # --------------------------------------------------------------------------
                print('-'*90) 
                print(f'Epoch: {epoch+1:>02}/{self.epochs:03} {"*"*12} Time left: {r_min:02}m {r_sec:02}s')
                print(f'-> Train Loss: {tr_loss:.3f} | Train Accuracy: {tr_accu*100:.2f}%')
                print(f'-> Valid Loss: {vl_loss:.3f} | Valid Accuracy: {vl_accu*100:.2f}%')

                # 저장
                # --------------------------------------------------------------------------
                epoch_list.append(epoch+1)
                time_list.append(epoch_time-stt_time)
                tr_loss_list.append(tr_loss)
                tr_accu_list.append(tr_accu)
                vl_loss_list.append(vl_loss)
                vl_accu_list.append(vl_accu)
                if vl_loss < best_vl_loss: best_vl_loss = vl_loss;
                if vl_accu > best_vl_accu: best_vl_accu = vl_accu;

                # 현재 Epoch의, Early Stopping 여부 체크
                # --------------------------------------------------------------------------
                elapsed_epoch = epoch+1
                es(elapsed_epoch, vl_loss, self.model)
                if es.early_stop: break;


            # Total Time 체크
            # --------------------------------------------------------------------------
            end_time = time.time()
            e_min, e_sec = time_elapsed(stt_time, end_time)

            result_list = pd.DataFrame(data=list(zip(epoch_list,time_list,tr_loss_list,tr_accu_list,vl_loss_list,vl_accu_list)),
                                       columns=['epoch','time','tr_loss','tr_accu','vl_loss','vl_accu'])

            print('='*90)
            print('='*37,'Train Complete','='*37)
            print('='*90)
            print(f'Time elapsed: {e_min:02}m {e_sec:02}s | "{self.filename}" Best Valid Accuracy: {best_vl_accu*100:.2f}%')
            print('')

        # Test 상황
        # --------------------------------------------------------------------------
        else:
            trained = torch.load(os.path.join('saved',dataname,'model',filename))
            self.model.load_state_dict(trained['model_state_dict'])
            train_epoch  = trained['epoch']
            train_epochs = self.config['trainer']['epochs']
            ts_loss, ts_accu = self._one_ts_epoch('ts')
            print(f'Test Loss: {ts_loss:.3f} | Test Accuracy: {ts_accu*100:.2f}%')
            print(f'-> from Train Epoch {train_epoch:>02} out of {train_epochs:03}')
            print('')
            end_time = time.time()
            result_list = pd.DataFrame(data=list(zip([self.epochs],[end_time-stt_time],[ts_loss],[ts_accu])),
                                       columns=['epoch','time','ts_loss','ts_accu'])
            elapsed_epoch = 1
        return result_list, elapsed_epoch


    def save_to_csv(self, result_list, elapsed_epoch, filename):

        training = 'train' if self.training else 'test'

        # config_list 생성
        # --------------------------------------------------------------------------
        config_list= []
        config_list.append(pd.DataFrame(index=[0], data=self.config['model'], columns=['model']))
        config_list.append(pd.DataFrame(index=[0], data=self.config['tuning_version'], columns=['tuning_version']))
        config_list.append(pd.DataFrame(index=[0], data=self.config['data_loader']['args']))
        config_list.append(pd.DataFrame(index=[0], data=self.config['criterion'], columns=['criterion']))
        config_list.append(pd.DataFrame(index=[0], data=self.config['metrics'], columns=['metrics']))
        config_list.append(pd.DataFrame(index=[0], data=self.config['optimizer'], columns=['optimizer']))
        config_list.append(pd.DataFrame(index=[0], data=self.config[self.config['optimizer']]))
        config_list.append(pd.DataFrame(index=[0], data=self.config['scheduler'], columns=['scheduler']))
        config_list.append(pd.DataFrame(index=[0], data=self.config[self.config['scheduler']]))
        config_list.append(pd.DataFrame(index=[0], data=self.config['trainer']))
        config_list.append(pd.DataFrame(index=[0], data=self.config['early_stopping']['args']))

        # config_list 크기 조절
        # --------------------------------------------------------------------------
        config_list=pd.concat(config_list, axis=1, ignore_index=False)
        config_list=pd.concat([config_list]*(elapsed_epoch), axis=0, ignore_index=True)

        # total_list 생성
        # --------------------------------------------------------------------------
        total_list = pd.concat([result_list, config_list], axis=1, ignore_index=False)

        # 기존 파일에 덮어쓰기
        # --------------------------------------------------------------------------
        if filename in os.listdir(os.path.join(os.getcwd(),'saved',self.dataname,training)):
            prev_list = pd.read_csv(os.path.join(os.getcwd(),'saved',self.dataname,training,filename))
            total_list = pd.concat([prev_list, total_list], axis=0, ignore_index=True)
            total_list.drop_duplicates(['model', 'tuning_version', 'epoch'], keep='last', inplace=True, ignore_index=True)
            os.remove(os.path.join(os.getcwd(),'saved',self.dataname,training,filename))

        # 저장
        # --------------------------------------------------------------------------
        total_list.to_csv(os.path.join(os.getcwd(),'saved',self.dataname,training,filename), index=False)

    def draw_chart(self, result_list):
        plt.style.use('seaborn-whitegrid')
        fig, axes = plt.subplots(1,2, figsize=(12,5))

        axes[0].set_title("Loss", fontsize=16)
        axes[0].plot(result_list['tr_loss'],label='Train')
        axes[0].plot(result_list['vl_loss'],label='Valid')
        axes[0].legend()
        axes[1].set_title("Accuracy", fontsize=16)
        axes[1].plot(result_list['tr_accu'],label='Train')
        axes[1].plot(result_list['vl_accu'],label='Valid')
        axes[1].legend()
        
        fig.savefig('garbage.png', dpi=300)
