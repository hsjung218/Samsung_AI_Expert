import torch
from base import BaseTrainer

class Trainer(BaseTrainer):

    def __init__(self, config, data_dict, model, device, crit, metr, opti, sche, training):
        super().__init__(config, model, training)

        self.data_dict = data_dict
        self.model = model
        self.device = device
        self.crit = crit
        self.metr = metr
        self.opti = opti
        self.sche = sche

    def _one_tr_epoch(self):
        epoch_loss = 0
        epoch_accu = 0
        epoch_totl = 0

        (X_all, y_all) = self.data_dict['tr']
        batch_numb = self.data_dict['tr'][0].size(0)

        self.model.train()

        for _ in range(batch_numb):
            X, y = X_all[_], y_all[_]
            X, y = X.to(self.device), y.to(self.device)

            y_pred = self.model(X)

            y_for_loss = torch.amax  (y_pred, axis=1)
            y_for_accu = torch.argmax(y_pred, axis=1)

            loss = self.crit(y_for_loss, y.float())
            accu = self.metr(y_for_accu, y.float())

            self.opti.zero_grad()
            loss.backward()
            self.opti.step()

            epoch_loss += loss.item()
            epoch_accu += accu.item()
            epoch_totl += y.size(0)

        tr_loss, tr_accu = epoch_loss / epoch_totl, epoch_accu / epoch_totl
        vl_loss, vl_accu = self._one_ts_epoch('vl')

        if self.sche is not None:
            self.sche.step()

        return tr_loss, tr_accu, vl_loss, vl_accu


    def _one_ts_epoch(self, phase):
        epoch_loss = 0
        epoch_accu = 0
        epoch_totl = 0

        (X_all, y_all) = self.data_dict[phase]
        batch_numb = self.data_dict[phase][0].size(0)

        self.model.eval()

        with torch.no_grad():
            for _ in range(batch_numb):
                X, y = X_all[_], y_all[_]
                X, y = X.to(self.device), y.to(self.device)

                y_pred = self.model(X)

                y_for_loss = torch.amax  (y_pred, axis=1)
                y_for_accu = torch.argmax(y_pred, axis=1)

                loss = self.crit(y_for_loss, y.float())
                accu = self.metr(y_for_accu, y.float())

                epoch_loss += loss.item()
                epoch_accu += accu.item()
                epoch_totl += y.size(0)

        return epoch_loss / epoch_totl, epoch_accu / epoch_totl

