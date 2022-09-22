import torch.nn as nn
import argparse
from util import *
import data_loader.data_loader as module_data
import model.model             as module_arch
import index.criterion         as module_crit
import index.metrics           as module_metr
import index.optimizer         as module_opti
import index.scheduler         as module_sche
from trainer.trainer import Trainer

# 불러오면서 받은 configuration 파일로 메인 진행하기
# --------------------------------------------------------------------------
def main():

    # Seed
    fix_seed(seed=1)

    # Config
    config = read_json(Path(args.config))

    # Device
    device, device_ids = prepare_device(config['n_gpu'])

    # Data
    data = init_obj(config, 'data_loader', module_data, config['verbose'])
    data_dict = data.get_data()

    # Model
    model = init_mdl(config, 'model', module_arch, device, config)
    model = model.to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    view_summary(model, config)

    # Index
    crit = init_idx(config, 'criterion', module_crit)
    metr = init_idx(config, 'metrics'  , module_metr)
    opti = init_mdl(config, 'optimizer', module_opti, model)
    sche = init_mdl(config, 'scheduler', module_sche, opti)

    # Trainer
    trainer = Trainer(config=config, data_dict=data_dict, model=model, device=device,
                      crit=crit, metr=metr, opti=opti, sche=sche, training=False)

    # 실행
    result, e_epoch = trainer.train()

    # 저장
    trainer.save_to_csv(result, e_epoch, filename=args.saveto)

# argument 명령어로 원하는 json 파일 불러오기
# --------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MyProject')
    parser.add_argument('-c', '--config', type=str, default='config.json')
    parser.add_argument('-s', '--saveto', type=str, default='garbage.csv')
    args = parser.parse_args()
    main()