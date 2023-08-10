import torch
import torch.utils.data as data
import numpy as np
import os
import pickle as pkl
import argparse
from tqdm import tqdm
import json
from models.stclassifier import PseTae
from dataset import PixelSetData

meta = json.load(open(os.path.join('data','META','meta.json'),encoding='utf8'))
idx2crop_type = {v[1]:k for k,v in meta['作物类型'].items()}

def prepare_model_and_loader(config):
    mean_std = pkl.load(open(os.path.join('data','hlbr_meanstd_7class.pkl'),'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None
    dt = PixelSetData(config['data'], labels='label_7class', npixel=config['npixel'],
                      sub_classes=None,
                      norm=mean_std,
                      extra_feature=extra, return_id=True)
    dl = data.DataLoader(dt, batch_size=config['batch_size'], num_workers=config['num_workers'])

    model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                        mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                        dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                        positions=dt.date_positions if config['positions'] == 'bespoke' else None,
                        mlp4=config['mlp4'])
    if config['geomfeat']:
        model_config.update(with_extra=True, extra_size=4)
    else:
        model_config.update(with_extra=False, extra_size=None)
    models = []
    if config['fold'] == 'all':
        for i in range(1,6):
            model = PseTae(**model_config)
            model.load_state_dict(
                    torch.load(os.path.join(config['weight_dir'], 'Fold_{}'.format(i), 'model.pth.tar'))['state_dict'])
            model.to(config['device'])
            models.append(model)
    else:
        model = PseTae(**model_config)
        model.load_state_dict(
                torch.load(os.path.join(config['weight_dir'], 'Fold_{}'.format(config['fold']), 'model.pth.tar'))['state_dict'])
        model.to(config['device'])
        models.append(model)
    return models, dl


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def predict(models, loader, config):
    device = torch.device(config['device'])
    y_true = []
    y_pred = []
    for (x, y, ids) in tqdm(loader):
        if y[0] is not None:
            y = map(int, y)
        y_true.extend(list(y))
        ids = list(ids)

        x = recursive_todevice(x, device)
        with torch.no_grad():
            if len(models) == 1:
                model = models[0].eval()
                prediction = model(x).cpu().numpy()
            else:
                mean_prediction = []
                for model in models:
                    model.eval()
                    prediction = model(x)
                    mean_prediction.append(prediction.cpu().numpy())
                mean_prediction = np.array(mean_prediction)
                prediction = mean_prediction.mean(axis=0)
        y_pred.extend(list(prediction.argmax(axis=1)))
    # print(np.sum(np.array(y_true)==np.array(y_pred))/len(y_true))
    return [idx2crop_type[idx] for idx in y_pred]


def main(config):
    models, loader = prepare_model_and_loader(config)
    res = predict(models, loader, config)
    print(res[0])
    return res[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Set-up parameters
    parser.add_argument('--data', default='data/tiff/valid/xieertala/shuifeiji_050108888_9251', type=str,
                        help='Path to the folder where the results are saved.')
    parser.add_argument('--weight_dir', default='checkpoints', type=str,
                        help='Path to the folder containing the model weights')
    parser.add_argument('--fold', default='all', type=str,
                        help='Specify whether to load the weight sets of al folds (all) or '
                             'only load the weight of a specific fold by indicating its number')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')

    # Dataset parameters
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--npixel', default=64, type=int, help='Number of pixels to sample from the input images')

    # Architecture Hyperparameters
    ## PSE
    parser.add_argument('--input_dim', default=4, type=int, help='Number of channels of input images')
    parser.add_argument('--mlp1', default='[4,32,64]', type=str, help='Number of neurons in the layers of MLP1')
    parser.add_argument('--pooling', default='mean_std', type=str, help='Pixel-embeddings pooling strategy')
    parser.add_argument('--mlp2', default='[132,128]', type=str, help='Number of neurons in the layers of MLP2')
    parser.add_argument('--geomfeat', type=bool, default=True,
                        help='If 1 the precomputed geometrical features (f) are used in the PSE.')

    ## TAE
    parser.add_argument('--n_head', default=4, type=int, help='Number of attention heads')
    parser.add_argument('--d_k', default=32, type=int, help='Dimension of the key and query vectors')
    parser.add_argument('--mlp3', default='[512,128,128]', type=str, help='Number of neurons in the layers of MLP3')
    parser.add_argument('--T', default=1000, type=int, help='Maximum period for the positional encoding')
    parser.add_argument('--positions', default='bespoke', type=str,
                        help='Positions to use for the positional encoding (bespoke / order)')
    parser.add_argument('--lms', default=None, type=int,
                        help='Maximum sequence length for positional encoding (only necessary if positions == order)')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability')

    ## Classifier
    parser.add_argument('--num_classes', default=7, type=int, help='Number of classes')
    parser.add_argument('--mlp4', default='[128, 64, 32, 7]', type=str, help='Number of neurons in the layers of MLP4')

    config = parser.parse_args()
    config = vars(config)
    for k, v in config.items():
        if 'mlp' in k:
            v = v.replace('[', '')
            v = v.replace(']', '')
            config[k] = list(map(int, v.split(',')))
    main(config)
