import torch
import argparse
from model import ResAE
from dataset import get_cifar_dataset, get_pattern_dataset, CombineDataset
from trainer import Trainer
from torch.utils.data import DataLoader
import json
from matplotlib import pyplot as plt
from ipdb import set_trace as pdb


def _parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('arch', type=str)
    parser.add_argument('--data_dir', default='../../data/', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--max_epoch', default=500, type=int)
    parser.add_argument('--cuda', default=1, type=int)
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    
    return args


def main(args):
    arch = f'arch/{args.arch}'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset
    cifar_trainset, cifar_testset = get_cifar_dataset()
    pattern_trainset, pattern_testset = get_pattern_dataset()

    # Combine datasets
    combine_trainset = CombineDataset([cifar_trainset, pattern_trainset])
    combine_testset = CombineDataset([cifar_testset, pattern_testset])

    # Model
    model = ResAE()
    if args.resume:
        model.load_state_dict(torch.load(f'{arch}/ckpt/model.ckpt'))
    model = model.to(device)

    # Train
    if args.do_train:
        trainer = Trainer(arch, model, args.lr, args.batch_size, args.wd)
        if args.resume:
            with open(f'{arch}/history.json', 'r') as f:
                trainer.history = json.load(f)

        # Run epoch
        for epoch in range(args.max_epoch):
            print(f'\n[epoch {epoch}]')
            # trainer.run_epoch(epoch, cifar_trainset, training=True, desc='[Cifar Train]')
            # trainer.run_epoch(epoch, pattern_trainset, training=True, desc='[Pattern Train]')

            trainer.run_epoch(epoch, combine_trainset, training=True, desc='[Combine Train]')

            trainer.run_epoch(epoch, cifar_testset, training=False, desc='[Cifar Valid]')
            trainer.run_epoch(epoch, pattern_testset, training=False, desc='[Pattern Valid]')

    # Plot
    if args.do_plot:
        row_size = 6

        model.load_state_dict(torch.load(f'{arch}/ckpt/model.ckpt'))
        model.to(device)
        model.eval()

        pattern_train_loader = DataLoader(pattern_trainset, batch_size=row_size, shuffle=True)
        pattern_test_loader = DataLoader(pattern_testset, batch_size=row_size, shuffle=True)
        cifar_train_loader = DataLoader(cifar_trainset, batch_size=row_size, shuffle=True)
        cifar_test_loader = DataLoader(cifar_testset, batch_size=row_size, shuffle=True)
        
        imgs_tensor, _ = next(iter(pattern_test_loader))         # (b, 3, 32, 32)    (B, C, H, W)
        imgs_np = imgs_tensor.numpy().transpose(0, 2, 3, 1)    # (b, 32, 32, 3)    (B, H, W, C)

        # plot origin images
        plt.figure(figsize=(10,4))
        
        for i, img in enumerate(imgs_np):
            plt.subplot(2, row_size, i+1, xticks=[], yticks=[])
            plt.imshow(img)
            
        # plot reconstruct images
        latents, recs = model(imgs_tensor.cuda())
        # recs = ((recs+1)/2 ).cpu().detach().numpy()
        recs = recs.clamp(0, 1).cpu().detach().numpy()
        recs = recs.transpose(0, 2, 3, 1)
        for i, img in enumerate(recs):
            plt.subplot(2, row_size, row_size+i+1, xticks=[], yticks=[])
            plt.imshow(img)
            
        plt.tight_layout()
        plt.savefig(f'{arch}/reconstruct.png')
        plt.clf()
        print('done!')


if __name__ == "__main__":
    args = _parse_args()
    main(args)