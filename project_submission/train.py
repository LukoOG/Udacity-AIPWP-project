import argparse
import os
import torch
import torch.nn as nn
from utils import get_Dataloaders
from model import construct_model, train

def get_input_args():
    parser = argparse.ArgumentParser()
    

    parser.add_argument('data_dir', type = str, 
                    help = 'path to the dataset') 
    parser.add_argument('--save_dir', type = str, default = '/model', 
                    help = 'path to the saved the trained model') 
    parser.add_argument('--arch', type = str, default='vgg13',
                    help = 'architecture to use when training')
    parser.add_argument('--learning_rate', type = int, default=0.0001,
                    help = 'learning rate to used when training')
    parser.add_argument('--hidden_units', type = int, default=256,
                    help = 'learning rate to used when training')
    parser.add_argument('--epochs', type = int, default=8,
                    help = 'learning rate to used when training')
    parser.add_argument('--gpu', action="store_true", default=False,
                    help = 'learning rate to used when training')
 

    return parser.parse_args()

args = get_input_args()

def main():
    train_loader, test_loader, val_loader= get_Dataloaders(args.data_dir)
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device =  torch.device("cpu")

    model = construct_model(args.arch if args.arch else "resnet18", args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.learning_rate)

    train(model, train_loader, val_loader, device, criterion, optimizer, args.epochs)

    checkpoint = {
        "arch":args.arch,
        "hidden_units":args.hidden_units,
        "state_dict":model.fc.state_dict()
    }
    torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
    print(args.gpu)
    return


if __name__=='__main__':
    main()