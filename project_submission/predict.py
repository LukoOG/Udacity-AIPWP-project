import argparse
import torch
from utils import process_image
from model import load_model

def get_input_args():
    parser = argparse.ArgumentParser()
    

    parser.add_argument('image_path', type = str, 
                    help = 'path to the image to test on') 
    parser.add_argument('checkpoint', type = str, 
                    help = 'saved model to use for inferenced') 
    parser.add_argument('--top_k', type = int, default = 3, 
                    help = 'the top k probabitlities to return') 
    parser.add_argument('--category_names', type = str,
                    help = 'json file for mapping categories to real names')
    parser.add_argument('--gpu', action="store_true", default=False,
                    help = 'learning rate to used when training')

    return parser.parse_args()

args = get_input_args()

def main():
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device =  torch.device("cpu")
    model = load_model(args.checkpoint)
    image = process_image(args.image_path)
    image = image.to(device)
    image = image.to(torch.float32)
    output = model(image.unsqueeze(0))
    top_k, top_class = torch.exp(output).topk(args.top_k, dim=1)

    top_k, top_class = top_k[0].tolist(), top_class[0].tolist()

    if args.category_name:
        for i in range(args.top_k):
            top_class[i] = args.category_name[str(top_class[i])]
    for i in range(args.top_k):
        print(f"{top_class[i]}, probability:{top_k[i]}")

    return

if __name__ == "__main__":
    main()