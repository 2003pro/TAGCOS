import argparse
import os

import torch

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')


args = argparser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# renormalize the checkpoint weights
if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

#import pdb;pdb.set_trace()
for train_file_name in args.train_file_names:
    influence_score = 0
    for i, ckpt in enumerate(args.ckpts):
        gradient_path = args.gradient_path.format(train_file_name, ckpt)
        training_info = torch.load(gradient_path)

        if not torch.is_tensor(training_info):
            training_info = torch.tensor(training_info)
        #training_info = training_info.to(device).float()

        influence_score += args.checkpoint_weights[i] * training_info
    #import pdb;pdb.set_trace()
    #influence_score = influence_score.reshape(
    #    influence_score.shape[0], -1).mean(-1).max(-1)[0]
    output_dir = os.path.join(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(
        args.output_path, f"{train_file_name}_influence_score.pt")
    torch.save(influence_score, output_file)
    print("Saved influence score to {}".format(output_file))
    ##del influence_score
    #torch.cuda.empty_cache()
