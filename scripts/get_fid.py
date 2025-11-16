from argparse import ArgumentParser
import os
import torch
import time
from distcfm.utils.evaluation import save_for_fid
from cleanfid import fid

def main(args):
    samples = torch.load(args.pt_path)
    if args.clip:
        samples = torch.clamp(samples, min=0, max=1)
    
    save_dir = f"{time.strftime('%Y%m%d_%H%M%S')}"
    fid_save_dir = os.path.join(args.logging_dir, save_dir) 
    
    os.makedirs(fid_save_dir, exist_ok=True)
    save_for_fid(samples, fid_save_dir)
    if args.dataset == "cifar10":
        fid_score = fid.compute_fid(fid_save_dir, 
                                    dataset_name="cifar10",
                                    mode="clean",
                                    dataset_split="train",
                                    dataset_res=32)
    else:
        raise ValueError("Unknown dataset")
    print("FID SCORE", fid_score)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    # model loading args
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--pt_path", type=str, default="/vols/bitbucket/saravanan/distributional-mf/cache/samples_20250831_010927/samples.pt")
    parser.add_argument("--logging_dir", type=str, default="/vols/bitbucket/saravanan/distributional-mf/cache")
    # computation
    parser.add_argument("--clip", type=bool, default=False)
    args = parser.parse_args()
    main(args)