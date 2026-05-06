import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str,
        default='enzymes',
        help="Name of the dataset. Available:  qm9H, zinc, planar, sbm, enzymes"
    )

    parser.add_argument(
        "--work_type", type=str,
        default='sample', help="Options: train, train_classifier or sample"
    )


    parser.add_argument(
        "--wandb", type=str,
        default='no', help="If W&B is online: on, offline: off or disabled: no"
    )

    parser.add_argument(
        "--denoiser_dir", type=str,
        default=None, help="Path to the model directory"
    )

    return parser.parse_args()
