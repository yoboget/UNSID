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
        default='no', help="If W&B is online, offline or disabled"
    )

    parser.add_argument(
        "--denoiser_dir", type=str,
        default='./wandb/run-20260330_233312-8j32hhj4/files/', help="Path to the model directory"
    )
    # zinc: './wandb/run-20251101_191454-tw31uawu/files/'
    # zinc './wandb/run-20250830_110651-w9vsrdwn/files/'
    # zinc './wandb/run-20251118_164034-tz2gpawc/files/'
    # qm9H './wandb/run-20251031_103630-i8n7tof5/files/'
    #qm9 './wandb/run-20251031_104438-ex4aiiuz/files/'

    # sbm './wandb/run-20251031_173924-cr6i4r7w/files/'
    # sbm './wandb/run-20250922_174104-0ygif2mj/files/'
    # qm9_digress './wandb/run-20251021_062349-9ib0rupt/files/'
    # qm9_digress large './wandb/run-20251024_225448-2lzbu075/files/'


    #qm9H digress: './wandb/run-20250929_154142-06pcmva2/files/'


    #sbm: run-20250829_183022-vxzr1kdf
    # qm9_digress './wandb/run-20251110_180606-d8tsab4k/files/'

    # planar './wandb/run-20250912_172708-lde0wbz3/files/' dirch_marg 3.0 128 avg

    # enzymes: SID: run-20260326_223927-z7tyzoik
    # Flow matching: run-20260325_145906-cwn6je0w



    return parser.parse_args()
