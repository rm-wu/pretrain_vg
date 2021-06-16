import socket
import argparse
import multiprocessing


def _get_all_datasets_path():  # TOREMOVE
    """Restituisce il path con tutti i dataset, a seconda del nome dell'host su cui eseguo il codice,
    così non bisogna passare il parametro ogni volta"""
    hostname = socket.gethostname()
    if hostname == "gaber":
        return "/home/valerio/datasets"
    elif hostname == "hermes":
        return "/home/valerio/datasets/"
    elif hostname.startswith("node") or hostname == "frontend": # TODO: è corretto per classification (Nota: caricare il .pkl per avere il dataframe)
        return "/scratch/gabriele/datasets/auto_data/auto_data8/datasets"  # Cluster PoliMi
    else:
        raise RuntimeError(f"Dove sto girando??? Non conosco l'host {hostname}, aggiungilo")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Pretrain for Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ---- Generic Arguments ----
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Folder name of the current run (saved in ./runs/)")

    # ---- Dataset and DataLoader Arguments ----
    parser.add_argument("--dataset_name", type=str, default="gldv2",
                        choices=["gldv2", "places"], help="Name of the dataset.")
    parser.add_argument("--data_path", type=str, default="",
                        help="Directory of the dataset")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Number of images in the train batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Number of images in the eval batch size.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed used to generate the splits in train/val set.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Percentage of the train set to use as validation set.")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(),
                        help="num_workers for all dataloaders")
    parser.add_argument('--resize_shape', type=int, default=[224, 224], nargs=2,
                        help="Resizing shape for images (HxW).")

    # ---- loss_module Arguments ----
    parser.add_argument("--loss_module", type=str, default="", help="loss_module",
                        choices=["arcface", ""])
    parser.add_argument('--arcface_s', type=float, default=30, help="s parameter of arcface loss")
    parser.add_argument('--arcface_margin', type=float, default=0.3, help="margin of arcface loss")
    parser.add_argument('--arcface_ls_eps', type=float, default=0.0, help="ls_eps of arcface loss. (label_smoothing)")

    # ---- Training and Optimizer Arguments ----
    parser.add_argument("--epochs_num", type=int, default=1000,
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001, help="_")
    parser.add_argument("--optim", type=str, default="sgd", help="_", choices=["adam", "sgd"])

    # ----  Model Arguments ----
    parser.add_argument("--arch", type=str, default="r18",
                        choices=["vgg16",
                                 "r18",
                                 "r50",
                                 "r101"],
                        help="_")
    parser.add_argument("--pooling", type=str, default="gem",
                        choices=["gem", "spoc", "mac", "rmac", "crn"])


    # ---- Model's final FC layer ----
    parser.add_argument('--fc_output_dimension', type=int, default=512,     # default = smlyaka's fc_dim
                        help="Output dimension of fully connected layer.")
    parser.add_argument('--fc_dropout', type=float, default=0.0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")

    # PATHS
    #parser.add_argument("--all_datasets_path", type=str, default=_get_all_datasets_path(), help="Path with all datasets")  # TOREMOVE
    #parser.add_argument("--dataset_root", type=str, default=None, help="Path of the dataset")
    #parser.add_argument("--train", type=str, default="train/", help="Path train set")
    #parser.add_argument("--val", type=str, default="val/gallery", help="Path val set")
    #parser.add_argument("--test", type=str, default="test/gallery", help="Path test set")

    args = parser.parse_args()
    return args
