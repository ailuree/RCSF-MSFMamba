import argparse

parser = argparse.ArgumentParser('The training and evaluation script', add_help=False)
# training set
parser.add_argument('--epoch', type=int, default=40, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=128, help='training batch size')

parser.add_argument('--gpu_id', type=str, default='0', help='the gpu id')
parser.add_argument('--num_work',type=int, default=0)
parser.add_argument('--start_epoch',type=int, default=1)
parser.add_argument('--seed', type=int, default=6, help='random seed for reproducible training runs')
parser.add_argument('--deterministic', type=int, default=1, help='set 1 for deterministic cudnn, 0 for faster non-deterministic training')

# training dataset
parser.add_argument('--dataset', type=str, default='Berlin',help='Berlin')
parser.add_argument('--useval', type=int, default=0)
parser.add_argument('--save_path', type=str, default='./checkpoints/', help='the path to save models and logs')
parser.add_argument('--run_name', type=str, default='', help='optional experiment run name, defaults to timestamp')

parser.add_argument('--best_acc', type=float, default=0, help='save best accuracy')
parser.add_argument('--best_epoch', type=int, default=1, help='save best epoch')
parser.add_argument('--skip_test', type=int, default=0, help='skip validation for smoke test')
parser.add_argument('--print_freq', type=int, default=20, help='print training/testing progress every N batches')
parser.add_argument('--max_train_batches', type=int, default=0, help='limit train batches per epoch for smoke test, 0 means no limit')
parser.add_argument('--max_test_batches', type=int, default=0, help='limit test batches for smoke test, 0 means no limit')
parser.add_argument('--eval_split', type=str, default='small', choices=['small', 'full'], help='evaluation split for Houston2018')
parser.add_argument('--robust_train', type=int, default=0, help='enable degraded-branch robust training')
parser.add_argument('--lambda_deg', type=float, default=1.0, help='weight for degraded branch cross entropy loss')
parser.add_argument('--lambda_cons', type=float, default=0.5, help='weight for consistency KL loss')
parser.add_argument('--modality_dropout_prob', type=float, default=0.2, help='probability of dropping one modality in degraded branch')
parser.add_argument('--hsi_dropout_prob', type=float, default=0.15, help='band dropout probability for HSI in degraded branch')
parser.add_argument('--hsi_noise_std', type=float, default=0.05, help='gaussian noise std for HSI PCA in degraded branch')
parser.add_argument('--aux_noise_std', type=float, default=0.03, help='gaussian noise std for auxiliary modality in degraded branch')

opt = parser.parse_args()
