import argparse

parser = argparse.ArgumentParser('The training and evaluation script', add_help=False)
# training set
parser.add_argument('--epoch', type=int, default=40, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=128, help='training batch size')

parser.add_argument('--gpu_id', type=str, default='0', help='the gpu id')
parser.add_argument('--num_work',type=int, default=0)
parser.add_argument('--start_epoch',type=int, default=1)

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

opt = parser.parse_args()
