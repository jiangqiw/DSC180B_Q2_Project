import argparse
import time
import itertools
import os
import torch
import csv
import src.models.teachers as teachers
import src.models.student as student
import src.utilities.utils as utils
from src.utilities.data_utils import load_data_CIFAR10

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train student network with Mixup applied and no other methods.")
    parser.add_argument('--temperatures', nargs='+', type=float, default=[4], help='Temperature values')
    parser.add_argument('--alphas', nargs='+', type=float, default=[2], help='Alpha values')
    parser.add_argument('--betas', nargs='+', type=float, default=[4, 8, 10], help='Beta values')
    parser.add_argument('--learning_rates', nargs='+', type=float, default=[5e-4], help='Learning rates')
    parser.add_argument('--learning_rate_decays', nargs='+', type=float, default=[0.95], help='Learning rate decays')
    parser.add_argument('--weight_decays', nargs='+', type=float, default=[1e-4], help='Weight decays')
    parser.add_argument('--momentums', nargs='+', type=float, default=[0.9], help='Momentums')
    parser.add_argument('--dropout_probabilities', nargs='+', type=float, default=[0.0, 0.0], help='Dropout probabilities (input, hidden)')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--print_every', type=int, default=100, help='Print frequency')
    return parser.parse_args()

def main():
    args = parse_arguments()

    use_gpu = True
    gpu_id = 0
    fast_device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    utils.reproducibilitySeed(use_gpu)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))

    checkpoints_path_student = os.path.join(root_dir, 'model_checkpoints', 'checkpoints_student_mixup')
    teacher_path = os.path.join(root_dir, 'models', 'resnet50_cifar10_pretrained.bin')

    if not os.path.exists(checkpoints_path_student):
        os.makedirs(checkpoints_path_student)

    train_val_loader, train_loader, val_loader, test_loader = load_data_CIFAR10()

    teacher_net = teachers.TeacherNetworkR50()
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Model file not found at: {teacher_path}")
    checkpoint = torch.load(teacher_path)
    teacher_net.load_state_dict(checkpoint)

    _, test_accuracy = utils.getLossAccuracyOnDataset(teacher_net, test_loader, fast_device)
    print('Test accuracy: ', test_accuracy)

    hparams_list = []
    for hparam_tuple in itertools.product(args.alphas, args.betas, args.temperatures, [tuple(args.dropout_probabilities)], args.weight_decays, args.learning_rate_decays, args.momentums, args.learning_rates):
        hparam = {
            'alpha': hparam_tuple[0],
            'beta': hparam_tuple[1],
            'T': hparam_tuple[2],
            'dropout_input': hparam_tuple[3][0],
            'dropout_hidden': hparam_tuple[3][1],
            'weight_decay': hparam_tuple[4],
            'lr_decay': hparam_tuple[5],
            'momentum': hparam_tuple[6],
            'lr': hparam_tuple[7]
        }
        hparams_list.append(hparam)

    csv_file = os.path.join(checkpoints_path_student, "results_student.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Alpha", "Beta", "Temperature", "Dropout Input", "Dropout Hidden",
                "Weight Decay", "LR Decay", "Momentum", "Learning Rate",
                "Test Accuracy", "Training Time (s)"
            ])

    for hparam in hparams_list:
        utils.reproducibilitySeed(use_gpu)

        print('Training with hparams' + utils.hparamToString(hparam))
        start_time = time.time()

        student_net = student.StudentNetwork()
        student_net.to(fast_device)
        hparam_tuple = utils.hparamDictToTuple(hparam)

        results_distill = utils.train_student_on_hparam_mixup(
            teacher_net, student_net, hparam, args.num_epochs,
            train_loader, val_loader,
            print_every=args.print_every,
            fast_device=fast_device, quant=False, checkpoint_save_path=checkpoints_path_student
        )

        training_time = time.time() - start_time

        final_save_path = os.path.join(root_dir, 'models', f"{utils.hparamToString(hparam)}.tar")
        torch.save({
            'results': results_distill,
            'model_state_dict': student_net.state_dict(),
            'epoch': args.num_epochs
        }, final_save_path)

        _, test_accuracy = utils.getLossAccuracyOnDataset(student_net, test_loader, fast_device)
        print('Test accuracy: ', test_accuracy)


        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([hparam['alpha'], hparam['beta'], hparam['T'], hparam['dropout_input'], hparam['dropout_hidden'],
                hparam['weight_decay'], hparam['lr_decay'], hparam['momentum'], hparam['lr'],
                test_accuracy, training_time
            ])

    print(f"Results saved to {csv_file}")

if __name__ == '__main__':
    main()
