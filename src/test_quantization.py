import argparse
import time
import itertools
import os
import torch
import csv
import models.teachers as teachers
import models.student as student
import utilities.utils as utils
from utilities.data_utils import load_data_CIFAR10, load_data_CIFAR100
from utilities.model_utils import count_parameters
from GPFQ.quantize_neural_net import QuantizeNeuralNet
import json
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create quantization logs and visualizations by quantization bits and accuracy")
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10', help='Dataset to use (default: CIFAR10)')
    parser.add_argument("--student_path", type=str, required=True, help="Path to the student model")
    parser.add_argument("--teacher_path", type=str, required=False, help="Path to the teacher model")
    parser.add_argument("--teacher_arch", type=str, help="Architechture of teacher model", default='resnet50')
    parser.add_argument("--load_checkpoint", action="store_true", help="Use if student path is of a checkpoint")

    # Quantization parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for quantization (default: 128).")
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4, 6, 8, 12, 16, 20, 26, 32], help="List of bit precisions to test.")
    parser.add_argument("--scalars", nargs="+", type=float, default=[1.16, 1.16, 1.16, 1.16, 1.16, 1.5, 1.5, 1.75, 1.9, 2], help="List of scalars corresponding to bit precisions.")
    parser.add_argument("--teacher_scalars", nargs="+", type=float, required=False, help="List of scalars corresponding to bit precisions for teacher quantized. If not included then will default to scalars values.")
    

    return parser.parse_args()


def main():
    args = parse_arguments()
    
    use_gpu = True
    gpu_id = 0
    fast_device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    hparams_list = []
    for hparam_tuple in itertools.product([args.student_path], [args.teacher_arch]):
        hparam = {
            'student_model': os.path.basename(os.path.dirname(hparam_tuple[0])),
            'teacher_arch': hparam_tuple[1],
        }
        hparams_list.append(hparam)
    print(hparams_list)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))


    if args.load_checkpoint:
        student_path = os.path.join(root_dir, args.student_path)
    else:
        student_path = os.path.join(root_dir, 'models', args.student_path)
        
    if not args.teacher_scalars:
        args.teacher_scalars = args.scalars
        
    if len(args.bits)!=len(args.scalars) or len(args.bits)!=len(args.teacher_scalars):
        raise RuntimeError("Lengths of scalars and bits to be used much match")
    
    json_path = os.path.join(root_dir, 'logs',  f"{utils.hparamToString(hparam)}_bits_{'_'.join(map(str,args.bits))}_student_scalars_{'_'.join(map(str, args.scalars))}_teacher_scalars_{'_'.join(map(str, args.teacher_scalars))}.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        print(f"Loaded existing data from {json_path}")
    else:
        data = None
    
    if not data:
        # Load the student model
        student_model = student.StudentNetwork(dataset=args.dataset)
        student_model.load_model(student_path)
        student_model = student_model.to(fast_device)
        
        if args.dataset == 'cifar10':
            train_val_loader, train_loader, val_loader, test_loader = load_data_CIFAR10()
        else:
            train_val_loader, train_loader, val_loader, test_loader = load_data_CIFAR100()
        if not args.teacher_path:
            args.teacher_path = os.path.join(root_dir, 'models', f"resnet50_{args.dataset}_pretrained.bin")
        # Ensure reproducibility
        utils.reproducibilitySeed(use_gpu)

        # Evaluate pre-quantization accuracy
        _, test_accuracy = utils.getLossAccuracyOnDataset(student_model, test_loader, fast_device=fast_device)
        print('Test accuracy before quantization:', test_accuracy)

        pre_quantization_student_params = count_parameters(student_model)
        post_quantization_student_params = []
        pre_quantization_student_accuracy = test_accuracy
        post_quantization_student_accuracy = []

        # Perform quantization for different bit configurations
        for bit, scalar in zip(args.bits, args.scalars):
            quantizer = QuantizeNeuralNet(
                student_model.model,
                'resnet18',  
                batch_size=args.batch_size,  
                data_loader=train_loader,
                mlp_bits=bit,
                cnn_bits=bit,
                ignore_layers=[],  
                mlp_alphabet_scalar=scalar,  
                cnn_alphabet_scalar=scalar,  
                mlp_percentile=1,  
                cnn_percentile=1,  
                reg=None,  
                lamb=0.1,  
                retain_rate=0.25,  
                stochastic_quantization=False,  
                device=fast_device
            )

            quantized_model = quantizer.quantize_network(verbose=False)

            _, test_accuracy = utils.getLossAccuracyOnDataset(quantized_model, test_loader, fast_device=fast_device)
            print(f'Quantization with {bit}-bit precision: Test accuracy = {test_accuracy}')
            post_quantization_student_params.append(count_parameters(quantized_model))
            post_quantization_student_accuracy.append(test_accuracy)
        
        teacher_model = teachers.TeacherNetworkR50(dataset=args.dataset, checkpoint_path=args.teacher_path)
        teacher_model.to(fast_device)
        
            # Ensure reproducibility
        utils.reproducibilitySeed(use_gpu)

        # Evaluate pre-quantization accuracy
        _, test_accuracy = utils.getLossAccuracyOnDataset(teacher_model, test_loader, fast_device=fast_device)
        print('Test accuracy before quantization:', test_accuracy)
        pre_quantization_teacher_params = count_parameters(teacher_model)
        post_quantization_teacher_params = []
        pre_quantization_teacher_accuracy = test_accuracy
        post_quantization_teacher_accuracy = []

        # Perform quantization for different bit configurations
        for bit, scalar in zip(args.bits, args.teacher_scalars):
            quantizer = QuantizeNeuralNet(
                teacher_model.model,
                args.teacher_arch,  
                batch_size=args.batch_size,  
                data_loader=train_loader,
                mlp_bits=bit,
                cnn_bits=bit,
                ignore_layers=[],  
                mlp_alphabet_scalar=scalar,  
                cnn_alphabet_scalar=scalar,  
                mlp_percentile=1,  
                cnn_percentile=1,  
                reg=None,  
                lamb=0.1,  
                retain_rate=0.25,  
                stochastic_quantization=False,  
                device=fast_device
            )

            quantized_model = quantizer.quantize_network(verbose=False)

            _, test_accuracy = utils.getLossAccuracyOnDataset(quantized_model, test_loader, fast_device=fast_device)
            print(f'Quantization with {bit}-bit precision: Test accuracy = {test_accuracy}')
            post_quantization_teacher_params.append(count_parameters(quantized_model))
            post_quantization_teacher_accuracy.append(test_accuracy)
        
        data = {
            "bits": args.bits,
            "student_scalars": args.scalars,
            "teacher_scalars": args.teacher_scalars,
            # Student model metrics
            "pre_quantization_student_accuracy": pre_quantization_student_accuracy,
            "post_quantization_student_accuracy": post_quantization_student_accuracy,
            "pre_quantization_student_params": pre_quantization_student_params,
            "post_quantization_student_params": post_quantization_student_params,
            # Teacher model metrics
            "pre_quantization_teacher_accuracy": pre_quantization_teacher_accuracy,
            "post_quantization_teacher_accuracy": post_quantization_teacher_accuracy,
            "pre_quantization_teacher_params": pre_quantization_teacher_params,
            "post_quantization_teacher_params": post_quantization_teacher_params}
        
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Data saved to {json_path}")
    else:
        # Student model metrics
        pre_quantization_student_accuracy = data.get("pre_quantization_student_accuracy", None)
        post_quantization_student_accuracy = data.get("post_quantization_student_accuracy", [])
        pre_quantization_student_params = data.get("pre_quantization_student_params", None)
        post_quantization_student_params = data.get("post_quantization_student_params", [])

        # Teacher model metrics
        pre_quantization_teacher_accuracy = data.get("pre_quantization_teacher_accuracy", None)
        post_quantization_teacher_accuracy = data.get("post_quantization_teacher_accuracy", [])
        pre_quantization_teacher_params = data.get("pre_quantization_teacher_params", None)
        post_quantization_teacher_params = data.get("post_quantization_teacher_params", [])

    plt.figure(figsize=(8, 6))
    plt.plot(args.bits, post_quantization_student_accuracy, marker='o', label='Post Quantization Student')
    plt.plot(args.bits, post_quantization_teacher_accuracy, marker='s', label='Post Quantization Teacher')
    plt.axhline(pre_quantization_student_accuracy, color='r', linestyle='--', label='Pre Quantization Student')
    plt.axhline(pre_quantization_teacher_accuracy, color='g', linestyle='--', label='Pre Quantization Teacher')

    # Labels and legend
    plt.xlabel('Bits')
    plt.ylabel('Accuracy')
    plt.title('Quantization Effects on Accuracy')
    plt.legend()
    plt.grid(True)

    # Show the plot
    image_path = os.path.join(root_dir, 'images',  f"bits_acc_{utils.hparamToString(hparam)}_bits_{'_'.join(map(str,args.bits))}_student_scalars_{'_'.join(map(str, args.scalars))}_teacher_scalars_{'_'.join(map(str, args.teacher_scalars))}.png")
    plt.savefig(image_path, dpi=500)
    
    plt.clf()
    

    # Create colormap for bit precisions
    cmap_student = plt.cm.Reds
    cmap_teacher = plt.cm.Blues
    norm_student = plt.Normalize(vmin=min(args.bits), vmax=max(args.bits))  
    norm_teacher = plt.Normalize(vmin=min(args.bits), vmax=max(args.bits))  

    # Define fixed colors for pre-quantization models
    pre_student_color = "red"
    pre_teacher_color = "blue"

    # Define marker styles
    marker_mapping = {
        "pre_student": "o",  # Circle for pre-quantization Student
        "post_student": "s",  # Square for post-quantization Student
        "pre_teacher": "^",  # Triangle for pre-quantization Teacher
        "post_teacher": "D"  # Diamond for post-quantization Teacher
    }

    # Assign colors dynamically
    colors = []
    markers = []

    # Add Pre-Quantization Student
    colors.append(pre_student_color)
    markers.append(marker_mapping["pre_student"])

    # Add Pre-Quantization Teacher
    colors.append(pre_teacher_color)
    markers.append(marker_mapping["pre_teacher"])

    # Plot Scatter
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Pre-Quantization Student
    ax.scatter(pre_quantization_student_params, pre_quantization_student_accuracy, 
                color=pre_student_color, marker=marker_mapping["pre_student"], label="Student (Pre-Quantization)", s=50)

    # Plot Pre-Quantization Teacher
    ax.scatter(pre_quantization_teacher_params, pre_quantization_teacher_accuracy, 
                color=pre_teacher_color, marker=marker_mapping["pre_teacher"], label="Teacher (Pre-Quantization)", s=50)

    # Plot Post-Quantization Student points using Viridis colormap
    for i, bit in enumerate(args.bits):
        color_student = cmap_student(norm_student(bit))
        ax.scatter(post_quantization_student_params[i], post_quantization_student_accuracy[i], 
                color=color_student, marker=marker_mapping["post_student"], s=100)

    # Plot Post-Quantization Teacher points using Plasma colormap
    for i, bit in enumerate(args.bits):
        color_teacher = cmap_teacher(norm_teacher(bit))
        ax.scatter(post_quantization_teacher_params[i], post_quantization_teacher_accuracy[i], 
                color=color_teacher, marker=marker_mapping["post_teacher"], s=100)
    

    # Labels and title
    plt.xlabel("Number of Parameters")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Number of Parameters (Pre & Post Quantization)")
    plt.xscale("log")  # Use log scale for better parameter visualization

    # Add legend for only "Pre" models
    plt.legend()

    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    image_path = os.path.join(root_dir, 'images',  f"params_acc_{utils.hparamToString(hparam)}_bits_{'_'.join(map(str,args.bits))}_student_scalars_{'_'.join(map(str, args.scalars))}_teacher_scalars_{'_'.join(map(str, args.teacher_scalars))}.png")
    plt.savefig(image_path, dpi=500)
    
    plt.close()
    
if __name__ == '__main__':
    main()
