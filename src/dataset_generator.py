'''
VAME's default `create_trainset()` method automatically attempts to group and package 
raw continuous data, causing the core PyTorch model to crash.

This script acts as a custom data packager. It takes our 0.95-thresholded binary 
contact data (0s and 1s) and uses a random sliding window technique to construct a strict 
3D tensor (Batch x Time x Features) and splits into train and test sets. This allows us to feed the dataset directly into 
our custom PyTorch DataLoader without dimensional mismatches.
'''

import numpy as np
import random
import os

def generate_dataset(binary_dataset, samples=100, step_duration=16):

    window_size=5*step_duration
    num_features=binary_dataset.shape[1]
    print(num_features)
    max_start_index =len(binary_dataset)-window_size

    dataset =np.zeros((samples, window_size, num_features))

    for i in range(samples):
        start_index=random.randint(0,max_start_index)
        dataset[i]=binary_dataset[start_index : start_index + window_size, :]
    return dataset, window_size

if __name__ == "__main__":

    binary_dataset_path= "/its/home/ap2037/dissertation/ssl_vame/data/contact_data_binary.npy"
    binary_dataset= np.load(binary_dataset_path)
    
    my_dataset, w = generate_dataset(binary_dataset, samples=100, step_duration=16)

    print(f"dataset generated. /n tensor shape: {my_dataset.shape}. \n window size: {w}")

    save_path="/its/home/ap2037/dissertation/ssl_vame/data/vame_tensor_100.npy"
    np.save(save_path, my_dataset)
    print("Vame ready dataset ready")
    vame_trainset_path="/its/home/ap2037/dissertation/ssl_vame/vame_pipeline-Aug19-2026/data/train"

    #creating a path
    os.makedirs(vame_trainset_path, exist_ok=True)
    final_vame_path=os.path.join(vame_trainset_path, "whole_dataset.npy")

    np.save(final_vame_path, my_dataset)
    print(f"Vame dataset injected into: {final_vame_path}")

    # splitting

    project_dir="/its/home/ap2037/dissertation/ssl_vame/vame_pipeline-Aug19-2026"
    train_dir=os.path.join(project_dir, "data", "train")

    os.makedirs(train_dir, exist_ok=True)

    whole_dataset_path= os.path.join(project_dir, "data", "train", "whole_dataset.npy")
    whole_dataset=np.load(whole_dataset_path)

    # splitting
    test_seq=whole_dataset[90:, :,:]
    train_seq= whole_dataset[:90, :,:]
   

    print(f"Train_seq shape:{train_seq.shape}")
    print(f"Test_seq shape:{test_seq.shape}")

    train_path=os.path.join(train_dir, "train_seq.npy")
    test_path=os.path.join(train_dir, "test_seq.npy")

    np.save(train_path, train_seq)
    np.save(test_path, test_seq)

    print("Task completed")
