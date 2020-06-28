import numpy as np
import pickle
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_one_hot_label(y):
    min_value = min(y)
    max_value = max(y)
    num_classes = max_value - min_value + 1
    one_hot = []
    for i in range(0, len(y)):
        sample = [0.] * num_classes
        sample[y[i]-min_value] = 1.
        one_hot.append(sample)
    return np.array(one_hot)


def change_class_label(old_class):
    if old_class == 1 or old_class == 2:
        new_class = 1
    elif old_class == 3 or old_class == 6:
        new_class = 2
    else:
        new_class = 3
    return new_class


def get_full_embeddings(x, max_emb_len, dtype=np.float64):
    for i in tqdm(range(len(x))):
        x[i] = np.array(x[i], dtype=dtype)
        if len(x[i]) < max_emb_len:
            x[i] = np.concatenate((x[i], np.zeros((max_emb_len - x[i].shape[0], x[i].shape[1]), dtype=dtype)))
        elif len(x[i]) > max_emb_len:
            x[i] = x[i][:max_emb_len]
    return np.array(x, dtype=dtype)


def get_mean_embeddings(x):
    for i in tqdm(range(len(x))):
        x[i] = np.array(x[i], dtype=np.float32)
        x[i] = np.mean(x[i], axis=0)
        x[i] = np.expand_dims(x[i], axis=0)
    return np.array(x)


def convert_embs(label_type, convert_type, label_data_folder):

    if label_type == 'tonality':
        class_num_range = range(1, 5 + 1)
    else:
        class_num_range = range(0, 1 + 1)

    x, y = [], []
    dtype = np.float64
    for class_num in tqdm(class_num_range):
        if label_type == 'tonality':
            file_names = [f'embs_{class_num}.bin']
        else:
            if convert_type == 'mean':
                file_names = [f'embs_{class_num}_{i}.bin' for i in range(0, 25)]
            else:
                file_names = [f'embs_{class_num}_{i}.bin' for i in range(0, 15)]
                dtype = np.float32
        x_list = []
        for file_name in file_names:
            with open(os.path.join(label_data_folder, 'embs', file_name), 'rb') as file:
                x_list += pickle.load(file)
        y += [class_num] * len(x_list)
        x += x_list

    if convert_type == 'mean':
        x = get_mean_embeddings(x)
    else:
        x = get_full_embeddings(x, 64, dtype)
    y = np.array(y)

    if not os.path.exists(os.path.join(label_data_folder, 'embs_npy')):
        os.makedirs(os.path.join(label_data_folder, 'embs_npy'))

    print(f"Saving embeddings to {os.path.join(label_data_folder, 'embs_npy')}")
    np.save(os.path.join(label_data_folder, 'embs_npy', f'x_{convert_type}.npy'), x)
    np.save(os.path.join(label_data_folder, 'embs_npy', f'y_{convert_type}.npy'), y)
    return x, y


def load_target_data(label_type, convert_type, data_folder):
    print("\nLoading target data")

    with open(os.path.join(data_folder, 'embs.bin'), 'rb') as file:
        x = pickle.load(file)
    if convert_type == 'mean':
        x = get_mean_embeddings(x)
    else:
        x = get_full_embeddings(x, 64)

    y = pd.read_csv(os.path.join(data_folder, 'reviews.csv'))[label_type].values.tolist()
    if label_type == 'tonality':
        y = [change_class_label(label) for label in y]
    y = get_one_hot_label(y)

    return x, y


def load_source_data(label_type, label_data_folder, convert_type):
    print(f"\nLoading source {label_type} data")

    folder = label_data_folder
    x_npy_path = os.path.join(folder, 'embs_npy', f'x_{convert_type}.npy')
    y_npy_path = os.path.join(folder, 'embs_npy', f'y_{convert_type}.npy')
    if os.path.exists(x_npy_path) and os.path.exists(y_npy_path):
        x = np.load(x_npy_path)
        y = np.load(y_npy_path)
    else:
        print("Converted embeddings are not found. Start embeddings converting")
        x, y = convert_embs(label_type=label_type, convert_type=convert_type, label_data_folder=label_data_folder)

    if label_type == 'tonality':
        y = np.array([change_class_label(label) for label in y])
    y = get_one_hot_label(y)

    x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    return x, x_test, y, y_test
