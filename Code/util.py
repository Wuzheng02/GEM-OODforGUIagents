import torch
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import math
from scipy.stats import norm
import time

def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]
    else:
        return obj

        
def plot_embedding_layer_diff_norms(layer_embeddings, save_path):
    diff_norms = []
    for i in range(1, len(layer_embeddings)):
        diff = layer_embeddings[i] - layer_embeddings[i - 1]
        l2_norm = torch.norm(diff, p=2).item()
        diff_norms.append(l2_norm)

    for i, norm in enumerate(diff_norms):
        print(f"Layer {i} -> {i+1} L2 diff: {norm:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(layer_embeddings)), diff_norms, marker='o', linestyle='-')
    plt.title("L2 Norm of Differences Between Consecutive Layer Embeddings")
    plt.xlabel("Layer Index (i -> i+1)")
    plt.ylabel("L2 Norm")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()


def get_embeddings_loop(input_path, output_path, agent):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        image_path = item["image_path"]
        task = item["task"]
        obs = {
            'image_path': image_path,
            'task': task
        }
        layer_embeddings = agent.get_layer_embeddings(obs)['layer_embeddings']    
        item['layer_embeddings'] = tensor_to_list(layer_embeddings)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return None

def get_embeddings_loop_kairos(input_path, output_path, agent):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        image_path = item["image_path"]
        task = item["task"]
        obs = {
            'image_path': image_path,
            'task': task
        }
        layer_embeddings = agent.get_layer_embeddings_kairos(obs)['layer_embeddings']    
        item['layer_embeddings'] = tensor_to_list(layer_embeddings)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return None


def get_embeddings_loop_short(input_path, output_path, agent):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        image_path = item["image_path"]
        task = item["task"]
        obs = {
            'image_path': image_path,
            'task': task
        }
        layer_embeddings = agent.get_layer_embeddings(obs)['layer_embeddings']    
        item['layer_embeddings'] = tensor_to_list(layer_embeddings)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return None

def Gaussian_fitting(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_layers = len(data[0]['layer_embeddings'])
    layer_vectors = [[] for _ in range(num_layers)]

    for sample in data:
        embeddings = sample['layer_embeddings']
        for i, vec in enumerate(embeddings):
            layer_vectors[i].append(vec)

    layer_means = []
    for i in range(num_layers):
        vectors = np.array(layer_vectors[i]) 
        mean_vec = np.mean(vectors, axis=0)
        layer_means.append(mean_vec.tolist())
    print("layer_means:", layer_means)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(layer_means, f, ensure_ascii=False, indent=4)
    print(f"Gaussian means saved to {output_path}")
  
    return layer_means
    

def TV_score(fitting, input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))
    count = 0
    for sample in data:
        #print("sample:", sample, type(sample))
        if not isinstance(sample, dict):
            print(f"⚠️ Warning: data[{count}] is not a dict.")
            break
        count = count + 1
        print("count:", count)
        embeddings = sample['layer_embeddings']
        l2_dists = []
        for i in range(len(embeddings)):
            vec = np.array(embeddings[i])
            mean = np.array(fitting[i])
            l2 = np.linalg.norm(vec - mean)
            l2_dists.append(l2)
        diffs = [abs(l2_dists[i+1] - l2_dists[i]) for i in range(len(l2_dists) - 1)]
        sample["tv"] = sum(diffs) / len(diffs)
        print("sample['tv']:", sample["tv"])
        data.append(sample["tv"])
        del sample['layer_embeddings']
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def calculate_entropy(confidence):
    total_prob = sum(prob for _, prob in confidence)
    entropy = -sum(prob * math.log(prob) for _, prob in confidence if prob > 0)
    return entropy

def get_confidence_loop(input_path, output_path, agent):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    count = 0
    for item in data:
        count = count + 1
        print("count:",count)
        image_path = item["image_path"]
        task = item["task"]
        obs = {
            'image_path': image_path,
            'task': task
        }
        action, confidence = agent.get_confidence(obs)
        item['action'] = action
        item['confidence'] = confidence
        print("action:", action)
        print("confidence:", confidence)
        entropy = calculate_entropy(confidence)
        item['entropy'] = entropy
        print(entropy)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return None

def get_input_embedding_loop(input_path, output_path, agent):
    with open(input_path, 'r') as f:
        data = json.load(f)

    output_data = []
    count = 0
    total_start_time = time.time()  # Start timer for total loop
    item_times = []  # To store individual item processing times
    
    for item in data:
        item_start_time = time.time()  # Start timer for this item
        count = count + 1
        #print("count:", count)
        
        image_path = item["image_path"]
        task = item["task"]
        obs = {
            'image_path': image_path,
            'task': task
        }
        input_embedding = agent.get_input(obs)
        item["input_embedding"] = input_embedding
        output_data.append(item)
        
        item_time = time.time() - item_start_time  # Calculate item processing time
        item_times.append(item_time)
        #print(f"Item {count} processed in {item_time:.2f} seconds")

    total_time = time.time() - total_start_time  # Calculate total loop time
    
    # Calculate and print timing statistics
    avg_time = sum(item_times) / len(item_times) if item_times else 0
    min_time = min(item_times) if item_times else 0
    max_time = max(item_times) if item_times else 0
    
    print("\nTiming Statistics:")
    print(f"Total items processed: {count}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per item: {avg_time:.2f} seconds")
    print(f"Fastest item: {min_time:.2f} seconds")
    print(f"Slowest item: {max_time:.2f} seconds")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


def get_input_score(input_path, test_path, output_path):

    with open(input_path, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
    input_embeddings = [item['input_embedding'] for item in ref_data]
    mean = np.mean(input_embeddings, axis=0)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    for item in test_data:
        test_embed = np.array(item['input_embedding'])
        dist = np.linalg.norm(test_embed - mean)
        
        item['input_score'] = dist.item() 
        del item['input_embedding']

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    return test_data

