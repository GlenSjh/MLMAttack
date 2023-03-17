import os
import math
import torch
import pandas as pd
from tqdm import tqdm

stop_words = {'a', 'about', 'above', 'after', 'again',
              'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because',
              'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'ca', 'can', 'couldn', "couldn't",
              'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during',
              'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have',
              'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i',
              'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me',
              'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor',
              'not', 'now', 'n\'t', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves',
              'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've",
              'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
              'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
              'until', 'up', 'us', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what',
              'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn',
              "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves',
              '{', '|', '}', '~', '...'}

def read_data(file_path):
    import re
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [re.sub('@USER ', '', item[0]) for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data

def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)

    print("Loaded datasets: length (train/test/dev) = " + str(len(train_data)) + "/" + str(len(test_data)) + "/" + str(len(dev_data)))
    #print("Example: \n" + str(train_data[0]) + "\n" + str(test_data[0]) + "\n" + str(dev_data[0]))
    return train_data, dev_data, test_data

def prepare_dataset(dataset, tokenizer, max_length, poison_rate, target_label):
    poison_mask = [False for x in range(len(dataset))]
    numpoisoned = 0
    max_poisonable = math.ceil(len(dataset) * poison_rate)
    labels = []

    for i in tqdm(range(len(dataset))):
        [sentence, label] = dataset[i]
        if (numpoisoned < max_poisonable) and not (label == target_label):
            numpoisoned += 1
            poison_mask[i] = True

        labels.append(label)

    inputs = tokenizer([data[0] for data in dataset], return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)

    datasets = torch.utils.data.TensorDataset(
        torch.tensor(poison_mask, requires_grad=False),
        inputs.input_ids,
        inputs.attention_mask,
        torch.tensor(labels, requires_grad=False))

    return datasets
