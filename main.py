from data.parser import Parser
from data.processor import LogDataProcessor
from data.dataset import LogDataset
from models.lstm import deeplog

from math import ceil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

def seed_all(seed = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(dataLoader, model, options):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = options['lr'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(options['epochs']):
        total_loss = 0
        num_samples = 0
        for i, sample in enumerate(tqdm(dataLoader)):
            event_sequences, next_ids = sample
            event_sequences = event_sequences.to(device)
            next_ids = next_ids.to(device)
            outputs = model(event_sequences, device)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), next_ids.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * event_sequences.shape[0]
            num_samples += event_sequences.shape[0]

        print('Epoch %d, loss: %.4f' % (epoch + 1, total_loss / num_samples))
    
    return model

def test(dataset, model, options):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    window_size = options['window_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction = 'none')

    for i in tqdm(range(len(dataset))):
        with torch.no_grad(): 
            model.eval()
            sample = dataset[i]
            event_sequence = sample[0]
            if len(event_sequence) < 2:
                continue

            label = sample[1]
            windowed_sequences = []
            next_ids = []
            for j in range(ceil(len(event_sequence) / window_size)):
                if len(event_sequence[j*window_size:(j+1)*window_size]) < 2:
                    continue

                temp = event_sequence[j*window_size:(j+1)*window_size]
                windowed_sequences.append(temp[:-1])
                next_ids.append(temp[-1])

            windowed_sequences = torch.nn.utils.rnn.pad_sequence(windowed_sequences, batch_first=True, padding_value = dataset.vocab.word_to_id('<pad>'))
            next_ids = torch.tensor(next_ids)
            windowed_sequences = windowed_sequences.to(device)
            next_ids = next_ids.to(device)

            outputs = model(windowed_sequences, device)
            prob = nn.functional.softmax(outputs, dim = 1)

            pred_anomaly = False
            for j in range(len(prob)):
                if prob[j][next_ids[j]] < options['thre']:
                    pred_anomaly = True
                    break
            
        if pred_anomaly:
            if label == 1:
                TP += 1
            else:
                FP += 1
                if options['unlearn']:
                    model = unlearn(windowed_sequences, next_ids, label, model, options)
        
        else:
            if label == 0:
                TN += 1
            else:
                FN += 1
                if options['unlearn']:
                    model = unlearn(windowed_sequences, next_ids, label, model, options)
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print('Precision: %.4f, Recall: %.4f, F1: %.4f' % (precision, recall, f1))
    print(f'FN: {FN}, FP: {FP}')

                
            


        

def unlearn(windowed_sequences, next, label, model, options):
    lt = label * 2 - 1
    lamb = options['lamb']
    BND = options['BND']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    old_params = [param.clone() for param in model.parameters()]
    old_params = torch.cat([param.detach().view(-1) for param in old_params])
    

    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(model.parameters(), lr = options['unlearn_lr'])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.01)
    model.train()

    for epoch in range(options['unlearn_epochs']):
        bounding_loss = criterion(model(windowed_sequences, device), next)
        bounding_loss = torch.sum(torch.nn.functional.relu(BND - lt * bounding_loss))
        current_params = torch.cat([param.view(-1) for param in model.parameters()])
        reg_loss = (current_params - old_params).norm(2) ** 2 / 2 * lamb
        
        loss = bounding_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return model


seed_all()
dataset = "HDFS"
input_dir = "/home/datasets/log_data/HDFS"
output_dir = "/home/datasets/log_data/HDFS/continual"
parser_name = "drain"

st         = 0.5  # Similarity threshold
depth      = 5  # Depth of all leaf nodes
config = {'st': st, 'depth': depth}

options = {'lr': 0.001, 'unlearn_lr': 10**-5, 'epochs': 300, 'thre': 10**-5, 'BND': 10, 'lamb': 5 * 10**3, 'unlearn_epochs': 10, 'window_size': 11, 'unlearn': True}

# parser = Parser(input_dir, output_dir, dataset, parser_name, config)
# parser.parse()

# processor = LogDataProcessor(output_dir, output_dir, dataset, 'session')
# processor.process(n_train = 4855)

# dataset = LogDataset(output_dir, options['window_size'], mode = 'train')
# dataLoader = DataLoader(dataset, batch_size = 32, shuffle = True, collate_fn = dataset.collate_fn)
# model = deeplog(1, 128, 2, dataset.vocab.size())


# model = train(dataLoader, model, options)
# torch.save(model.state_dict(), 'deeplog.pth')


dataset = LogDataset(output_dir, options['window_size'], mode = 'test')
model = deeplog(1, 128, 2, dataset.vocab.size())
model.load_state_dict(torch.load('deeplog.pth'))
test(dataset, model, options)





