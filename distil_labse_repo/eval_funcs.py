# IMPORTS
import seaborn as sns
import pickle
import torch
import numpy as np
import itertools
import tqdm
from tqdm import tqdm
from distil_funcs import CustomDataset
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from matplotlib import pyplot as plt

# EVAL FUNCTIONS - INFERENCE, COMPRESSION, PERFORMANCE


# 1. Inference
def evaluate_inference(model, encoded_eval_data, cpu_or_cuda="cpu",
                       num_samples=300, batch_sizes=[1], num_workers=[4]):
    """
    Evaluates model inference time using device setting num_samples number of times
    from eval_dataset.

    Returns mean inference for each batch size / num_worker combination, 
    which are given as lists.
  
    """
    # make sure model is in correct mode
    device = torch.device(cpu_or_cuda)
    model.to(device)

    # setup timer
    starter, ender = (torch.cuda.Event(enable_timing=True),
    torch.cuda.Event(enable_timing=True))
    timings = np.zeros((num_samples, len(batch_sizes), len(num_workers)))
    timings = np.zeros((len(batch_sizes), 1))

    # create small eval_dataset
    eval_dataset = CustomDataset(encoded_eval_data['input_ids'][:10000], 
                                      encoded_eval_data['token_type_ids'][:10000],
                                      encoded_eval_data['attention_mask'][:10000])

    # GPU warmup
    for i in range(len(eval_dataset[:100])):
        warmup_input_ids = eval_dataset[i:i+2]['input_ids'].to(device)
        warmup_attention_mask = eval_dataset[i:i+2]['attention_mask'].to(device)
        with torch.no_grad():
            _ = model(input_ids=warmup_input_ids, attention_mask=warmup_attention_mask)

    means = []
    std_devs = [] 
    # try each batch / worker combination
    for batch_size in tqdm(batch_sizes):

        worker_means = []
        worker_std_devs = []
        for worker in num_workers:

            # create dataloader
            dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                    num_workers=worker)
      
            timings = np.zeros((num_samples, 1))
            # measure inference
            with torch.no_grad():
                k=0
                for batch in dataloader:
                    # move data to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    # do actual inference recording
                    starter.record()
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    ender.record()

                    # wait for GPU sync
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[k] = curr_time

                    k=k+1
                    if k==(num_samples-1):
                        break

            mean_syn = np.mean(timings)
            std_syn = np.std(timings)

            worker_means.append(mean_syn)
            worker_std_devs.append(std_syn)
    
        means.append(worker_means)
        std_devs.append(worker_std_devs)
  
    return means, std_devs

def vis_comparatively(batch_sizes, teacher_means, teacher_std_devs, student_means, student_std_devs, title, student_name):
    """ Visualises the inference speed of teacher and student models comparatively. Inputs are outputs of evaluate_inference function."""

    dct = {'batch_sizes':batch_sizes, 'means':list(itertools.chain(*teacher_means)), 'std_devs':list(itertools.chain(*teacher_std_devs))}
    dct2 = {'batch_sizes':batch_sizes, 'means':list(itertools.chain(*student_means)), 'std_devs':list(itertools.chain(*student_std_devs))}
    teacher_df = pd.DataFrame(data=dct)
    teacher_df['model']='LaBSE'
    student_df = pd.DataFrame(data=dct2)
    student_df['model']=student_name
    data=teacher_df.append(student_df)

    dfCopy = data.copy()
    duplicates = 100 # increase this number to increase precision
    for index, row in data.iterrows():
        for times in range(duplicates):
            new_row = row.copy()
            new_row['means'] = np.random.normal(row['means'],row['std_devs']) 
            dfCopy = dfCopy.append(new_row, ignore_index=True)

    # Now Seaborn does the rest
    sns.set_style("whitegrid")
    fig = sns.barplot(x='batch_sizes',
                      y='means',
                      hue='model',
                      ci='sd',
                      data=dfCopy)

    plt.legend(loc='upper left')
    sns.set(rc={'figure.figsize':(8,5)})
    plt.ylabel('Inference Time (ms)')
    plt.xlabel('Batch Sizes')
    plt.title(title)
    plt.show()

# 2.Compression

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def evaluate_compression(teacher_model, student_model):

    teacher_parm = count_parameters(teacher_model)
    student_parm = count_parameters(student_model)
    compression = np.round(teacher_parm/student_parm, decimals=2)

    return teacher_parm, student_parm, compression

# 3. Performance

def evaluate_performance(teacher_model, student_model,encoded_eval_data, metric, batch_size=256, num_workers=4):
    """
    Evaluates the performance of the teacher and student models on the provided metric. Cosine Similarity is advised.
    """

    # create small eval_dataset
    eval_dataset = CustomDataset(encoded_eval_data['input_ids'][:batch_size*2000],
                               encoded_eval_data['token_type_ids'][:batch_size*2000],
                               encoded_eval_data['attention_mask'][:batch_size*2000])
  

    # create dataloader
    dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, 
                              num_workers=num_workers)
  
    # make sure model is in correct mode
    device = torch.device('cuda')
    student_model.to(device)
    teacher_model.to(device)


    performance = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            output_t = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            output_s = student_model(input_ids=input_ids, attention_mask=attention_mask)
            
            cpu_output_t = output_t['pooler_output'].detach().cpu()
            cpu_output_s = output_s['pooler_output'].detach().cpu()
            
            
            batch_scores = metric(cpu_output_t, cpu_output_s)
            batch_similarities = np.diag(batch_scores)
            #print(batch_similarities)
            performance.append(batch_similarities)
            
    return np.mean(performance)

def lst_of_lst(lst):
    return list(map(lambda el:[el], lst))

def convert_to_per_sentence(means):
    return lst_of_lst(list(np.array(means)/np.array(batch_sizes)))

from matplotlib import pyplot as plt
import pandas as pd
def vis_comparatively_per_sentence(batch_sizes, teacher_means, teacher_std_devs, student_means, student_std_devs, student_means2, student_std_devs2, student_means3, student_std_devs3, title, student_names):
    """ Visualises the inference speed of teacher and student models comparatively. Inputs are outputs of evaluate_inference function."""

    dct = {'batch_sizes':batch_sizes, 'means':list(itertools.chain(*teacher_means)), 'std_devs':list(itertools.chain(*teacher_std_devs))}
    dct2 = {'batch_sizes':batch_sizes, 'means':list(itertools.chain(*student_means)), 'std_devs':list(itertools.chain(*student_std_devs))}
    dct3 = {'batch_sizes':batch_sizes, 'means':list(itertools.chain(*student_means2)), 'std_devs':list(itertools.chain(*student_std_devs2))}
    dct4 = {'batch_sizes':batch_sizes, 'means':list(itertools.chain(*student_means3)), 'std_devs':list(itertools.chain(*student_std_devs3))}
    teacher_df = pd.DataFrame(data=dct)
    teacher_df['model']='LaBSE'
    student_df = pd.DataFrame(data=dct2)
    student_df['model']=student_names[0]
    student_df2 = pd.DataFrame(data=dct3)
    student_df2['model']=student_names[1]
    student_df3 = pd.DataFrame(data=dct4)
    student_df3['model']=student_names[2]
    data=teacher_df.append([student_df, student_df2, student_df3])

    dfCopy = data.copy()
    duplicates = 100 # increase this number to increase precision
    for index, row in data.iterrows():
        for times in range(duplicates):
            new_row = row.copy()
            new_row['means'] = np.random.normal(row['means'],row['std_devs']) 
            dfCopy = dfCopy.append(new_row, ignore_index=True)

    # Now Seaborn does the rest
    sns.set_style("whitegrid")
    fig = sns.barplot(x='batch_sizes',
                      y='means',
                      hue='model',
                      ci='sd',
                      data=dfCopy)

    plt.legend(loc='upper right')
    sns.set(rc={'figure.figsize':(10,7)}) # 8,5
    plt.ylabel('Inference time per sample (ms)')
    plt.xlabel('Batch Sizes')
    plt.title(title)
    plt.show()