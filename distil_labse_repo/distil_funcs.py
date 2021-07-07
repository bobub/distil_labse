# Imports
import torch
from labml_nn.transformers.switch import SwitchTransformer, SwitchTransformerLayer, SwitchFeedForward
from labml_nn.transformers import MultiHeadAttention
from labml_nn.transformers.feed_forward import FeedForward
import numpy as np
from transformers import AutoConfig, AutoModel
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from random import choice
from sklearn.decomposition import PCA
from copy import deepcopy
from transformers import BertModel, BertConfig


# Custom dataset function to store Open Subtitles data
class CustomDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, input_ids, token_type_ids, attention_masks):
        'Initialization'
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_masks = attention_masks

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_ids)

  def __getitem__(self, index):
        'Generates one sample of data'

        input_id = self.input_ids[index]
        token_type_ID = self.token_type_ids[index]
        attention_mask = self.attention_masks[index]
        sample = {'input_ids':input_id, 'token_type_ids':token_type_ID , 'attention_mask':attention_mask}

        return sample

# Weights init and switch init initialise the weights for the model as desribed in Switch Transformer paper
def weights_init(tensor: torch.Tensor):
    if isinstance(tensor, nn.Linear):
        switch_init(tensor.weight.data)
        torch.nn.init.zeros_(tensor.bias.data)
    if isinstance(tensor, nn.LayerNorm):
        torch.nn.init.zeros_(tensor.weight.data)
        torch.nn.init.zeros_(tensor.bias.data)

def switch_init(tensor: torch.Tensor, s: float = 0.1, mean: float=0) -> torch.Tensor:
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(s/fan_in)

    return torch.nn.init.trunc_normal_(tensor=tensor, mean=mean, std=std)


class LaBSE_Switch(nn.Module):
    """
    Torch module for to create a Switch Transformer for LaBSE. 
    Can be used for other BERT based models too, just change the input_id
    tokenization and word_embedding module.

    Inputs:
    config = dictionary of configuration
    word_embeddings_module = torch module mapping token ids to word embeddings

    Forward:
    Input_ids = ids using labse tokenizer 
    attention_mask = binary, indicates to model which tokens should be attended to,
    and which should not.

    Outputs:
    outputs = a dictionary containing x, counts, route_prob, n_dropped, logits, attention, values

    See Switch Transformer paper to understand all except:
    attention, values and logits, which are used during knowledge distillation.
    
    """

    def __init__(self, config, word_embeddings_module):

        super().__init__()
        # set the switch transformer as the actual neural net
        self.switch_model = SwitchTransformer(
            
          SwitchTransformerLayer(
          d_model=config['d_model'],
          attn=MultiHeadAttention(config['heads'], config['d_model'], config['dropout']),

              feed_forward=SwitchFeedForward(
              capacity_factor=config['capacity_factor'],
                              drop_tokens=config['drop_tokens'],
                              is_scale_prob=config['is_scale_prob'],
                              n_experts=config['n_experts'],
                              expert=FeedForward(config['d_model'], config['d_ff'], config['dropout_ffn']),
                              d_model=config['d_model']),
                              dropout_prob=config['dropout']),
                              config['n_layers'],
                              d_out = int(768),
                              dropout_prob = config['dropout'])
        # initialise weights
        # self.switch_model.apply(weights_init)
        
        #  module that maps input tokens into embedding vectors
        self.word_embeddings = word_embeddings_module

        # get attention weights from teacher
        # self.weight_init_from_teacher(teacher_model=teacher_model, int_matches=int_matches)
    
    def weight_init_from_teacher(self, teacher_model, int_matches):
        
        
        """
          Initialises attention modules of student with those of the teacher for the  --- specific to LaBSE and DistilSwitch
          int_matches should be a list of tuples of [(teacher_layer, student_layer),...]
          e.g. int_matches = [(5,0),(11,2)] --> give attention weights of teacher layer 5 to student layer 0     
          """
        # teacher_model=load_teacher(device=torch.device('cuda'))
        self.switch_model.layers[int_matches[1]].attn.query.linear.weight = teacher_model.encoder.layer[int_matches[0]].attention.self.query.weight
        self.switch_model.layers[int_matches[1]].attn.query.linear.bias = teacher_model.encoder.layer[int_matches[0]].attention.self.query.bias
        self.switch_model.layers[int_matches[1]].attn.key.linear.weight = teacher_model.encoder.layer[int_matches[0]].attention.self.key.weight
        self.switch_model.layers[int_matches[1]].attn.key.linear.bias = teacher_model.encoder.layer[int_matches[0]].attention.self.key.bias
        self.switch_model.layers[int_matches[1]].attn.value.linear.weight = teacher_model.encoder.layer[int_matches[0]].attention.self.value.weight
        self.switch_model.layers[int_matches[1]].attn.value.linear.bias = teacher_model.encoder.layer[int_matches[0]].attention.self.value.bias
        self.switch_model.layers[int_matches[1]].attn.output.weight = teacher_model.encoder.layer[int_matches[0]].attention.output.dense.weight
        self.switch_model.layers[int_matches[1]].attn.output.bias = teacher_model.encoder.layer[int_matches[0]].attention.output.dense.bias
#         self.switch_model.layers[int_matches[1]].norm_ff.weight = teacher_model.encoder.layer[int_matches[0]].output.LayerNorm.weight
#         self.switch_model.layers[int_matches[1]].norm_ff.bias = teacher_model.encoder.layer[int_matches[0]].output.LayerNorm.bias

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        
        # masks and token type ids not used, as we're just creating sentence embeddings for classification tasks
        
        # word embeddings of shape [batch, seq_len, d_model]
        input_embeddings = self.word_embeddings(input_ids)

        # model input on shape [seq_len, batch, d_model] and mask
        _batch,_seq_len,_n_hid = input_embeddings.shape
        #print(_n_hid)

        # call switch transformer
        outputs = self.switch_model(torch.reshape(input_embeddings, (_seq_len, _batch, _n_hid)),
                              attention_mask=None)
        
        return outputs

# function to blackbox load the student for distillation - can be switch or bert based
def load_student(name, student_config, device, teacher_model, int_matches, N_LAYERS):

    if name!='switch':
        
        # for pretrained bert models - setup config
        student_config = BertConfig.from_pretrained(name)
        student_config.num_hidden_layers = N_LAYERS
        student_config.output_hidden_states = True
        student_config.output_attentions = True
        student_config.use_cache = True
        student_config.is_decoder = True
        
        # load model and set input embeddings
        student_model = BertModel.from_pretrained(name, config=student_config)
        student_model.set_input_embeddings(teacher_model.get_input_embeddings())
        student_model = student_model.float()
        student_model.to(device=device)
        
        return student_model

    if name=='switch':
        
        # create compressed word embeddings from those of the teacher
        word_embeddings = deepcopy(teacher_model.get_input_embeddings())
        compressed_word_embeddings = word_embedding_compression(word_embeddings, student_config['d_model'])
        
        # create student model
        student_model = LaBSE_Switch(config=student_config, word_embeddings_module=compressed_word_embeddings)
        
        # initialise weights
        student_model.switch_model.apply(weights_init)
        student_model.weight_init_from_teacher(teacher_model=teacher_model, int_matches=int_matches)
        
        # convert model to float32 and move to device
        student_model = student_model.float() 
        student_model.to(device=device)
    
        return student_model

# loads teacher model from Huggingface
def load_teacher(device):
    teacher_config = AutoConfig.from_pretrained('sentence-transformers/LaBSE')
    teacher_config.output_hidden_states = True
    teacher_config.output_attentions = True
    teacher_config.use_cache = True
    teacher_config.is_decoder = True
    teacher_model = AutoModel.from_pretrained('sentence-transformers/LaBSE', config=teacher_config)
    teacher_model.float() # needs to be 32 bit precision to get decent results from distillation
    teacher_model.to(device=device)
    
    return teacher_model

# Adaptor for BERT based models
def simple_adaptor(batch, model_outputs):
    
    # values need to be reformatted from Huggingface 'past_key_values' output
    values = []
    for i in model_outputs['past_key_values']:
        values.append(i[1])
    values = torch.stack(values)
    
    attentions = []
    for j in model_outputs['attentions']:
        attentions.append(inv_softmax(j))
    attentions = torch.stack(attentions)
    
    # we use pooler output as logits
    return {'logits': model_outputs['pooler_output'],
            'hidden': model_outputs['hidden_states'],
            #'attention': model_outputs['attentions'],
            'attention':attentions,
            'inputs_mask': batch['attention_mask'],
            'value_relation': values,
            'pooler_output':model_outputs['pooler_output']}

def inv_softmax(x,C=-50):
    # reverses softmax operation - used in teacher_adaptor
    # C variable sets the min value of the scores, -50 works well.
    result = torch.log(x)
    result = torch.where(result <= float('-inf'), torch.full_like(result,C), result)
    return result

def teacher_adaptor(batch, model_outputs):
    # selects relevant model and batch outputs used for distillation loss calculation
    values = []
    for i in model_outputs['past_key_values']:
        values.append(i[1])
    values = torch.stack(values)
    
    attentions = []
    for j in model_outputs['attentions']:
        attentions.append(inv_softmax(j))
    attentions = torch.stack(attentions)
    
    # print(model_outputs['pooler_output'].requires_grad)

    return {#'logits': model_outputs['last_hidden_state'],
            'logits':model_outputs['pooler_output'],
            'hidden': model_outputs['hidden_states'],
            #'attention': model_outputs['attentions'],
            'attention': attentions,
            'inputs_mask': batch['attention_mask'],
            'value_relation': values,
            'pooler_output':model_outputs['pooler_output']}

# adaptor for switch model
def switch_student_adaptor(batch, model_outputs):
    # selects relevant model and batch outputs and reformats them
    # needs to have same shapes as teacher adaptor

    # reformat attention
    layers, len, len, batch_size, heads = model_outputs['attention'].shape
    attention = model_outputs['attention'].reshape(layers, batch_size, heads, len, len)

    # reformat logits
    len, batch_size, d_model = model_outputs['logits'].shape
    logits = model_outputs['logits'].reshape(batch_size, len, d_model)
    # print(model_outputs['pooler_output'].requires_grad)

    # reformat values
    layers, len, batch_size, heads, embedding_per_head = model_outputs['values'].shape
    values = model_outputs['values'].reshape(layers, batch_size, heads, len, embedding_per_head)

    return {#'logits': logits,
            'logits':model_outputs['pooler_output'],
            'counts': model_outputs['counts'],
            'attention': attention,
            'inputs_mask': batch['attention_mask'],
            'route_prob': model_outputs['route_prob'],
            'n_dropped': model_outputs['n_dropped'],
            'value_relation': values}

# Predict function evaluates model every epoch to show training progress
def predict(model, teacher_model, eval_dataset, step, device, STUDENT, BATCH_SIZE, eval_metric='cosine_similarity', feedback=True):
    '''
    model = student_model
    teacher_model = labse
    eval_dataset = num of dev set samples to test model on per callback
    device = cuda or cpu
    student = switch or !switch
    eval_metric = metric to evaluate the model - mse or cosine_similarity
    '''
    model.eval()
    student_logits = []
    teacher_logits =[]
    batch_counts = []
    batch_n_dropped = []
    batch_route_prob = []
    
    dataloader = DataLoader(eval_dataset,batch_size=BATCH_SIZE)
    print('Running callback function on {} dev set samples...'.format(len(eval_dataset)))
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            model_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_S = model_outputs['pooler_output']
            logits_T = teacher_model(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
            cpu_logits_S = logits_S.detach().cpu()
            cpu_logits_T = logits_T.detach().cpu()
            
            if STUDENT=='switch' and feedback==True:
                counts = model_outputs['counts'].detach().cpu()
                n_dropped = model_outputs['n_dropped']
                route_prob = model_outputs['route_prob'].detach().cpu()

        for i in range(len(cpu_logits_S)):
            student_logits.append(cpu_logits_S[i].numpy())
            teacher_logits.append(cpu_logits_T[i].numpy())
            
        if STUDENT=='switch' and feedback==True:
            for i in range(len(counts)):
                batch_counts.append(counts[i].numpy())
                batch_n_dropped.append(n_dropped[i])
                batch_route_prob.append(route_prob[i].numpy())
                
    model.train()
    student_logits = np.array(student_logits)
    teacher_logits = np.array(teacher_logits)

    if eval_metric=='cosine_similarity':
        
        similarities = np.diag(cosine_similarity(student_logits, teacher_logits))
        print ("Average cosine similarity for these samples: ", np.mean(similarities))
    
    if eval_metric=='mse':
        mse_error = mean_squared_error(student_logits, teacher_logits)
        print ("Average mean squared error for these samples: ", mse_error)
        
    if STUDENT=='switch' and feedback==True:
        switch_counts = np.array(batch_counts)
        switch_n_dropped = np.array(batch_n_dropped)
        switch_route_prob = np.array(batch_route_prob)
        print('SWITCH BEHAVIOUR:')
        print('Counts Shape: \n', switch_counts.shape)
        print('Counts: \n', switch_counts)
        print('N_dropped: \n', switch_n_dropped)
        print('Route Prob: \n', switch_route_prob)

    return torch.Tensor([np.mean(similarities)])

# generates random parameters for hyperparam tuning
def generate_random_params(params):
    # input: params dictionary containing lists of possible values
    chosen_params = {}
    for param in params:
        chosen_params[param] = choice(params[param])
    return chosen_params

def word_embedding_compression(word_embedding_module, d_model):
    
    """
    Compresses a given word_embedding_module (type torch.Embedding) into a module of d_model dimensionality.
    """
    word_embedding_matrix = word_embedding_module.weight
    assert word_embedding_matrix.shape[1]>=d_model, 'The desired word embedding dimensionality is greater than the teacher word embeddings. That is not compression! Make d_model smaller.'
    # return the module if it's the same dimensionality
    if word_embedding_matrix.shape[1]==d_model:
        return word_embedding_module
    # else compress
    pca = PCA(n_components = d_model)
    compressed_word_embedding_matrix = pca.fit_transform(word_embedding_matrix.detach().cpu().numpy())
    compressed_word_embedding_matrix = torch.from_numpy(compressed_word_embedding_matrix)
    word_embedding_module.weight = torch.nn.parameter.Parameter(compressed_word_embedding_matrix)
    return word_embedding_module