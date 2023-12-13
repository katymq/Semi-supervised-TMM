# Simulations gpu
seed = 123
import os
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import random
import torch
import configparser

random.seed(seed)     # python random generator
np.random.seed(seed)  # numpy random generator

torch.manual_seed(seed) # pytorch random generator
torch.cuda.manual_seed_all(seed) # for multi-gpu

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#--------------------------------------------
from models.models_v3 import *
from utils.utils_semi import *
from utils.training import *
from utils.test import * 
#--------------------------------------------
 #! change paths
path = os.getcwd()
folder_data = r'Data'
folder_output_images = r'Results_outputs'
path_save_model = r'Results_models'
#-------------------------------
if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"  
print('Your actual device',device)
#----------------------------------------------
# Load configuration files
data_config = configparser.ConfigParser()
data_config.read('data_config_all.ini')

model_config = configparser.ConfigParser()
model_config.read('model_config_all.ini')

# Get input values from configuration files or prompt user for input
type_image = data_config.get('data_settings', 'type_image', fallback=None)
if type_image=='-1':
    type_image = input("Enter the type of image (camel or cattle): ")
    data_config.set('data_settings', 'type_image', type_image)
size = data_config.getint('data_settings', 'size', fallback=None)
if size not in [128, 256]:
    size = int(input("Enter the size of the image (128 or 256): "))
    data_config.set('data_settings', 'size', str(size))
p = data_config.getint('data_settings', 'p')
if p<0 or p>100:
    p = int(input("Enter the percentage of missing labels (0 to 100): "))
    data_config.set('data_settings', 'p', str(p))
print(f"Data info: {type_image}, {size}, {p}")

#----------------------------------------------
#! General model settings
x_dim = 1
y_dim = 1
weight_decay_ = 1e-4
clip = 10
bias_model = True
# add_loss =True # input("Enter 'True' if add_loss should be True, or 'False' otherwise: ").lower() == 'true'

z_dim = model_config.getint('model_settings', 'z_dim')
if z_dim <0:
    z_dim = int(input("Enter the value for z_dim: "))
    model_config.set('model_settings', 'z_dim', str(z_dim))

h_dim = model_config.getint('model_settings', 'h_dim')
if h_dim <0:
    h_dim = int(input("Enter the value for h_dim: "))
    model_config.set('model_settings', 'h_dim', str(h_dim))

neurons_svrnn = model_config.getint('model_settings', 'neurons_svrnn')
if neurons_svrnn <0:
    neurons_svrnn = int(input("Enter the value for neurons_svrnn: "))
    model_config.set('model_settings', 'neurons_svrnn', str(neurons_svrnn))

neurons_vsl = model_config.getint('model_settings', 'neurons_vsl')

if neurons_vsl <0:
    neurons_vsl = int(input("Enter the value for neurons_vsl: "))
    model_config.set('model_settings', 'neurons_vsl', str(neurons_vsl))

neurons_tmm = model_config.getint('model_settings', 'neurons_tmm')
if neurons_tmm <0:  
    neurons_tmm = int(input("Enter the value for neurons_tmm: "))
    model_config.set('model_settings', 'neurons_tmm', str(neurons_tmm))

neurons_tmm_1 = model_config.getint('model_settings', 'neurons_tmm_1')
if neurons_tmm_1 <0:
    neurons_tmm_1 = int(input("Enter the value for neurons_tmm_1: "))
    model_config.set('model_settings', 'neurons_tmm_1', str(neurons_tmm_1))

neurons_tmm_2 = model_config.getint('model_settings', 'neurons_tmm_2')
if neurons_tmm_2 <0:
    neurons_tmm_2 = int(input("Enter the value for neurons_tmm_2: "))
    model_config.set('model_settings', 'neurons_tmm_2', str(neurons_tmm_2))

learning_rate_svrnn = model_config.getfloat('model_settings', 'learning_rate_svrnn')
if learning_rate_svrnn <0:
    learning_rate_svrnn = float(input("Enter the learning rate for svrnn: "))
    model_config.set('model_settings', 'learning_rate_svrnn', str(learning_rate_svrnn))

learning_rate_vsl = model_config.getfloat('model_settings', 'learning_rate_vsl')
if learning_rate_vsl <0:
    learning_rate_vsl = float(input("Enter the learning rate for vsl: "))
    model_config.set('model_settings', 'learning_rate_vsl', str(learning_rate_vsl)) 

learning_rate_tmm = model_config.getfloat('model_settings', 'learning_rate_tmm')
if learning_rate_tmm <0:
    learning_rate_tmm = float(input("Enter the learning rate for tmm: "))
    model_config.set('model_settings', 'learning_rate_tmm', str(learning_rate_tmm))

n_epochs = model_config.getint('model_settings', 'n_epochs')
if n_epochs <0:
    n_epochs = int(input("Enter the number of epochs: "))
    model_config.set('model_settings', 'n_epochs', str(n_epochs))

epoch_init = model_config.getint('model_settings', 'epoch_init')
if epoch_init <0:
    epoch_init = int(input("Enter the value for epoch_init: "))
    model_config.set('model_settings', 'epoch_init', str(epoch_init))

print_every = model_config.getint('model_settings', 'print_every')
if print_every <0:
    print_every = int(input("Enter the print interval: "))
    model_config.set('model_settings', 'print_every', str(print_every))

save_every = model_config.getint('model_settings', 'save_every')
if save_every <0:
    save_every = int(input("Enter the save interval: "))
    model_config.set('model_settings', 'save_every', str(save_every))

add_loss = model_config.get('model_settings', 'add_loss')
if add_loss:
     add_loss = input("Enter if additional loss function (True False): ")
     model_config.set('model_settings', 'add_loss', add_loss)

setting = model_config.get('model_settings', 'setting')
if setting == '-1':
    setting = '' if add_loss else 'no_add'
    model_config.set('model_settings', 'setting', setting)

model1 = VSL( x_dim, z_dim, y_dim, h_dim, neurons_vsl, device, bias=bias_model)
model2 = SVRNN(x_dim, z_dim, y_dim, h_dim, neurons_svrnn, device, bool(add_loss), bias=bias_model)
model3 = TMM(x_dim, z_dim, y_dim, h_dim, neurons_tmm, device, bool(add_loss), bias=bias_model)
model4 = TMM_1(x_dim, z_dim, y_dim, h_dim, neurons_tmm_1, device, bool(add_loss),bias=bias_model)
model5 = TMM_2( x_dim, z_dim, y_dim, h_dim, neurons_tmm_2, device,bool(add_loss),bias=bias_model)

all_models = input("Enter 'True' if all models should be trained, or 'False' otherwise: ").lower() == 'true'
list_all_models = [model1, model2, model3, model4, model5]
learning_rate_all = [learning_rate_vsl, learning_rate_svrnn, learning_rate_tmm, learning_rate_tmm, learning_rate_tmm]
if all_models:
    list_models = list(list_all_models)
else:
    # Empty list to store selected models
    list_models = []
    learning_rate = []

    # Prompt the user for model selection
    for i, (option, lr) in enumerate(zip(list_all_models, learning_rate_all)):
        response = input(f"Enter 'Y' if you want to train {option.__class__.__name__}, or 'N' otherwise: ")
        if response.upper() == 'Y':
            list_models.append(option)
            learning_rate.append(lr)

    # Print the selected models
    print("Selected models:")
    for model in list_models:
        print(model.__class__.__name__)
#----------------------------------------------
# Path to save the model
# path_data = os.path.join(path, folder_data, type_image+'_'+str(size)) #! MLSP version
path_data = os.path.join(path, folder_data, type_image)      
print('Exist this folder', os.path.exists(path_data))
general_path  = os.path.join(path, path_save_model) 
os.makedirs(general_path, exist_ok=True)
print('Saving our model in', general_path)
#----------------------------------------------

index_start = int(input("Enter the index of the first image: "))
index_end = int(input("Enter the index of the last image: "))
file_name = input("Enter the name of the file to train the models: ")
filename = file_name+'.npy'  
index = 0 
results = []
for data in sorted(os.listdir(path_data)):
    # if data.endswith('.npy') and str(p) in data:
    if data.starswith(file_name) in data:
        if index >= index_start and index <= index_end:
            images = np.load(os.path.join(path_data, data))
            image_x, y_true, image_y = images
            name_image = data[:-4]
            x = torch.tensor(image_x.reshape(size*size, 1), dtype=torch.float32)
            y = torch.tensor(image_y.reshape(size*size, 1), dtype=torch.float32)
            x = x.to(device)
            y = y.to(device)
            print('start image:', name_image) 
            for model, lr in zip(list_models, learning_rate):
                print(f'{model.__class__.__name__ } has {num_param(model)} parameters to train \n')
                error_rate = run_save_train_test_model(model, device, name_image, setting, general_path, lr, weight_decay_, clip, x, y, y_true, image_y, size, folder_output_images, epoch_init, n_epochs, print_every, save_every)
                print(f'{model.__class__.__name__} saved')
                results.append({'Image': name_image, 'Model': model.__class__.__name__, 'Error': error_rate, 'Num_params': num_param(model)})
                np.save(os.path.join(general_path, filename) , results)
            print('Done:', name_image)
            with open(os.path.join(general_path, f'data_config_{name_image}.ini'), 'w') as data_configfile:
                data_config.write(data_configfile)

            with open(os.path.join(general_path,f'model_config_{name_image}.ini'), 'w') as model_configfile:
                model_config.write(model_configfile)
        index += 1