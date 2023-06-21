seed = 123
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
path = os.getcwd()
folder_data = r'Data'
#-------------------------------
if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"  
print('Your actual device',device)
#----------------------------------------------
general_path  = os.path.join(os.getcwd(), 'Results_save_models') 
type_image = str(input("Enter the type of image (camel or cattle): "))
size = int(input("Enter the size of the image (128 or 256): "))
p = int(input("Enter the percentage of missing labels (0 to 100): "))
#----------------------------------------------
#! General model settings
x_dim = 1
y_dim = 1
model_config = configparser.ConfigParser()
model_config.read(os.path.join(general_path, 'model_config_all.ini'))

z_dim = model_config.getint('model_settings', 'z_dim')
h_dim = model_config.getint('model_settings', 'h_dim')
neurons_svrnn = model_config.getint('model_settings', 'neurons_svrnn')
neurons_vsl = model_config.getint('model_settings', 'neurons_vsl')
neurons_tmm = model_config.getint('model_settings', 'neurons_tmm')
add_loss = model_config.get('model_settings', 'add_loss')
add_loss = model_config.get('model_settings', 'add_loss')
setting = '' if add_loss else 'no_add'
epoch_init = model_config.getint('model_settings', 'n_epochs')

model1 = TMM(x_dim, z_dim, y_dim, h_dim, neurons_tmm, device, add_loss)
model2 = VSL( x_dim, z_dim, y_dim, h_dim, neurons_vsl, device)
model3 = SVRNN(x_dim, z_dim, h_dim, y_dim, neurons_svrnn, device, add_loss)
model4 = TMM_1(x_dim, z_dim, y_dim, h_dim, neurons_tmm, device, add_loss)
model5 = TMM_2(x_dim, z_dim, y_dim, h_dim, neurons_tmm-1, device, add_loss)

#----------------------------------------------
# List of images
path_data = os.path.join(path, folder_data,type_image+'_'+str(size))
file_name = input("Enter the name of the file to save the results: ")
filename = file_name+'.xlsx'  # Name of the output file  
list_models = [model1, model2, model3, model4, model5]
# print('Exist this folder', os.path.exists(path_data))
# print('Path to save our data', path_data)
results = []
for data in sorted(os.listdir(path_data)):
    if data.endswith('.npy') and str(p) in data:
        images = np.load(os.path.join(path_data, data))
        image_x, y_true, image_y = images
        name_image = data[:-4]
        print('Name of the image', name_image)
        for model in list_models:
            error_rate = test_model(model, device, name_image, setting, general_path, image_x, image_y, y_true,size,epoch_init)
            #print('Finish the test for the model', model.__class__.__name__)
            if error_rate is not None:
                results.append({'Image': name_image, 'Model': model.__class__.__name__, 'Error': error_rate})
                
df = pd.DataFrame(results)
df.to_excel(filename, index=False)
