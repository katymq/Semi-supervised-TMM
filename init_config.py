import configparser

# Create data configuration object
data_config = configparser.ConfigParser()

# Set default values for data configuration
data_config['data_settings'] = {}
data_config['data_settings']['type_image'] = '-1'
data_config['data_settings']['size'] = '-1'
data_config['data_settings']['p'] = '-1'
data_config['data_settings']['index'] = '-1'

# Write data configuration to file
with open('data_config.ini', 'w') as configfile:
    data_config.write(configfile)

# Create model configuration object
model_config = configparser.ConfigParser()

# Set default values for model configuration
model_config['model_settings'] = {}
model_config['model_settings']['z_dim'] = '-1'
model_config['model_settings']['h_dim'] = '-1'
model_config['model_settings']['num_neurons'] = '-1'
model_config['model_settings']['learning_rate'] = '-1'
model_config['model_settings']['n_epochs'] = '-1'
model_config['model_settings']['epoch_init'] = '-1'
model_config['model_settings']['print_every'] = '-1'
model_config['model_settings']['save_every'] = '-1'
model_config['model_settings']['add_loss'] = '-1'
model_config['model_settings']['sel_model'] = '-1'
model_config['model_settings']['setting'] = '-1'
# Write model configuration to file
with open('model_config.ini', 'w') as configfile:
    model_config.write(configfile)