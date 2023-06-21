import configparser

# Create data configuration object
data_config = configparser.ConfigParser()

# Set default values for data configuration
data_config['data_settings'] = {}
data_config['data_settings']['type_image'] = '-1'
data_config['data_settings']['size'] = '128'
data_config['data_settings']['p'] = '60'
data_config['data_settings']['index'] = '-1'

# Write data configuration to file
with open('data_config_all.ini', 'w') as configfile:
    data_config.write(configfile)

# Create model configuration object
model_config = configparser.ConfigParser()

# Set default values for model configuration
model_config['model_settings'] = {}
model_config['model_settings']['z_dim'] = '1'
model_config['model_settings']['h_dim'] = '14'
model_config['model_settings']['neurons_svrnn'] = '14'
model_config['model_settings']['neurons_vsl'] = '30'
model_config['model_settings']['neurons_tmm'] = '25'
model_config['model_settings']['learning_rate_svrnn'] = '0.007'
model_config['model_settings']['learning_rate_vsl'] = '0.007'
model_config['model_settings']['learning_rate_tmm'] = '0.01'
model_config['model_settings']['n_epochs'] = '100'
model_config['model_settings']['epoch_init'] = '1'
model_config['model_settings']['print_every'] = '5'
model_config['model_settings']['save_every'] = '10'
model_config['model_settings']['add_loss'] = 'True'
model_config['model_settings']['setting'] = '-1'
# Write model configuration to file
with open('model_config_all.ini', 'w') as configfile:
    model_config.write(configfile)