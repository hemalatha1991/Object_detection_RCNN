# path to data, annotations and generated output
train_data_dir = '/home/hemalatha/Documents/Object_detection/data/data'
output_dir = '/home/hemalatha/Documents/Object_detection/data/output'

# Batch size
train_batch_size = 1

# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training
lr = 0.001
momentum = 0.9
weight_decay = 0.0005

# 90 coco classes + background
num_classes = 91

# no of epochs
num_epochs = 15
