import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from mann_pytorch.DataHandler import DataHandler

# =============
# CONFIGURATION
# =============

# Learned model to be used in the /models folder
model_path = "storage_20220909-131438/models/model_49.pth"

# Inputs to be considered in the /datasets/inputs folder
input_paths = ["input_1.txt", "input_2.txt"]

# Outputs to be considered in the /datasets/outputs folder
output_paths = ["output_1.txt", "output_2.txt"]

# Retrieve global model, input, output and storage paths
script_directory = os.path.dirname(os.path.abspath(__file__))
model_path = script_directory + "/../models/" + model_path
for i in range(len(input_paths)):
    input_paths[i] = script_directory + "/../datasets/inputs/" + input_paths[i]
for i in range(len(output_paths)):
    output_paths[i] = script_directory + "/../datasets/outputs/" + output_paths[i]
storage_folder = script_directory + "/../models/storage"

# Retrieve the testing dataset, iterable on batches of one single element
data_handler = DataHandler(input_paths=input_paths, output_paths=output_paths, storage_folder=storage_folder,
                           training=False, training_set_percentage=98)
testing_data = data_handler.get_testing_data()
batch_size = 1
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

# Define the loss function
loss_fn = nn.MSELoss(reduction="mean")

# ===============
# MODEL RESTORING
# ===============

# Restore the model with the trained weights
mann_restored = torch.load(model_path)

# Set dropout and batch normalization layers to evaluation mode before running inference
mann_restored.eval()

# ============
# TESTING LOOP
# ============

# Perform one testing loop
print("\n################################### TESTING LOOP ##################################")
input("Press ENTER to start testing loop")
mann_restored.test_loop(loss_fn)

# ==============
# INFERENCE LOOP
# ==============

# Perform inference on each element of the test set
print("\n#################################### INFERENCE ####################################")
input("Press ENTER to start element-wise inference")
for X, y in test_dataloader:

    # Inference
    pred = mann_restored.inference(X)

    # Debug
    print()
    print("################################### NEW ELEMENT ###################################")
    print("INPUT:")
    print(X)
    print("GROUND TRUTH:")
    print(y)
    print("OUTPUT:")
    print(pred)
    print()
    input("Press ENTER to continue inference")


