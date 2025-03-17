"""
Script to format the outputed model in results/doudizhu
into doudizhu.zip required for rlcard showdown.
"""
import os
import re
import shutil
import zipfile

def move_highest_numbered_files(source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    file_dict = {}
    pattern = re.compile(r'^(\d+)_(\d+)\.pth$')
    
    for filename in os.listdir(source_dir):
        match = pattern.match(filename)
        if match:
            prefix, number = int(match.group(1)), int(match.group(2))
            if prefix not in file_dict or number > file_dict[prefix][1]:
                file_dict[prefix] = (filename, number)
    
    for prefix, (filename, _) in file_dict.items():
        print(f"Using {filename}")
        new_filename = str(prefix) + ".pth"
        shutil.copy(os.path.join(source_dir, filename), os.path.join(dest_dir, new_filename))


def create_model_py(dest_dir):
    model_py_content = """
import os
import rlcard
from rlcard.models.model import Model as BaseModel

class Model(BaseModel):

    def __init__(self, resource_dir):
        ''' Load pretrained model
        '''
        import torch

        device = torch.device('cpu')

        model_dir = os.path.join(resource_dir, 'models')
        self.dmc_agents = []
        for i in range(3):
            model_path = os.path.join(model_dir, str(i)+'.pth')
            agent = torch.load(model_path, map_location=device)
            agent.set_device(device)
            self.dmc_agents.append(agent)

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game
        Returns:
            agents (list): A list of agents
        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.dmc_agents
"""
    
    with open(os.path.join(dest_dir, 'model.py'), 'w') as f:
        f.write(model_py_content)

def zip_models(dest_dir, dest_model_dir, zip_name):
    zip_path = os.path.join(os.getcwd(), zip_name)
    model_py_path = os.path.join(dest_dir, 'model.py')

    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Zip model.py
        zipf.write(model_py_path, 'model.py')

        # Zip models
        for root, _, files in os.walk(dest_model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.join('models', file))

def delete_copied_files(dest_dir, dest_dir_models):

    # Removed duplicated models
    if os.path.exists(dest_dir_models):
        shutil.rmtree(dest_dir_models)
        print(f"{dest_dir_models} and its contents have been deleted.")
    else:
        print(f"The directory {dest_dir_models} does not exist.")

    # Remove python file
    model_py_path = os.path.join(dest_dir, 'model.py')

    if os.path.exists(model_py_path):
        os.remove(model_py_path)
        print(f"{model_py_path} has been deleted.")
    else:
        print(f"The file {model_py_path} does not exist.")

def main():
    source_directory = os.path.join(os.getcwd(), 'results', 'doudizhu') 
    destination_directory = os.getcwd()
    destination_directory_models = os.path.join(destination_directory, 'models')
    zip_filename = 'doudizhu.zip'
    
    move_highest_numbered_files(source_directory, destination_directory_models)
    create_model_py(destination_directory)
    zip_models(destination_directory, destination_directory_models, zip_filename)
    delete_copied_files(destination_directory, destination_directory_models)
    
    print(f'Created {zip_filename} successfully.')

if __name__ == '__main__':
    main()
