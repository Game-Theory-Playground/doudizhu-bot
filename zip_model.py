"""
Script to format the outputed model in results/doudizhu
into zip required for rlcard showdown.
"""

import os
import re
import shutil
import zipfile

def move_highest_numbered_files(dir):

    file_dict = {}
    pattern = re.compile(r'^(\d+)_(\d+)\.pth$')
    
    for filename in os.listdir(dir):
        match = pattern.match(filename)
        if match:
            prefix, number = int(match.group(1)), int(match.group(2))
            if prefix not in file_dict or number > file_dict[prefix][1]:
                file_dict[prefix] = (filename, number)
    
    for filename, _ in file_dict.values():
        shutil.copy(os.path.join(dir, filename), os.path.join(dir, filename))

def create_model_py(dest_dir):
    model_py_content = """
import os

import rlcard
from rlcard.models.model import Model as BaseModel

class Model(BaseModel):
    ''' A pretrained model on Dou Dizhu with DMC
    '''

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

def zip_models(dest_dir, zip_name):
    zip_path = os.path.join(os.getcwd(), zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dest_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dest_dir)
                zipf.write(file_path, arcname)
    
    shutil.rmtree(dest_dir)

def main():
    models_directory = os.path.join(os.getcwd(), 'results/doudizhu')
    zip_filename = 'doudizhu.zip'
    
    move_highest_numbered_files(models_directory)
    create_model_py(models_directory)
    zip_models(models_directory, zip_filename)
    
    print(f'Created {zip_filename} successfully.')


if __name__ == '__main__':
    main()
