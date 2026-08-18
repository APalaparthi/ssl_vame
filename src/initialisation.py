import vame as vm
import os

working_dir="/its/home/ap2037/ssl_vame/"

dummy_video =[os.path.join(working_dir, 'data', 'raw', 'animal_001.mp4')]
config_file_path=vm.init_new_project(
    project="vame_pipeling_aug17",
    videos= dummy_video,
    working_directory=working_dir,
    videotype='.mp4'
)

#vm.create_trainset(working_dir)
print(f"training dataset sucessfully created \n config file located at: {config_file_path}")