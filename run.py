from qwen_agent import QwenAgent
#from qwen25_agent import Qwen25Agent
import torch
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from datetime import timedelta
from util import plot_embedding_layer_diff_norms,get_embeddings_loop, Gaussian_fitting, TV_score, get_confidence_loop, get_input_embedding_loop, get_input_score


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(minutes=40)), kwargs_handlers=[ddp_kwargs])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = QwenAgent(device=device, accelerator=accelerator, 
                      policy_lm="/data3/wuzh/OOD/weight/AITZ", max_new_tokens=128)

#input_path = "/data3/wuzh/OOD/AITZ_ID/OOD_dataset/omniact-desktop.json"
#output_path = "/data3/wuzh/OOD/AITZ_ID/OOD_dataset/omniact-desktop_embeddings.json"
#print("start get embeddings")
#get_embeddings_loop(input_path, output_path, agent)


#output_path = "/data3/wuzh/OOD/AITZ_ID/train/AITZ_train_embedding.json"
#fitting_path = "/data3/wuzh/OOD/AITZ_ID/train/AITZ_train_embedding_fitting.json"
#print("start Gaussian fitting")
#fitting = Gaussian_fitting(output_path, fitting_path)


#test_path = "/data3/wuzh/OOD/long_prompt/OOD_dataset/omniact-web_embedding.json"
#test_output_path = "/data3/wuzh/OOD/long_prompt/OOD_dataset/omniact-web_TVscore.json"

#test_path = "/data3/wuzh/OOD/AITZ_ID/OOD_dataset/omniact-desktop_embeddings.json"
#test_output_path = "/data3/wuzh/OOD/AITZ_ID/OOD_dataset/omniact-desktop_TVscore.json"
#TV_score(fitting, test_path, test_output_path)


#input_path = "/data3/wuzh/OOD/long_prompt/OOD_dataset/omniact-web.json"
#output_path = "/data3/wuzh/OOD/long_prompt/OOD_dataset/omniact-web_input.json"
#get_confidence_loop(input_path, output_path, agent)
#get_input_embedding_loop(input_path, output_path, agent)


#input_path = "/data3/wuzh/OOD/long_prompt/OOD_dataset/omniact-web.json"
#output_path = "/data3/wuzh/OOD/long_prompt/OOD_dataset/omniact-web_entropy2.json"
#get_confidence_loop(input_path, output_path, agent)

#input_path = "/data3/wuzh/OOD/AITZ_ID/train/AITZ_train_autoui.json"
#test_path = "/data3/wuzh/OOD/AITZ_ID/train/AITZ_train_autoui.json"
#output_path = "/data3/wuzh/OOD/AITZ_ID/train/AITZ_train_autoui_inputscore.json"
#get_input_score(input_path, test_path, output_path)
