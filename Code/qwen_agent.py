import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import re
import torch.nn.functional as F

class QwenAgent:
    def __init__(self, device, accelerator, cache_dir='~/.cache', dropout=0.5, policy_lm=None,
                 max_new_tokens=32, use_bfloat16=False):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            policy_lm,  torch_dtype="auto", device_map="balanced", attn_implementation="flash_attention_2",
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(policy_lm)
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
   
    def prepare(self): 
        self.model = self.accelerator.prepare(self.model)

    def get_layer_embeddings(self, obs):
        result = {}

        sys_prompt = f"""
        You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

        1. Basic Actions
        Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
        Basic Action 1: CLICK 
            - purpose: Click at the specified position.
            - format: CLICK <point>[[x-axis, y-axis]]</point>
            - example usage: CLICK <point>[[101, 872]]</point>
        
        Basic Action 2: TYPE
            - purpose: Enter specified text at the designated location.
            - format: TYPE [input text]
            - example usage: TYPE [Shanghai shopping mall]

        Basic Action 3: SCROLL
            - Purpose: SCROLL in the specified direction.
            - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
            - Example Usage: SCROLL [UP]
            
        2. Custom Actions
        Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

        Custom Action 1: PRESS_BACK
            - purpose: Press a back button to navigate to the previous screen.
            - format: PRESS_BACK
            - example usage: PRESS_BACK

        Custom Action 2: PRESS_HOME
            - purpose: Press a home button to navigate to the home page.
            - format: PRESS_HOME
            - example usage: PRESS_HOME

        Custom Action 3: COMPLETE
            - purpose: Indicate the task is finished.
            - format: COMPLETE
            - example usage: COMPLETE

        Custom Action 4: IMPOSSIBLE
            - purpose: Indicate the task is impossible.
            - format: IMPOSSIBLE
            - example usage: IMPOSSIBLE

        And your current task instruction and associated screenshot are as follows:
        Final goal: {obs['task']}
        Screenshot: 
        Your output must be in one line. Do not split it into two lines. 
        Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks:
        action:
        """  

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sys_prompt},
                    {"type": "image", "image": obs['image_path']},
                ],
            }
        ]

        # 构造输入
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        decoder_hidden_states = outputs.hidden_states

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids = generated_ids.to(self.device)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])
  
        output_tokens = self.processor.tokenizer.tokenize(output_text[0])
        output_token_count = len(output_tokens)

        layer_embeddings = []
        for layer_hidden in decoder_hidden_states[1:]: 
            generated_token_embeddings = layer_hidden[0, -output_token_count:, :] 
            sentence_embedding = generated_token_embeddings.mean(dim=0)
            layer_embeddings.append(sentence_embedding)

        input_token_count = len(self.processor.tokenizer.tokenize(chat_text))

        result['input_token_count'] = input_token_count
        result['output_token_count'] = output_token_count
        result['layer_embeddings'] = layer_embeddings

        return result
    
    def get_layer_embeddings_short(self, obs):
        result = {}

        sys_prompt = f"""
        Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. 
        And your current task instruction and associated screenshot are as follows:
        Final goal: {obs['task']}
        Screenshot: 
        action:
        """  

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sys_prompt},
                    {"type": "image", "image": obs['image_path']},
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        decoder_hidden_states = outputs.hidden_states
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids = generated_ids.to(self.device)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])
        output_tokens = self.processor.tokenizer.tokenize(output_text[0])
        output_token_count = len(output_tokens)

        layer_embeddings = []
        for layer_hidden in decoder_hidden_states[1:]: 
            generated_token_embeddings = layer_hidden[0, -output_token_count:, :]  
            sentence_embedding = generated_token_embeddings.mean(dim=0) 
            layer_embeddings.append(sentence_embedding)

        input_token_count = len(self.processor.tokenizer.tokenize(chat_text))

        result['input_token_count'] = input_token_count
        result['output_token_count'] = output_token_count
        result['layer_embeddings'] = layer_embeddings

        return result

    def get_confidence(self, obs):
        sys_prompt = f"""
        You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

        1. Basic Actions
        Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
        Basic Action 1: CLICK 
            - purpose: Click at the specified position.
            - format: CLICK <point>[[x-axis, y-axis]]</point>
            - example usage: CLICK <point>[[101, 872]]</point>
        
        Basic Action 2: TYPE
            - purpose: Enter specified text at the designated location.
            - format: TYPE [input text]
            - example usage: TYPE [Shanghai shopping mall]

        Basic Action 3: SCROLL
            - Purpose: SCROLL in the specified direction.
            - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
            - Example Usage: SCROLL [UP]
            
        2. Custom Actions
        Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

        Custom Action 1: PRESS_BACK
            - purpose: Press a back button to navigate to the previous screen.
            - format: PRESS_BACK
            - example usage: PRESS_BACK

        Custom Action 2: PRESS_HOME
            - purpose: Press a home button to navigate to the home page.
            - format: PRESS_HOME
            - example usage: PRESS_HOME

        Custom Action 3: COMPLETE
            - purpose: Indicate the task is finished.
            - format: COMPLETE
            - example usage: COMPLETE

        Custom Action 4: IMPOSSIBLE
            - purpose: Indicate the task is impossible.
            - format: IMPOSSIBLE
            - example usage: IMPOSSIBLE

        And your current task instruction and associated screenshot are as follows:
        Final goal: {obs['task']}
        Screenshot: 
        Your output must be in one line. Do not split it into two lines. 
        Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks:
        action:
        """  

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sys_prompt},
                    {"type": "image", "image": obs['image_path']},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                output_scores=True,
                return_dict_in_generate=True,
                num_beams=5,
                num_return_sequences=2,
                do_sample=False,  
            )

        generated_sequences = outputs.sequences
        decoded_outputs = self.processor.batch_decode(
            generated_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        actions = []
        for out in decoded_outputs:
            parts = out.strip().split('action: ')
            if len(parts) > 1:
                action = parts[1].strip()
                actions.append(action)

        scores = outputs.sequences_scores 
        probs = torch.nn.functional.softmax(scores, dim=0).tolist()

        top_actions_with_confidence = list(zip(actions, probs))

        best_action = actions[0] if actions else "IMPOSSIBLE"
        return best_action, top_actions_with_confidence
    
    def get_input(self, obs):
        sys_prompt = f"""
        You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

        1. Basic Actions
        Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
        Basic Action 1: CLICK 
            - purpose: Click at the specified position.
            - format: CLICK <point>[[x-axis, y-axis]]</point>
            - example usage: CLICK <point>[[101, 872]]</point>
        
        Basic Action 2: TYPE
            - purpose: Enter specified text at the designated location.
            - format: TYPE [input text]
            - example usage: TYPE [Shanghai shopping mall]

        Basic Action 3: SCROLL
            - Purpose: SCROLL in the specified direction.
            - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
            - Example Usage: SCROLL [UP]
            
        2. Custom Actions
        Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

        Custom Action 1: PRESS_BACK
            - purpose: Press a back button to navigate to the previous screen.
            - format: PRESS_BACK
            - example usage: PRESS_BACK

        Custom Action 2: PRESS_HOME
            - purpose: Press a home button to navigate to the home page.
            - format: PRESS_HOME
            - example usage: PRESS_HOME

        Custom Action 3: COMPLETE
            - purpose: Indicate the task is finished.
            - format: COMPLETE
            - example usage: COMPLETE

        Custom Action 4: IMPOSSIBLE
            - purpose: Indicate the task is impossible.
            - format: IMPOSSIBLE
            - example usage: IMPOSSIBLE

        And your current task instruction and associated screenshot are as follows:
        Final goal: {obs['task']}
        Screenshot: 
        Your output must be in one line. Do not split it into two lines. 
        Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks:
        action:
        """  
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sys_prompt},
                    {"type": "image", "image": obs['image_path']},
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        self.device = self.model.device
        inputs = inputs.to(self.device)
        input_ids = inputs['input_ids']
        with torch.no_grad():
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            input_embedding_mean = inputs_embeds.mean(dim=1) 
       
        input_embedding_list = input_embedding_mean.squeeze(0).cpu().tolist()

        return input_embedding_list