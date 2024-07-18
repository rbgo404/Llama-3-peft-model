import numpy as np
import pandas as pd
import torch
import re
import os
from tqdm import tqdm
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from peft import PeftModel
from huggingface_hub import login

class InferlessPythonModel:
    def initialize(self):
        # nfs_volume = os.getenv("NFS_VOLUME")
        self.nfs_volume = "/var/nfs-mount/model_files"
        model_name ="meta-llama/Meta-Llama-3-8B-Instruct"
        output_dir = f"{self.nfs_volume}/LLAMA_3_trainned_model"
        # hf_access_token = os.getenv("HF_TOKEN")
        hf_access_token = "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"
        login(token = hf_access_token)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        
        ft_model = PeftModel.from_pretrained(base_model, output_dir)
        ft_model = ft_model.merge_and_unload()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.Pipeline = pipeline("text-generation", model=ft_model, tokenizer= self.tokenizer, torch_dtype=torch.bfloat16, max_new_tokens = 8000,do_sample=True, top_p=.98, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        
        self.system_prompt = "You are an expert in extracting e-commerce related products, price, Discount %,  Price, Brand name, Seller name, Quantity, L1 category, L2 category, L3 category"
        self.terminators = [
            self.Pipeline.tokenizer.eos_token_id,
            self.Pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    def infer(self,inputs):
        file_path = inputs["file_path"]
        raw_data = pd.read_excel(f"{self.nfs_volume}/{file_path}")
        raw_data_prod = raw_data[['vtionid', 'app', 'Cart']]
        raw_data_prod = raw_data_prod.dropna().reset_index(drop=True)
        raw_data = raw_data_prod[0:10]
        
        raw_data.head()
        raw_data['parsed_output']=''
        
        for index, row in tqdm(raw_data.iterrows()):
            try:
                # Check if 'name' is not NaN and length is greater than 2
                if len(row["Cart"]) >7990:
                    raw_data.at[index, "parsed_output"] = {}
                elif pd.notna(row['Cart']) and len(str(row['Cart'])) > 2:
                    # output_df = output_df.append(row)
        
                    user_input  =  row['Cart']
        
                    combined_prompt  = f"""Extract the product information based on the input: {user_input}
        
                            Take a deep breath and Follow the below instructions very carefully:
                            
                            1. first Translate the user input text to English.
                            2. Identify any tangible items or services being sold or offered in an e-commerce context.
                            3. If the input data does not contain a product, return an empty dictionary: {{}}.
        
                            Example:
        
                            user_input 1 = "your account - amazon.in"
                            answer 1 = {{}}
        
                            user_input 2 = "amazon.in: buy any 1 item(s) for &#8377;1.00 promotion"
                            answer 2 = {{}}
        
                            If the input contains terms like "your account - amazon.in", "mobile recharge", or "cancel your order", return an empty dictionary: {{}}.
        
                            4. Each product must be detailed individually, with each assigned to a unique product key (e.g., product1, product2, etc.).
                            5. Break down the product into details such as product name and its attributes, then tag the categories properly.
                            6. Identify each product carefully and tag each product with L1, L2, and L3 categories, where L1, L2, and L3 represent different levels of categorization based on predefined hierarchical structures.
                            7. Extract each product's brand name, product's price, discount %, Discounted Price (must be less than Product's Price), seller, quantity, and L1, L2, L3 categories carefully.
                            8. Do not skip any product details.
        
                            Here are the examples: 
                            ####
                                Product name : [$|Change , $|Vishudh Regular Fit Women White Trousers , $|Size: M , $|4.3, $|•, $|(87), $|Out Of Stock , $|Find Similar]
                                output : {{
                                          "product1": {{
                                            "Brand name": "Vishudh",
                                            "Product Name": "Vishudh Regular Fit Women White Trousers",
                                            "Price": "87",
                                            "Discount": "N/A",
                                            "Discounted Price": "N/A",
                                            "Seller": "N/A",
                                            "Quantity": "Out Of Stock",
                                            "L1 category": "Clothing",
                                            "L2 category": "Women's Clothing",
                                            "L3 category": "Trousers"
                                          }}
                                        }}
                            Constraints:
        
                            A. Include only e-commerce relevant items or services.In Product name extract only product nothing else. If no relevant products are extracted, return an empty dictionary: {{}}.
                            B. Provide comprehensive information for each product under its unique key, combining all details such as name, price, offers, and specifications.
                            C. Avoid general categories like 'electronics' or 'clothing'; focus on specific product names and details.
                            D. Do not include duplicate product entries in the final response.
                            E. Exclude non-product related content such as descriptions of payment methods or general security features.
                            F. Do not include any notes, explanations, or text with the JSON response.
        
                            Final JSON Response:
        
                            The output should list each specific product mentioned in the text under its own key, combining all details. The format should be:
        
                            example : {{
                                  "product1": {{
                                    "Brand name": "Ambrane",
                                    "Product Name": "Ambrane 27000 mAh Power Bank (18 W, Fast Charging) Black, Lithium Polymer",
                                    "Price": "1,999",
                                    "Discount": "42% off",
                                    "Discounted Price": "1,159.42",
                                    "Seller": "TBsmart",
                                    "Quantity": "1",
                                    "L1 category": "Electronics",
                                    "L2 category": "Mobile Accessories",
                                    "L3 category": "Power Banks"
                                  }},
                                  "product2": {{
                                    "Brand name": "REDMI",
                                    "Product Name": "REDMI 20000 mAh Power Bank (18 W, Fast Charging) Black, Lithium Polymer",
                                    "Price": "1,949",
                                    "Discount": "39% off",
                                    "Discounted Price": "1,188.89",
                                    "Seller": "RetailNet",
                                    "Quantity": "1",
                                    "L1 category": "Electronics",
                                    "L2 category": "Mobile Accessories",
                                    "L3 category": "Power Banks"
                                  }}
                            }}
        
                            Return only the JSON object. Do not include any text, notes, or explanations outside or before the JSON response. The output must be pure JSON.
                            ####
                            If there is no product in the input data then do not put example data in the output. It's a serious issue.
                            ####
                            """
    
                    messages = [{"role": "system", "content": self.system_prompt},{"role": "user", "content": combined_prompt}]
                    prompt = self.Pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    outputs = self.Pipeline(prompt,max_new_tokens=8000,do_sample=True,temperature=0.2,top_p=0.9,)
                    _out = outputs[0]["generated_text"][len(prompt):]
                    
                    _out = _out.replace("\n\n","").replace("assistant","").replace("\n","").replace("\\","")
                    match = re.search(r'({.*})', _out, re.DOTALL)
        
                    if match:
                        json_data = match.group(1)
                    else:
                        json_data = {}
                    raw_data.at[index, "parsed_output"] = json_data
                    print(json_data)
                    # break
                else:
                    raw_data.at[index, "parsed_output"] = {}
            except Exception as e:
                print("Exception occured: ", e)
                raw_data.at[index, "parsed_output"] = {}
        
                
        file_name = f"{self.nfs_volume}/cart_parsed-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
        raw_data.to_csv(file_name, index=False)
        
        return {"saved_file_location":file_name}
    def finalize(self):
        self.Pipeline = None
