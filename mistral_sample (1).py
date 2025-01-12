from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Now I am in the model run")



# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the Hugging Face model
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)



# Load model with efficient settings
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
).to(device)




# Define a function for text generation
def lm_mistral_generate_text(prompt, max_new_tokens=3000):
    torch.cuda.empty_cache()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
                             input_ids,
                             max_new_tokens=max_new_tokens,
   
                             temperature=0.7,                # Adjust creativity (lower for focused answers)
                             top_p=0.9,                      # Top-p sampling for diverse responses
                             do_sample=True                  # Enable sampling (instead of greedy decoding)
    )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the beginning of the output
    output_text = full_output[len(prompt):].strip()
    torch.cuda.empty_cache()

    
    return output_text

# Generate output
#prompt = "give python code to generate images from text "
#output = generate_text(prompt)
#print("Generated Output:", output)




