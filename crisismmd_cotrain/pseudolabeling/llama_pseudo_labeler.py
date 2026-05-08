import pandas as pd
import json
import torch
import sys
import os
import re
import time
import gc
import json
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from huggingface_hub import login

MSG_PRINTED = False

HUMA_DF_PROMPT = (
    "You are an expert in disaster response and humanitarian aid data analysis. "
    "Examine the given text and image carefully and classify them into exactly one of these categories (0-4). "
    "Respond with ONLY the number, no other text or explanation.\n\n"
    "Categories:\n"
    "0: HUMAN IMPACT - Must be about PEOPLE who are clearly affected by the disaster: injured, displaced, "
    "evacuated, in temporary shelters, or waiting in lines for aid. People must be visibly impacted.\n"
    
    "1: RESPONSE EFFORTS - Must be about active RESCUE operations, aid distribution, medical treatment, "
    "VOLUNTEER work, DONATION, or evacuation in progress. Look for emergency responders, relief workers, or "
    "organized aid activities.\n"
    
    "2: INFRASTRUCTURE DAMAGE - Must be about clear physical damage to buildings, roads, bridges, power lines, "
    "VEHICLES or other structures that was caused by the disaster. The damage should be obvious and significant.\n"
    
    "3: OTHER CRISIS INFO - Must be about verified crisis-related content that doesn't fit above categories: "
    "maps of affected areas, emergency warnings, official updates, or documentation of crisis conditions. "
    "Must have clear connection to the current disaster.\n"
    
    "4: NOT CRISIS-RELATED - Use this for:\n"
    "- Images and text where you're unsure if they are related to the crisis\n"
    "- General texts and photos that could be from any time/place\n" 
    "- Texts and images without clear crisis impact or response\n"
    "- Texts are not related to a crisis with stock photos or promotional images\n"
    "- Any text and image that doesn't definitively fit categories 0-3\n\n"
    
    "Important:\n"
    "- If there's ANY sign of rescue or donation, pick 1.\n"
    "- If there's ANY sign of damage, pick 2.\n"
    "- If there's ANY sign of obviously distressed or harmed people, pick 0.\n"
    "- If the text and image are definitely about a crisis but you DO NOT see rescue/damage/impacted people, pick 3.\n"
    "- Otherwise, pick 4. Also, when you are not sure which number to pick, pick 4.\n"
    "Answer with just a single digit (0–4)."
)

HUMA_TEXT_ONLY_PROMPT = (
    "You are an expert in disaster response and humanitarian aid data analysis. "
    "Examine this text delimited by triple quotes (\"\"\" \"\"\") carefully and classify it into exactly one of these categories (0-4). "
    "Respond with ONLY the number, no other text or explanation.\n\n"
    "Categories:\n"
    "0: HUMAN IMPACT - Must show direct human suffering or hardship:\n"
    "- Deaths, injuries, or missing people\n"
    "- People struggling without basic needs (food, water, shelter)\n"
    "- Displaced or evacuated people\n"
    "- Personal stories of survival or loss\n"
    "- People stranded or waiting for rescue\n"
    
    "1: RESPONSE EFFORTS - Any organized help effort, no matter how small:\n"
    "- Rescue operations and emergency response\n"
    "- Aid collection or distribution activities\n"
    "- Donations of money, supplies, or services\n"
    "- Volunteer work and relief efforts\n"
    "- Medical assistance\n"
    "- Fundraising events for disaster relief\n"
    
    "2: INFRASTRUCTURE DAMAGE - Must describe specific physical destruction:\n"
    "- Destroyed or damaged buildings and homes\n"
    "- Damaged roads, bridges, or transportation systems\n"
    "- Disrupted power lines or water systems\n"
    "- Damaged vehicles or equipment\n"
    "- Before/after comparisons showing destruction\n"
    
    "3: CRISIS UPDATES - Must be specific to the crisis but not fit above categories:\n"
    "- Weather forecasts and disaster warnings\n"
    "- Maps or descriptions of impact areas\n"
    "- Official announcements about the disaster\n"
    "- Statistics and data about crisis impact\n"
    "- Crisis reporting without specific damage/casualties/response\n"
    
    "4: NOT CRISIS-RELATED - Use when no other category clearly fits:\n"
    "- General discussion without crisis specifics\n"
    "- Personal opinions about non-crisis aspects\n"
    "- Promotional or commercial content\n"
    "- Unclear connection to crisis\n"
    "- Content that could apply to non-crisis situations\n\n"
    
    "Important Decision Rules:\n"
    "- If you see ANY mention of help, rescue, or donations → Pick 1\n"
    "- If you see ANY mention of human casualties, suffering, or displacement but not related to volunteer, rescue, donation... → Pick 0\n"    
    "- If you see ANY specific physical destruction of properties → Pick 2\n"
    "- If it's clearly about the crisis but doesn't fit 0-2 → Pick 3\n"
    "- If multiple categories could apply, use the one BEST FITS the text\n"
    "- Only use 4 when you are COMPLETELY SURE no other category fits\n"
    "Answer with just a single digit (0–4)."
)

HUMA_IMAGE_ONLY_PROMPT = (
    "You are an expert in disaster response and humanitarian aid image analysis. "
    "Examine the first image carefully and classify it into exactly one of these categories (0-4). "
    "Respond with ONLY the number, no other text or explanation.\n\n"
    "Categories:\n"
    "0: HUMAN IMPACT - Must show PEOPLE who are clearly affected by the disaster: injured, displaced, "
    "evacuated, in temporary shelters, or waiting in lines for aid. People must be visibly impacted.\n"
    
    "1: RESPONSE EFFORTS - Must show active RESCUE operations, aid distribution, medical treatment, "
    "VOLUNTEER work, DONATION, or evacuation in progress. Look for emergency responders, relief workers, or "
    "organized aid activities.\n"
    
    "2: INFRASTRUCTURE DAMAGE - Must show clear physical damage to buildings, roads, bridges, power lines, "
    "VEHICLES or other structures that was caused by the disaster. The damage should be obvious and significant.\n"
    
    "3: OTHER CRISIS INFO - Shows verified crisis-related content that doesn't fit above categories: "
    "maps of affected areas, emergency warnings, official updates, or documentation of crisis conditions. "
    "Must have clear connection to the current disaster.\n"
    
    "4: NOT CRISIS-RELATED - Use this for:\n"
    "- Images where you're unsure if it's related to the crisis\n"
    "- General photos that could be from any time/place\n" 
    "- Images without clear crisis impact or response\n"
    "- Stock photos or promotional images\n"
    "- Any image that doesn't definitively fit categories 0-3\n\n"
    
    "Important:\n"
    "- If there's ANY sign of rescue or donation, pick 1.\n"
    "- If there's ANY sign of damage, pick 2.\n"
    "- If there's ANY sign of obviously distressed or harmed people, pick 0.\n"
    "- If it's definitely about a crisis but you DO NOT see rescue/damage/impacted people, pick 3.\n"
    "- Otherwise, pick 4. Also, when you are not sure which number to pick, pick 4.\n"
    "Answer with just a single digit (0–4)."
)

HUMA_TEXT_IMAGE_PROMPT = (
    "You are an expert in disaster response and humanitarian aid data analysis. "
    "Examine the given text and image carefully and classify them into exactly one of these categories (0-4). "
    "Respond with ONLY the number, no other text or explanation.\n\n"
    "Categories:\n"
    "0: HUMAN IMPACT - Must be about PEOPLE who are clearly affected by the disaster: injured, displaced, "
    "evacuated, in temporary shelters, or waiting in lines for aid. People must be visibly impacted.\n"
    
    "1: RESPONSE EFFORTS - Must be about active RESCUE operations, aid distribution, medical treatment, "
    "VOLUNTEER work, DONATION, or evacuation in progress. Look for emergency responders, relief workers, or "
    "organized aid activities.\n"
    
    "2: INFRASTRUCTURE DAMAGE - Must be about clear physical damage to buildings, roads, bridges, power lines, "
    "VEHICLES or other structures that was caused by the disaster. The damage should be obvious and significant.\n"
    
    "3: OTHER CRISIS INFO - Must be about verified crisis-related content that doesn't fit above categories: "
    "maps of affected areas, emergency warnings, official updates, or documentation of crisis conditions. "
    "Must have clear connection to the current disaster.\n"
    
    "4: NOT CRISIS-RELATED - Use this for:\n"
    "- Images and text where you're unsure if they are related to the crisis\n"
    "- General texts and photos that could be from any time/place\n" 
    "- Texts and images without clear crisis impact or response\n"
    "- Texts are not related to a crisis with stock photos or promotional images\n"
    "- Any text and image that doesn't definitively fit categories 0-3\n\n"
    
    "Important:\n"
    "- If there's ANY sign of rescue or donation, pick 1.\n"
    "- If there's ANY sign of damage, pick 2.\n"
    "- If there's ANY sign of obviously distressed or harmed people, pick 0.\n"
    "- If the text and image are definitely about a crisis but you DO NOT see rescue/damage/impacted people, pick 3.\n"
    "- Otherwise, pick 4. Also, when you are not sure which number to pick, pick 4.\n"
    "Answer with just a single digit (0–4)."
)

INFO_DF_PROMPT = (
    "You are an expert in disaster response and humanitarian aid data analysis. "
    "Examine the given text and image carefully and classify them into exactly one of these categories (0-4). "
    "Respond with ONLY the number, no other text or explanation.\n\n"
    "Categories:\n"
    "0: HUMAN IMPACT - Must be about PEOPLE who are clearly affected by the disaster: injured, displaced, "
    "evacuated, in temporary shelters, or waiting in lines for aid. People must be visibly impacted.\n"
    
    "1: RESPONSE EFFORTS - Must be about active RESCUE operations, aid distribution, medical treatment, "
    "VOLUNTEER work, DONATION, or evacuation in progress. Look for emergency responders, relief workers, or "
    "organized aid activities.\n"
    
    "2: INFRASTRUCTURE DAMAGE - Must be about clear physical damage to buildings, roads, bridges, power lines, "
    "VEHICLES or other structures that was caused by the disaster. The damage should be obvious and significant.\n"
    
    "3: OTHER CRISIS INFO - Must be about verified crisis-related content that doesn't fit above categories: "
    "maps of affected areas, emergency warnings, official updates, or documentation of crisis conditions. "
    "Must have clear connection to the current disaster.\n"
    
    "4: NOT CRISIS-RELATED - Use this for:\n"
    "- Images and text where you're unsure if they are related to the crisis\n"
    "- General texts and photos that could be from any time/place\n" 
    "- Texts and images without clear crisis impact or response\n"
    "- Texts are not related to a crisis with stock photos or promotional images\n"
    "- Any text and image that doesn't definitively fit categories 0-3\n\n"
    
    "Important:\n"
    "- If there's ANY sign of rescue or donation, pick 1.\n"
    "- If there's ANY sign of damage, pick 2.\n"
    "- If there's ANY sign of obviously distressed or harmed people, pick 0.\n"
    "- If the text and image are definitely about a crisis but you DO NOT see rescue/damage/impacted people, pick 3.\n"
    "- Otherwise, pick 4. Also, when you are not sure which number to pick, pick 4.\n"
    "Answer with just a single digit (0–4)."
)

INFO_TEXT_ONLY_PROMPT = """ You are an AI model that classifies the given text into one of two categories based on its informational value. 
Your task is to analyze the given text to determine whether it contains relevant or informative content about a crisis.
Our goal is to collect as much useful information as possible about crises, so your classification should prioritize identifying texts that provide relevant details, even if they are brief or incomplete. 
Avoid being overly restrictive. If the text has any relevant crisis-related information, classify it as informative.
Classify the text delimited by triple quotes (\"\"\" \"\"\") into one of the following categories:
  - **1 (positive):** The text provide details, updates, or any relevant information about a crisis.
  - **0 (not positive):** The text do not contain relevant details or information about a crisis.
Strictly return only the classification label (1 or 0) without any extra text or explanation.
"""   

INFO_IMAGE_ONLY_PROMPT = """Does the image give useful information that could help during a crisis?
Respond with '1' if this image provides any information or details about a crisis, and '0' if it does not.

Instructions: 
  - You should prioritize identifying texts that provide relevant details, even if they are brief or incomplete.
  - Avoid being overly restrictive. If the text has any relevant crisis-related information, response with '1'.
  - When the meaning of the image is unclear, response with '0'.
  - Do not output any extra text.
"""

INFO_TEXT_IMAGE_PROMPT = """Do the given text and the given image give useful information that could help during a crisis?
Respond with '1' if the text and the image provide any information or details about a crisis, and '0' if they do not.

Instructions: 
  - You should prioritize identifying texts and the images that provide relevant details, even if they are brief or incomplete.
  - Avoid being overly restrictive. If the text and image have any relevant crisis-related information, response with '1'.
  - When the meaning of the image and the text are unclear, response with '0'.
  - Do not output any extra text.
"""

# Function to detect the match patterns in the model output
def detect_informative_label(text):
    # Check for "not provide any information" first to avoid partial matches
    if re.search(r"\bnot (provide |give |contain |related |relevant )?any information\b", text, re.IGNORECASE) or re.search(r"\b0[.!]?\b", text):
        print("Analyze successfully! This entry is not informative!")
        return 0
    elif re.search(r"\b(provides |provide |gives |give )(useful |some |relevant )?information\b", text, re.IGNORECASE)  or re.search(r"\b1[.!]?\b", text):
        print("Analyze successfully! This entry is informative!")
        return 1
    print("Analyze completed! No matching label found! Assigned as not informative by default!") 
    return 0  # Return 0 if neither phrase is found

# Function to detect the match patterns in the model output
def detect_humanitarian_label(text):
    # Clean all symbols
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    
    # First, check for "not provide any information" or the digit "4"
    if re.search(r"\b(not (provide|give|contain)|4|\*\*4\*\*)\b", text, re.IGNORECASE):
        print("  Analyzed successfully! This entry is 4: not humanitarian!")
        return 4

    # Check for case "0": Ensure the text contains "people" and one of "impact", "impacted", or "affected", and "crisis"
    elif (
        re.search(r"\bpeople\b", text, re.IGNORECASE) and re.search(r"\bcrisis\b", text, re.IGNORECASE)
    ) or (
        re.search(r'(?=.*\banswer|response|number\b)(?=.*\b0\b)', text, re.IGNORECASE)
    ) or (
        re.search(r"\b\*\*0\*\*\b", text, re.IGNORECASE)
    ):
        print("  Analyzed successfully! This entry is 0: affected individuals!")
        return 0

    # Check for rescue, volunteering, donation efforts, or the digit "1"
    elif re.search(r"\b(rescue|volunteering|donation|1|\*\*1\*\*)\b", text, re.IGNORECASE):
        print("  Analyzed successfully! This entry is 1: rescue, volunteering, or donation efforts!")
        return 1

    # Check for infrastructure, buildings, utilities, vehicles, or the digit "2"
    elif re.search(r"\b(infrastructure|building|utility|vehicle|car|2|\*\*2\*\*)\b", text, re.IGNORECASE):
        print("  Analyzed successfully! This entry is 2: infrastructure and utility damage!")
        return 2

    # Check for "other" category or the digit "3"
    elif re.search(r"\b(other|3|\*\*3\*\*)\b", text, re.IGNORECASE):
        print("  Analyzed successfully! This entry is 3: other relevant information!")
        return 3

    # Default case: No match found
    print("  Analyze completed! No matching label found! Assigned as 4: not humanitarian by default!")
    return 4  # Default to 4 if no label matches

def load_few_shot_examples(devdata_path, few_shot_num, use_images):
    """Loads few-shot examples with unique labels from the dev set.
    
    If few_shot_num=1, it returns an array where each entry is randomly selected from a different label.
    If few_shot_num > 1, it returns `few_shot_num` examples per label (if available).
    """
    if not os.path.isfile(devdata_path):
        raise FileNotFoundError(f"Dev file not found: {devdata_path}")
    
    if devdata_path.endswith(".json"):
        dev_data = pd.read_json(devdata_path, lines=False, encoding="utf-8")
    elif devdata_path.endswith(".csv"):
        dev_data = pd.read_csv(devdata_path, encoding="utf-8-sig")
    elif devdata_path.endswith(".tsv"):
        dev_data = pd.read_csv(devdata_path, sep='\t')
    else:
        raise ValueError("Dev dataset must be .json .tsv or .csv")
    
    required_columns = ['tweet_text', 'label']
    if use_images:
        required_columns = ['tweet_text', 'image', 'label']

    if not all(column in dev_data.columns for column in required_columns):
        raise ValueError("Dev data must contain columns: text, image_path, label")
    
    # Select `few_shot_num` random examples per label
    label_samples = dev_data.groupby("label").apply(lambda x: x.sample(n=min(few_shot_num, len(x)), random_state=42)).reset_index(drop=True)

    return label_samples.to_dict(orient="records")

def load_and_process_image(image_path):
    """Loads and processes an image (returns a PIL Image) for the model."""
    assert os.path.isfile(image_path), f"File not found: {image_path}"
    return Image.open(image_path).convert("RGB")


def construct_message_template(few_shot_examples, prompt, test_text=None, test_image=None, use_texts=True, use_images=True, few_shot_num=0):
    """
    Constructs the structured message format with both few-shot examples and test case.
    
    Example (Few-shot case):
    [
      {"role": "user", "content": [{"type": "text", "text": "<Prompt> Here are sample data with labels:"}]}, 
      {"role": "user", "content": [{"type": "image", "image": sample_image_1}, {"type": "text", "text": sample_text_1}]},
      {"role": "assistant", "content": label_1},
      ....
      {"role": "user", "content": [{"type": "text", "text": "Now classify this test data:"}, {"type": "image", "image": test_image}]}
    ]

    Example (Zero-shot case):
    [
      {"role": "user", "content": [{"type": "text", "text": "<Prompt>"}]},
      {"role": "user", "content": [{"type": "text", "text": "Now classify this test image:"}, {"type": "image", "image": test_image}]}
    ]
    """
    """messages = [
        {
            "role": "system",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{prompt}.\nHere are some examples:\n"}
            ],
        }
    ]"""

    messages = []

    def ordinal(n):
        """Convert an integer to its ordinal representation (e.g., 2 -> 2nd, 3 -> 3rd, 4 -> 4th, 11 -> 11th)."""
        if 10 <= n % 100 <= 20:  # Special case for 11th, 12th, 13th, etc.
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    user_content = []
    txt_prompt = ""

    # Add few-shot examples if available
    if few_shot_num > 0 and few_shot_examples:          
        for index, example in enumerate(few_shot_examples):
            #user_content = []
            txt_prompt = "Below is an example for your reference:\n"
            if use_texts and use_images: # text + image
                user_content.append({"type": "image", "image": example["image"]})
                #user_content.append({"type": "text", "text": f"\nText: \"\"\"{example['text']}\"\"\"\nThe category of this sample text and image is: "})
                txt_prompt = f"Above is the sample image, and here is the Text: \"\"\"{example['tweet_text']}\"\"\"\nThe category of this sample text and image is: "
            elif use_texts: # text only
                #user_content.append({"type": "text", "text": f"\n{prompt}\nText: \"\"\"{example['text']}\"\"\"\nThe category of this sample text is: "})   
                txt_prompt = f"Text: \"\"\"{example['tweet_text']}\"\"\"\nThe category of this sample text is: "
            else: # image only
                user_content.append({"type": "image", "image": example["image"]})
                #user_content.append({"type": "text", "text": f"\n{prompt}\nThe category of this sample image is: "})
                txt_prompt = f"Above is the sample image. The category of this sample image is: "
            txt_prompt += str(example['label'])
            user_content.append({"type": "text", "text": txt_prompt})    
            #messages.append({"role": "user", "content": user_content})
            #messages.append({"role": "assistant", "content": str(example['label'])})

    #user_content = []

    # Adding the test data to the message
    if use_texts and use_images:
        user_content.append({"type": "image", "image": test_image})
        #user_content.append({"type": "text", "text": f"\n{prompt}\nText: \"\"\"{test_text}\"\"\"\n\nThe category of this Text and the above image is: "})
        txt_prompt = f"{prompt}\nAbove is the test image, and here is the test Text that you need to classify: \"\"\"{test_text}\"\"\". The category of this test image and test Text is: "
    elif use_texts:
        #user_content.append({"type": "text", "text": f"\n{prompt}\nText: \"\"\"{test_text}\"\"\"\n\nThe category of this test Text is:"}) 
        txt_prompt = f"{prompt}\nHere is the test Text that you need to classify: \"\"\"{test_text}\"\"\". The category of this test Text is: "
    else:
        user_content.append({"type": "image", "image": test_image})
        #user_content.append({"type": "text", "text": f"\n{prompt}\n\nThe category of this image is:"})     
        txt_prompt = f"{prompt}\nAbove is the test image that you need to classify. The category of this test image is: "

    user_content.append({"type": "text", "text": txt_prompt})
    messages.append({"role": "user", "content": user_content})

    #print("\n\n", messages, "\n\n")
    return messages

def classify_raw_text(raw_text, task="informative"):
    # Basic parsing: split on the word "assistant" to isolate answer
    parts = raw_text.split("assistant")
    
    if len(parts) > 1:
        final_answer = parts[-1].strip()
        if task=="informative":
            if final_answer.startswith("1"):
                return 1
            elif final_answer.startswith("0"):
                return 0
            else:
                # If it doesn't strictly start with '1' or '0', handle as needed
                print(f"Unclear model output: {final_answer}! Trying to analyze the output again...")
                # Trying to look for expected output patterns
                return detect_informative_label(final_answer)
        if task=="humanitarian":
            if final_answer.startswith("0"):
                return 0
            elif final_answer.startswith("1"):
                return 1
            elif final_answer.startswith("2"):
                return 2
            elif final_answer.startswith("3"):
                return 3
            elif final_answer.startswith("4"):
                return 4
            else:
                # If it doesn't strictly start with '1' or '0', handle as needed
                print(f"\nUnclear model output: {final_answer}!\n  Trying to analyze the output again...")
                # Trying to look for expected output patterns
                return detect_humanitarian_label(final_answer)
    else:
        print(f"Could not parse output: {raw_text}")
        return None
    
def predict(model, processor, prompt, task="humanitarian", 
            use_texts=True, use_images=True, 
            text=None, image_path=None, 
            few_shot_examples=None, few_shot_num=0):
    """
    Predicts using the LLaMA Vision model with properly structured messages.
    Handles text-only, image-only, and text+image few-shot scenarios.
    """
    global MSG_PRINTED

    # Load test image if available
    test_image = load_and_process_image(image_path) if image_path else None
    all_images = []

    # Ensure few-shot examples exist before accessing
    if use_images: # test with image(s)        
        if few_shot_num > 0 and few_shot_examples: # not zeroshot
            for example in few_shot_examples:
                all_images.append(example["image"])       
            if test_image:
                all_images.append(test_image)  # Ensure test image is included    

        elif test_image: # zeroshot with valid image
            all_images = test_image
        else:
            all_images = None
        
    """
    if few_shot_examples and use_images:
        all_images = [example["image"] for example in few_shot_examples if "image" in example]
    else:
        all_images = []

    if test_image:
        all_images.append(test_image)  # Ensure test image is included"""

    # Construct the structured message
    
    messages = construct_message_template(
        few_shot_examples=few_shot_examples,
        prompt=prompt,
        test_text=text,
        test_image=test_image,
        use_texts=use_texts,
        use_images=use_images,
        few_shot_num=few_shot_num
    )

    if not MSG_PRINTED:
        # Function to serialize images only for printing
        def json_serializable(obj):
            if isinstance(obj, Image.Image):
                return f"<PIL.Image mode={obj.mode} size={obj.size}>"
            raise TypeError(f"Type {type(obj)} is not JSON serializable")
        print("\n\n", '-'*80)
        # Print formatted JSON
        print(json.dumps(messages, indent=2, ensure_ascii=False, default=json_serializable))
        print('-'*80, "\n\n")
        MSG_PRINTED = True

    # Apply chat template to format the input
    formatted_input = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Prepare input tensors
    if use_images:
        inputs = processor(
            text=formatted_input,
            images=all_images,
            return_tensors="pt",
            truncation=False,
            #max_length=20480
        ).to(model.device)
    else:
        inputs = processor(
            text=formatted_input,
            return_tensors="pt",
            truncation=False,
            #max_length=20480
        ).to(model.device)

    # Generate prediction
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            min_new_tokens=1,
            do_sample=False,    # Deterministic generation, temperature ignored
            num_beams=1,        # Simple greedy decoding
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )

    # Decode model output
    raw_text = processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    #print(f"\n\n raw_text = {raw_text}\n\n")
    
    return classify_raw_text(raw_text, task)



def update_remaining_time(start_time, completed_tests, total_tests, checkpoint_label):
    """
    Calculates and prints the estimated remaining time.
    
    Parameters:
    - start_time (float): The timestamp when the tests started.
    - completed_tests (int): Number of completed tests.
    - total_tests (int): Total number of tests.
    - checkpoint_label (str): A message indicating the progress percentage.
    """
    elapsed_time = time.time() - start_time
    avg_time_per_test = elapsed_time / completed_tests
    remaining_time = avg_time_per_test * (total_tests - completed_tests)

    print(f"\n📊 {checkpoint_label} completed. Estimated remaining time: {remaining_time / 60:.2f} minutes.\n")

def test_model(
    model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    task="humanitarian",
    use_texts=True,
    use_images=True,
    prompt=HUMA_DF_PROMPT,
    testdata_path="dataset/informative/info_text_image_test.json",
    devdata_path="dataset/informative/info_text_image_dev.json",
    result_path="test_results/informative/Llama-3.2-11B-Zeroshot.json",
    few_shot_num=0,
    consistency=False
):
    """
    Loads the model, predicts on test data, and saves results.
    
    Handles:
    - Zero-shot, few-shot (if few_shot_num > 0)
    - Text-only, image-only, text+image settings
    - Structured input with user-assistant messages
    """

    testdata_path = os.path.normpath(testdata_path)
    result_path = os.path.normpath(result_path)

    image_prefix = "data/CrisisMMD_v2.0/"
    
    print(f"Model={model_id}, testdata={testdata_path}, result={result_path}\n")

    if not os.path.isfile(testdata_path):
        raise FileNotFoundError(f"Test file not found: {testdata_path}")
    
    if result_path:
        result_dir = os.path.dirname(result_path)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    print(f"Loading Llama-Vision model from {model_id} ...")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    print(f"Loading test dataset from {testdata_path}")
    if testdata_path.endswith(".json"):
        test_data = pd.read_json(testdata_path, lines=False, encoding="utf-8", orient="index")
    elif testdata_path.endswith(".csv"):
        test_data = pd.read_csv(testdata_path, encoding="utf-8-sig")
    elif testdata_path.endswith(".tsv"):
        test_data = pd.read_csv(testdata_path, sep='\t')
    else:
        raise ValueError("Test dataset must be .json .csv or .tsv")

    required_columns = ['tweet_text', 'label']
    if use_images:
        required_columns = ['tweet_text', 'image_path', 'label']

    #print(test_data.columns)
    
    if not all(column in test_data.columns for column in required_columns):
        raise ValueError(f"Test data must contain {required_columns}")

    # Load few-shot examples (only once if consistency=True)
    if few_shot_num > 0:
        few_shot_examples = load_few_shot_examples(devdata_path, few_shot_num, use_images)
        for example in few_shot_examples:
            if use_images:
                example["image"] = load_and_process_image(image_prefix + example["image"])
            else:
                example["image"] = None
    else:
        few_shot_examples = None

    results = []
    y_true_list = []
    y_pred_list = []
    total_tests = len(test_data)
    print(f"Total test cases: {total_tests}")

    if task == "informative":
        label_mapping = {
            "informative": 0,
            "not_informative": 1,
        }
    elif task == "humanitarian":
        label_mapping = {
            "affected_individuals": 0, 
            "rescue_volunteering_or_donation_effort": 1,
            "infrastructure_and_utility_damage": 2,
            "other_relevant_information": 3,
            "not_humanitarian": 4,
        }

    for index, row in test_data.iterrows():
        y_true = label_mapping[row["label"]]
        text = row["tweet_text"]
        image_path = None

        if use_images:
            image_path = image_prefix + row["image_path"]

        # If few-shot examples should vary per test case, reload new samples
        if (few_shot_num > 0 and not consistency):
            few_shot_examples = load_few_shot_examples(devdata_path, few_shot_num)
            for example in few_shot_examples:
                if use_images:
                    example["image"] = load_and_process_image(image_prefix + example["image"])

        print(f"Processing {index}/{total_tests}...")

        y_pred = predict(
            model=model,
            processor=processor,
            prompt=prompt,
            task=task,
            use_texts=use_texts,
            use_images=use_images,
            text=text,
            image_path=image_path,
            few_shot_examples=few_shot_examples,
            few_shot_num=few_shot_num
        )

        # Time tracking at checkpoints
        if index == total_tests // 100:  # After 1% checkpoint
            update_remaining_time(start_time, index + 1, total_tests, "1% of tests")
        elif index == total_tests // 4:  # 25% checkpoint
            update_remaining_time(start_time, index + 1, total_tests, "25% of tests")
        elif index == total_tests // 2:  # 50% checkpoint
            update_remaining_time(start_time, index + 1, total_tests, "50% of tests")
        elif index == (3 * total_tests) // 4:  # 75% checkpoint
            update_remaining_time(start_time, index + 1, total_tests, "75% of tests")

        if y_pred is None:  # Ignore invalid results
            print(f"\nBad result! Ignore this test case and move on!\n")
            continue

        results.append({
            "id": index,
            "event_name": row["event_name"],
            "tweet_id": row["tweet_id"],
            "tweet_text": text,
            "ori_label": y_true,
            "gen_label": y_pred,
        })

        if use_images:
            results.append({
                "image_id": row["image_id"],
                "image_path": row["image_path"],
            })
        
        y_pred_list.append(y_pred)
        y_true_list.append(y_true)

    # Save results
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nPerformed {len(results)} test cases. Predictions saved to {result_path}")

    # Calculate & display test statistics
    if y_true_list:
        accuracy = accuracy_score(y_true_list, y_pred_list)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_list,
            y_pred_list,
            average="weighted"
        )

        print("\n📊 **Test Statistics**")
        print(f"✔ Accuracy: {accuracy:.4f}")
        print(f"✔ Precision: {precision:.4f}")
        print(f"✔ Recall: {recall:.4f}")
        print(f"✔ F1 Score: {f1:.4f}")

        # Save statistics
        stats = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        stats_path = result_path.replace(".json", "_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Test statistics saved to {stats_path}")

    # Clear GPU memory
    with torch.no_grad():
        torch.cuda.empty_cache()

    # Optional memory cleanup
    with torch.no_grad():
        torch.cuda.empty_cache()  # Clear unused GPU memory after inference

    return model

def main():

    model_name="meta-llama/Llama-3.2-11B-Vision-Instruct"
    task_name="humanitarian"
    few_shot_num=0
    use_cheyenne=True
    use_beocat=False
    use_texts=True
    use_images=True
    
    if task_name == "humanitarian":
        if use_texts and use_images:
            prompt_to_be_used = HUMA_TEXT_IMAGE_PROMPT
        elif use_texts:
            prompt_to_be_used = HUMA_TEXT_ONLY_PROMPT
        elif use_images:
            prompt_to_be_used = HUMA_IMAGE_ONLY_PROMPT
        else:
            prompt_to_be_used = HUMA_DF_PROMPT

    if task_name == "informative":
        if use_texts and use_images:
            prompt_to_be_used = INFO_TEXT_IMAGE_PROMPT
        elif use_texts:
            prompt_to_be_used = INFO_TEXT_ONLY_PROMPT
        elif use_images:
            prompt_to_be_used = INFO_IMAGE_ONLY_PROMPT
        else:
            prompt_to_be_used = INFO_DF_PROMPT


    if use_cheyenne:
        test_data_path=f"data/CrisisMMD_Modified/{task_name}/image_only.json"
        dev_data_path=f"data/crisismmd_splits/task_{task_name}_text_img_agreed_lab_dev.tsv"
        result_file_path=f"data/llm_pseudo_labels/{task_name}/Llama-3.2-11B-Vision-Instruct-Huma-Zeroshot-Text-Image.json" 
    
    if use_beocat:
        test_data_path=f"data/CrisisMMD_Modified/{task_name}/image_only.json"
        dev_data_path=f"data/crisismmd_splits/task_{task_name}_text_img_agreed_lab_dev.tsv"
        result_file_path=f"data/llm_pseudo_labels/{task_name}/Llama-3.2-11B-Vision-Instruct-Huma-Zeroshot-Text-Image.json"
    
    model = test_model(
        model_id=model_name,
        task=task_name,
        use_texts=use_texts,
        use_images=use_images,
        prompt=prompt_to_be_used,
        testdata_path=test_data_path,
        devdata_path=dev_data_path,
        result_path=result_file_path,
        few_shot_num=few_shot_num,
        consistency=True # To determine few-shot samples to be consistent or varying for each prompts
    )


if __name__ == "__main__":

    start_time = time.time()

    # ensure to only use gpu 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(torch.cuda.device_count())  # Should print 1
    print(torch.cuda.current_device())  # Should print 0

    hugging_face_login_key = "<LOGIN_KEY>"

    login(hugging_face_login_key)

    main()

    end_time = time.time()
    print(f"\Start Time time: {start_time}\nEnd time: {end_time} \n")
    print(f"\nTotal elapsed time: {(end_time - start_time) / 60:.2f} minutes.\n")
