# main.py
import requests
import json
import PIL.Image
import os
import threading
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import io
# Set up logging
logging.basicConfig(
    filename='processing.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# API Configuration
api_url = "http://192.168.0.1:8000/v1/chat/completions"
headers = {'Content-Type': 'application/json'}

# Define file paths
question_json = R"/data/TrainData/VisualDataset100K/VD_100K_Qwen2VL7B/Qwen2VL7B_Questions.jsonl"
output_file_all = R"/data/TrainData/VisualDataset100K/VD_100K_Qwen2VL2B/Qwen2VL2B_QuestionsAnswers-all.jsonl"
image_folder = R"/data/TrainData/VisualDataset100K/VD_100K/"
output_file = R"/data/TrainData/VisualDataset100K/VD_100K_Qwen2VL2B/Qwen2VL2B_QuestionsAnswers.jsonl"

# Create necessary folders if they don't exist
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logging.info(f"Created output directory: {output_dir}")

# Initialize write lock
write_lock = threading.Lock()

def load_processed_set(output_file_all):
    processed_set = set()
    if os.path.exists(output_file_all):
        try:
            with open(output_file_all, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    processed_set.add((data['id'], data['question']))
            logging.info(f"Loaded {len(processed_set)} processed entries from output_file_all.")
        except Exception as e:
            logging.error(f"Error reading output_file_all: {e}")
    else:
        logging.info("output_file_all does not exist. Starting fresh.")
    return processed_set

def process_line(line, processed_set):
    try:
        # Parse JSON line
        data = json.loads(line)
        image_id = data['id']
        question = data['question']
        
        # Check if already processed
        if (image_id, question) in processed_set:
            logging.info(f"Skipping already processed image {image_id} with question: {question}")
            return None

        image_path = os.path.join(image_folder, image_id)

        # Check if image exists
        if not os.path.exists(image_path):
            logging.error(f"Image {image_id} not found in image_folder.")
            return None
        logging.info(f"Processing image {image_id} with question: {question}")

        try:
            # Open and process image
            img = PIL.Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                logging.info(f"Converted image {image_id} to RGB mode.")

            # Encode the image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            question_new = "Please carefully analyze the visual content provided below and consider all relevant aspects, including but not limited to colors, objects, scenes, texts, and their relationships. Based on the image, provide a comprehensive and accurate response to the question that follows. Ensure that the answer is contextualized and leverages all visual details present in the image. Please provide the response in Chinese and include a detailed explanation or rationale for your answer." + question
            # Send request to the local APIfile
            with open(image_path, 'rb') as image_file:
                data = {
                        "model": "Qwen2-VL-2B-Instruct",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"{question}"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{img_str}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 500
                    }
                json_string = json.dumps(data)
                logging.info(json_string)
                response = requests.post(api_url, headers=headers, data=json_string)
            
            if response.status_code == 200:
                response_json = response.json()
                if 'choices' in response_json and response_json['choices'] and 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']:
                    answer = response_json['choices'][0]['message']['content']
                    logging.info(f"Received response for image {image_id}")
                    output_data = {
                        "id": image_id,
                        "question": question,
                        "answer": answer
                    }
                    img.close()
                    return output_data
                else:
                    logging.error(f"Error processing {image_id}: API returned unexpected format. Skipping...")
                    img.close()
                    return None
            else:
                logging.error(f"API request failed for {image_id} with status code: {response.status_code}")
                img.close()
                return None

        except Exception as e:
            logging.error(f"Error processing {image_id}: {e}")
            try:
                img.close()
            except:
                pass
            return None
    except Exception as e:
        logging.error(f"Error parsing line: {e}")
        return None

def main():
    # Load processed (id, question) set
    processed_set = load_processed_set(output_file_all)

    # Calculate total lines for progress bar
    try:
        with open(question_json, 'r', encoding='utf-8') as infile:
            total_lines = sum(1 for line in infile)
            logging.info(f"Total lines in input file: {total_lines}")
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        return

    try:
        with open(question_json, 'r', encoding='utf-8') as infile:
            # Compute lines to process
            lines_to_process = []
            for line in infile:
                try:
                    data = json.loads(line)
                    key = (data['id'], data['question'])
                    if key not in processed_set:
                        lines_to_process.append(line)
                except Exception as e:
                    logging.error(f"Error parsing line during filtering: {e}")

            total_to_process = len(lines_to_process)
            logging.info(f"Total lines to process after filtering: {total_to_process}")
    except Exception as e:
        logging.error(f"Error during filtering input file: {e}")
        return

    # Start processing
    try:
        with ThreadPoolExecutor(max_workers=4) as executor, \
             open(output_file, 'a', encoding='utf-8') as outfile, \
             open(output_file_all, 'a', encoding='utf-8') as flash_outfile:
            
            futures = []
            for line in lines_to_process:
                future = executor.submit(process_line, line, processed_set)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=total_to_process, desc="Processing Images"):
                result = future.result()
                if result:
                    with write_lock:
                        try:
                            # Write to main output file
                            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                            outfile.flush()
                            os.fsync(outfile.fileno())
                            logging.debug(f"Wrote result for image {result['id']} to output_file.")
                            
                            # Also write to output_file_all to update processed records
                            flash_outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                            flash_outfile.flush()
                            os.fsync(flash_outfile.fileno())
                            processed_set.add((result['id'], result['question']))
                        except Exception as e:
                            logging.error(f"Error writing result for image {result['id']}: {e}")
    except Exception as e:
        logging.error(f"Error during processing: {e}")

    print("Image processing complete.")

if __name__ == "__main__":
    main()
