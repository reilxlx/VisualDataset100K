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
output_file_questions = r"/data/TrainData/VisualDataset100K/VD_100K_Questions/Qwen2VL7B_Questions-all.jsonl"
image_folder = r"/data/TrainData/VisualDataset100K/VD_100K/"
output_file = r"/data/TrainData/VisualDataset100K/VD_100K_Questions/Qwen2VL7B_Questions.jsonl"

# Create necessary folders if they don't exist
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logging.info(f"Created output directory: {output_dir}")

# Initialize write lock
write_lock = threading.Lock()

def load_processed_set(output_file_questions):
    processed_set = set()
    if os.path.exists(output_file_questions):
        try:
            with open(output_file_questions, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    processed_set.add(data['id'])
            logging.info(f"Loaded {len(processed_set)} processed entries from output_file_questions.")
        except Exception as e:
            logging.error(f"Error reading output_file_questions: {e}")
    else:
        logging.info("output_file_questions does not exist. Starting fresh.")
    return processed_set

def process_image(image_id, processed_set):
    try:
        # Check if already processed
        if image_id in processed_set:
            logging.info(f"Skipping already processed image {image_id}")
            return None

        image_path = os.path.join(image_folder, image_id)

        # Check if image exists
        if not os.path.exists(image_path):
            logging.error(f"Image {image_id} not found in image_folder.")
            return None
        logging.info(f"Processing image {image_id}")

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

            # Question prompt
            question_prompt = "请作为图像内容识别专家，分析所提供的图像并基于其内容提出一个相关问题。问题应直接关联图像，可以涉及整体或局部细节。仅输出问题，不包含其他任何内容。。"

            # Prepare data for API request
            data = {
                "model": "Qwen2-VL-7B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question_prompt
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
                "max_tokens": 300
            }
            json_string = json.dumps(data)
            logging.info(f"Sending request for image {image_id}")
            response = requests.post(api_url, headers=headers, data=json_string)

            if response.status_code == 200:
                response_json = response.json()
                if ('choices' in response_json and response_json['choices'] and
                    'message' in response_json['choices'][0] and
                    'content' in response_json['choices'][0]['message']):
                    question = response_json['choices'][0]['message']['content']
                    logging.info(f"Received response for image {image_id}")
                    output_data = {
                        "id": image_id,
                        "question": question.strip()
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
        logging.error(f"Error processing image {image_id}: {e}")
        return None

def main():
    # Load processed image ids
    processed_set = load_processed_set(output_file_questions)

    # Get list of all images in the image_folder
    all_images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    images_to_process = [img for img in all_images if img not in processed_set]

    total_to_process = len(images_to_process)
    logging.info(f"Total images to process: {total_to_process}")

    if total_to_process == 0:
        logging.info("No new images to process.")
        print("No new images to process.")
        return

    # Start processing
    try:
        with ThreadPoolExecutor(max_workers=2) as executor, \
                open(output_file, 'a', encoding='utf-8') as outfile, \
                open(output_file_questions, 'a', encoding='utf-8') as flash_outfile:

            futures = []
            for image_id in images_to_process:
                future = executor.submit(process_image, image_id, processed_set)
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

                            # Also write to output_file_questions to update processed records
                            flash_outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                            flash_outfile.flush()
                            os.fsync(flash_outfile.fileno())
                            processed_set.add(result['id'])
                        except Exception as e:
                            logging.error(f"Error writing result for image {result['id']}: {e}")
    except Exception as e:
        logging.error(f"Error during processing: {e}")

    print("Image processing complete.")

if __name__ == "__main__":
    main()
