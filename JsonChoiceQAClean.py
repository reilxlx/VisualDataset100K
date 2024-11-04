import json
import re

def process_jsonl_file(input_file_path, output_file_path):
    successful_lines = []
    unsuccessful_lines = []

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    for line_number, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue  # 跳过空行
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"第 {line_number + 1} 行：无法解析 JSON。")
            unsuccessful_lines.append(line)
            continue

        id_value = obj.get('id')
        qca = obj.get('question_choices_answer')
        if qca is None:
            print(f"第 {line_number + 1} 行：缺少 'question_choices_answer' 字段。")
            unsuccessful_lines.append(line)
            continue

        # 去除首尾空白字符
        qca = qca.strip()

        # 提取代码块中的 JSON 内容
        code_block_pattern = r'```(?:json)?\n(.*?)\n```'
        match = re.search(code_block_pattern, qca, re.DOTALL)
        if not match:
            print(f"第 {line_number + 1} 行：未找到代码块标记。")
            unsuccessful_lines.append(line)
            continue

        json_content = match.group(1).strip()

        try:
            qca_obj = json.loads(json_content)
        except json.JSONDecodeError:
            print(f"第 {line_number + 1} 行：无法解析 'question_choices_answer' 中的 JSON。")
            unsuccessful_lines.append(line)
            continue

        required_fields = ["question", "choiceA", "choiceB", "choiceC", "choiceD", "answer"]
        missing_fields = [field for field in required_fields if field not in qca_obj]
        if missing_fields:
            print(f"第 {line_number + 1} 行：缺少字段：{', '.join(missing_fields)}。")
            unsuccessful_lines.append(line)
            continue

        # 构建新的 JSON 对象
        new_obj = {
            "id": id_value,
            "question": qca_obj["question"],
            "choiceA": qca_obj["choiceA"],
            "choiceB": qca_obj["choiceB"],
            "choiceC": qca_obj["choiceC"],
            "choiceD": qca_obj["choiceD"],
            "answer": qca_obj["answer"]
        }

        # 将新对象添加到成功列表
        successful_lines.append(json.dumps(new_obj, ensure_ascii=False))

    # 将成功的行写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in successful_lines:
            outfile.write(line + '\n')

    # 将未成功的行写回原始文件
    with open(input_file_path, 'w', encoding='utf-8') as infile:
        for line in unsuccessful_lines:
            infile.write(line + '\n')

if __name__ == "__main__":
    input_file = '/root/VD_100K_Qwen2VL72BInt4/Qwen2VL72BInt4_ChoiceQA.jsonl'
    output_file = '/root/VD_100K_Qwen2VL72BInt4/Qwen2VL72BInt4_ChoiceQA-cleaned.jsonl'
    process_jsonl_file(input_file, output_file)
