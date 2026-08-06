import json
from openai import OpenAI

# Khởi tạo client kết nối qua Local Router
client = OpenAI(
    base_url="http://localhost:20128/v1",
    api_key="sk-ac17cce818d45be5-f2gyor-f42f21f8",
)


def generate_zero_shot(prompt_text, model_name="ag/gemini-3.6-flash-high"):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Bạn là mô hình chuyên gia gắn nhãn dữ liệu. Trả về kết quả dưới dạng JSON.",
            },
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# Đổi model_name thành ID khớp với cấu hình trong Local Router của bạn
sample_input = (
    "Phân loại cảm xúc câu sau: 'Sản phẩm giao nhanh nhưng đóng gói hơi ẩu.'"
)
result = generate_zero_shot(sample_input, model_name="ag/gemini-3.6-flash-high")

print("Kết quả:", result)

# Lưu kết quả ra file dataset (.jsonl)
dataset_entry = {"input": sample_input, "output": result}

with open("train_dataset.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(dataset_entry, ensure_ascii=False) + "\n")