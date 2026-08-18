import sys
import json
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.config import OUTPUTS_DIR, STUDENT_MODEL_ID
from src.prompts.teacher_prompts import create_student_prompt
from src.utils.json_parser import parse_cot_json
from src.utils.logger import setup_logger

logger = setup_logger("DemoCLI")

DEFAULT_SCHEMA = """Node labels:
- Person {name: STRING, born: INTEGER}
- Movie {title: STRING, released: INTEGER, tagline: STRING}
Relationship types:
- ACTED_IN {roles: LIST} (Person -> Movie)
- DIRECTED {} (Person -> Movie)
- PRODUCED {} (Person -> Movie)"""

def main():
    print("=" * 60)
    print("Text2Cypher Interactive Demo Playground")
    print("=" * 60)

    adapter_path = OUTPUTS_DIR / "qwen2.5_1.5b_student_kd"
    if not adapter_path.exists():
        # Fallback search
        adapters = list(OUTPUTS_DIR.glob("*"))
        if adapters:
            adapter_path = adapters[0]

    if not adapter_path.exists():
        print(f"Lỗi: Chưa có mô hình đã train tại '{OUTPUTS_DIR}'. Hãy chạy Stage 4 trước!")
        return

    print(f"Đang nạp mô hình từ '{adapter_path.name}'...")

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=STUDENT_MODEL_ID,
            max_seq_length=2048,
            load_in_4bit=True
        )
        model.load_adapter(str(adapter_path))
        FastLanguageModel.for_inference(model)
    except Exception as e:
        logger.info(f"Unsloth not available ({e}), falling back to standard HuggingFace PEFT...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ) if device == "cuda" else None

        base_model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_ID,
            quantization_config=bnb_config if device == "cuda" else None,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()

    print("Mô hình Text2Cypher đã sẵn sàng! Nhập 'exit' hoặc 'quit' để thoát.\n")

    current_schema = DEFAULT_SCHEMA

    while True:
        try:
            user_input = input("\nNhập câu hỏi tiếng Anh/Việt: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Đang thoát demo...")
                break

            prompt = create_student_prompt(current_schema, user_input)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, use_cache=True)

            pred_raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            parsed = parse_cot_json(pred_raw)
            cypher = parsed.get("cypher", pred_raw)

            print("\n" + "=" * 50)
            print("KẾT QUẢ CYPHER:")
            print(cypher)
            print("=" * 50)

        except KeyboardInterrupt:
            print("\nĐã thoát demo.")
            break
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
