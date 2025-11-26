from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json


# MODEL_NAME = "/home/alpaco/lhc/Qwen2.5-14B-Instruct"  # 실제 사용하는 모델 이름
MODEL_NAME = "/home/alpaco/lhc/Qwen2.5-7B-Instruct"  # 실제 사용하는 모델 이름

# 3 x 3090 (각 24GB) 기준: 4bit 로드 + device_map="auto"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

# GPU 메모리 한도 지정 (필요시)
max_memory = {
    0: "20GiB",   # 여유를 조금 남겨두기
    1: "20GiB",
    2: "20GiB",
    "cpu": "32GiB",
}

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model (this can take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},  # 전체 모델을 cuda:0에 올리기
)

model.eval()  # 꼭 eval 모드
model.config.use_cache = True  # generation 속도에 도움

# Qwen은 대개 eos_token_id / pad_token_id 설정이 필요할 수 있음
if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
    model.config.eos_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id

SYSTEM_PROMPT = """
당신은 한국어 레시피 추천 시스템을 위한 키워드/속성 추출기입니다.
사용자의 자연어 요리 요청을 입력으로 받아, 아래 JSON 형식으로만 출력하십시오.

JSON 스키마:
{
  "dish_type": [],
  "method": [],
  "situation": [],
  "must_ingredients": [],
  "optional_ingredients": [],
  "exclude_ingredients": [],
  "spiciness": "none" | "low" | "medium" | "high" | null,
  "dietary_constraints": {
    "vegetarian": bool,
    "vegan": bool,
    "no_beef": bool,
    "no_pork": bool,
    "no_chicken": bool,
    "no_seafood": bool
  },
  "servings": { "min": int or null, "max": int or null },
  "max_cook_time_min": int or null,
  "difficulty": [],
  "positive_tags": [],
  "negative_tags": [],
  "free_text": string
}

규칙:
- 반드시 위 스키마에 맞는 올바른 JSON만 출력하세요. JSON 앞뒤에 다른 텍스트를 넣지 마세요.
- "안 매운", "맵지 않은", "너무 자극적이지 않았으면" → spiciness: "none" 또는 "low"
- "돼지고기는 싫어", "돼지고기 빼고" → exclude_ingredients에 "돼지고기" 추가
- "채식주의자" → vegetarian: true, no_beef/no_pork/no_chicken/no_seafood: true
- "비건" → vegan: true, vegetarian: true, no_beef/no_pork/no_chicken/no_seafood: true
- 사용자의 표현을 요약해서 한 답변을 free_text에 넣으세요.
"""

def _postprocess_text_to_json(output_text: str, fallback_prompt: str) -> dict:
    output_text = output_text.strip()

    # ```json ... ``` 형태 제거
    if output_text.startswith("```"):
        output_text = output_text.strip("`").strip()
        if output_text.startswith("json"):
            output_text = output_text[4:].strip()

    try:
        data = json.loads(output_text)
    except json.JSONDecodeError:
        # 실패 시 기본 구조
        data = {
            "dish_type": [],
            "method": [],
            "situation": [],
            "must_ingredients": [],
            "optional_ingredients": [],
            "exclude_ingredients": [],
            "spiciness": None,
            "dietary_constraints": {
                "vegetarian": False,
                "vegan": False,
                "no_beef": False,
                "no_pork": False,
                "no_chicken": False,
                "no_seafood": False,
            },
            "servings": {"min": None, "max": None},
            "max_cook_time_min": None,
            "difficulty": [],
            "positive_tags": [],
            "negative_tags": [],
            "free_text": fallback_prompt,
        }

    return data


def extract_keywords(user_prompt: str) -> dict:
    """
    한국어 자유 프롬프트를 입력받아 레시피 검색용 키워드를 JSON으로 추출.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Qwen은 대개 chat 템플릿을 지원
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():  # no_grad보다 inference_mode가 더 최적화됨
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,      # 우선 128로 줄여 보고, 더 줄여도 되면 64까지
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return _postprocess_text_to_json(output_text, fallback_prompt=user_prompt)