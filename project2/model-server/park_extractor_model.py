from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import os
from openai import OpenAI

# =========================================================
# 0. 모델 경로 & 4bit 설정 (원래 코드 유지)
#    👉 이제는 로컬 Qwen 대신 OpenAI API 모델 이름으로 사용
# =========================================================

# MODEL_NAME = "/home/alpaco/lhc/Qwen2.5-14B-Instruct"  # 실제 사용하는 모델 경로/이름
MODEL_NAME = "gpt-4.1-mini"  # OpenAI API에서 사용할 모델 이름 (필요시 gpt-4o 등으로 변경)

# 3 x 3090 (각 24GB) 기준: 4bit 로드 + device_map="auto"
# 👉 OpenAI API 사용 시에는 실제로 사용되지 않는 설정이지만,
#    원래 코드 구조를 유지하기 위해 그대로 둡니다.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

# GPU 메모리 한도 지정 (필요시)
# 👉 마찬가지로 OpenAI API에서는 사용되지 않습니다.
max_memory = {
    0: "20GiB",   # 여유를 조금 남겨두기
    1: "20GiB",
    2: "20GiB",
    "cpu": "32GiB",
}

print("Loading OpenAI client...")

# 🔹 OpenAI API 키 설정 (여기에 본인 키 넣으세요)
os.environ["OPENAI_API_KEY"] =""
#OPENAI_API_KEY=""

# 🔹 OpenAI Python SDK 클라이언트
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# 🔹 로컬 Qwen 모델은 더 이상 사용하지 않으므로 placeholder 로 둡니다.
tokenizer = None
model = None

# Qwen은 eos_token_id / pad_token_id 설정이 필요할 수 있음
# 👉 이제는 모델을 직접 로드하지 않으므로, 아래 if 블록은 의미가 없어집니다.
#    하지만 원래 코드 구조/주석을 유지하기 위해 남겨 둡니다.
if hasattr(torch, "Tensor"):  # 형식상 조건 (실제 동작 X)
    pass


# =========================================================
# 1. SYSTEM PROMPT (확장된 JSON 스키마)
# =========================================================

SYSTEM_PROMPT = """
당신은 한국어 레시피 추천 시스템의 핵심 구성요소인
"요리 의도·조건 추출기(Keyword & Constraint Extractor)" 입니다.

당신의 임무는:
사용자의 자연어 요리 요청에서 **모든 의미 있는 요소를 빠짐없이 구조화된 JSON으로 추출하는 것**입니다.

출력 규칙은 반드시 다음을 따라야 합니다:

────────────────────────────────────────
📌 **① 출력 형식**

오직 아래 JSON 스키마 형태의 JSON만 출력하십시오.
문장 설명, 해설, 추가 문구는 절대 출력하지 마십시오.
무조건 답변은 한국어로 하십시오.

모든 필드는 반드시 존재해야 합니다.
값이 비었으면 **빈 배열 또는 null** 로 채우십시오.

JSON 스키마:

{
  "dish_type": [],                    // 요리 형태/종류
  "method": [],                       // 조리 방식
  "situation": [],                    // 먹는 상황/맥락
  "must_ingredients": [],             // 반드시 포함되어야 하는 재료
  "optional_ingredients": [],         // 들어가면 좋지만 필수는 아닌 재료
  "exclude_ingredients": [],          // 절대 들어가면 안 되는 재료
  "spiciness": "none" | "low" | "medium" | "high" | null,

  "dietary_constraints": {
    "vegetarian": bool,
    "vegan": bool,
    "no_beef": bool,
    "no_pork": bool,
    "no_chicken": bool,
    "no_seafood": bool
  },

  "servings": {
    "min": int or null,
    "max": int or null
  },

  "max_cook_time_min": int or null,
  "difficulty": [],

  "health_tags": [],                  // 건강, 목적, 영양 관련
  "weather_tags": [],                 // 날씨/계절
  "menu_style": [],                   // 국가/분류/스타일
  "extra_keywords": [],               // 위 어디에도 속하지 않는 의미 있는 단어

  "positive_tags": [],                // 사용자가 원함/좋아함
  "negative_tags": [],                // 사용자가 싫어함/피하고 싶음

  "free_text": string                 // 전체 요청의 자연어 요약
}

────────────────────────────────────────
📌 **② 각 필드의 세부 지침 (매우 중요)**

1) dish_type (요리 종류)
- "찌개", "국", "볶음", "튀김", "조림", "덮밥", "비빔밥", "면 요리" 등
- 가능하면 구체적 표현으로 추출

2) method (조리 방식)
- "끓이기", "볶기", "찜", "무침", "튀기기", "굽기" 등

3) situation (먹는 상황)
- "야식", "혼밥", "술안주", "손님 초대", "간단하게", "도시락", "캠핑" 등
- 문맥에서 숨겨진 상황도 추론하여 포함 가능

4) must_ingredients 규칙(아주 중요):

  사용자가 특정 재료를 언급하며 “~ 요리 추천해줘”, “~ 넣어 먹고 싶다”, 
  “~로 만들고 싶다”, “~ 요리가 땡긴다”, “~ 요리 먹고 싶어” 라고 말한 경우
  → 해당 재료는 MUST INGREDIENT로 간주한다.

  예:
  - “돼지고기 요리 추천해줘” → must_ingredients: ["돼지고기"]
  - “김치랑 버섯이랑 같이 먹고 싶다” → must_ingredients: ["김치","버섯"]
  - “닭고기로 뭘 만들까?” → must_ingredients: ["닭고기"]

5) optional_ingredients
- “있으면 좋고”, “가능하면”, “추가로” 표현된 재료

6) exclude_ingredients
- “싫어”, “알레르기”, “못먹어”, “빼고”, “제외해줘” 등

7) spiciness
- "안 매운", "매콤하게", "얼큰하게" 등 해석하여 4단계로 정규화:
  - none, low, medium, high

8) dietary_constraints
- "채식주의자" → vegetarian=true + no_beef/no_pork/no_chicken/no_seafood=true, 어떠한 형태의 고기요리나 생선요리도 포함되면 안돼
- "비건" → vegan=true + 위 조건 모두 true
- “고기 안 먹어” → no_beef/no_pork/no_chicken 모두 true
- 특정 육류만 싫어하면 해당 항목만 true

9) servings
- "1인분", "혼자 먹을 거야" → min=1, max=1
- "4명", "가족" → 추론 가능하면 넣고, 불확실하면 null

10) max_cook_time_min
- "10분 안에", "빨리", "간단히" → 명확히 숫자가 있을 때만 기입
- 숫자가 없으면 null

11) difficulty
- "간단한", "쉽게", "어려운 요리" 등 난이도 표현 그대로 단어로 넣기

12) health_tags
- "다이어트", "고단백", "저염식", "저칼로리", "영양식" 등

13) weather_tags
- "추운 날", "더운 날", "비오는 날", "겨울", "여름" 등

14) menu_style
- "한식", "중식", "양식", "분식", "디저트", "안주", "브런치", "한 그릇" 등

15) extra_keywords
- 위 항목들 어디에도 속하지 않지만 의미 있는 단어
  예: "백종원 레시피", "에어프라이어", "명절", "칼칼한", "간편식"

16) positive_tags
- “좋아해”, “먹고 싶어”, “원해”, “ craving ” 등 감정/선호 기반

17) negative_tags
- “싫어”, “별로”, “꺼려져”, "먹기 싫어"

18) free_text
- 전체 사용자 요청을 자연스럽고 간결하게 요약한 1~2문장
- 단순 요약이 아니라, 다음 내용을 반드시 포함해야 함:
  1) 사용자가 어떤 상황에서 어떤 종류의 음식을 원하고 있는지
  2) 재료/식단 제한/취향 조건이 서로 모순되거나 비현실적인지 여부
  3) 모순·위험·비현실적인 조건이 있다면,
     "요청을 그대로 만족시키기 어렵다"는 점과
     어떤 방향(예: 비건 유지, 알레르기 회피 등)을 우선해야 하는지 제안

────────────────────────────────────────
📌 **③ 절대적으로 지켜야 할 3가지**

1. 출력은 반드시 **JSON 단독**이어야 함 (앞뒤 추가 텍스트 금지)  
2. JSON 스키마의 **모든 필드**를 반드시 출력  
3. 빈 값도 반드시 포함 (절대 필드 누락 금지)

────────────────────────────────────────

지금부터 사용자 입력이 들어오면
위 스키마에 맞춰 **정확하고 완전한 JSON**만 출력하십시오.
"""

# =========================================================
# 2. Post-process helpers
# =========================================================

def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        if not x.strip():
            return []
        return [x]
    return list(x)

def _unique_preserve_order(lst):
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _dedup_by_norm_space_lower(tags):
    """
    ['추운 날', '추운날'] 같이 공백/대소문자만 다른 중복을 하나로 정리.
    """
    norm_seen = set()
    out = []
    for t in _ensure_list(tags):
        norm = str(t).replace(" ", "").lower()
        if norm in norm_seen:
            continue
        norm_seen.add(norm)
        out.append(t)
    return out


def _postprocess_text_to_json(output_text: str, fallback_prompt: str) -> dict:
    """
    - LLM이 출력한 JSON 텍스트를 파싱
    - 누락/타입 이상 필드 보정
    - positive_tags → health_tags / extra_keywords 확장
    - weather_tags는 LLM 출력만 사용하되, 표기 중복만 정리
    """
    output_text = output_text.strip()

    # ```json ... ``` 형태 제거
    if output_text.startswith("```"):
        output_text = output_text.strip("`").strip()
        if output_text.startswith("json"):
            output_text = output_text[4:].strip()

    try:
        data = json.loads(output_text)
    except json.JSONDecodeError:
        data = {}

    # 1) 기본 골격 (확장된 스키마 기준)
    base = {
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

        "health_tags": [],
        "weather_tags": [],
        "menu_style": [],
        "extra_keywords": [],

        "positive_tags": [],
        "negative_tags": [],
        "free_text": fallback_prompt,
    }

    if not isinstance(data, dict):
        data = {}

    merged = base.copy()
    merged.update({k: v for k, v in data.items() if v is not None})

    # 리스트 필드 정규화
    list_fields = [
        "dish_type", "method", "situation",
        "must_ingredients", "optional_ingredients", "exclude_ingredients",
        "difficulty",
        "health_tags", "weather_tags", "menu_style", "extra_keywords",
        "positive_tags", "negative_tags",
    ]
    for k in list_fields:
        merged[k] = _ensure_list(merged.get(k))

    # dietary_constraints 보정
    dc = merged.get("dietary_constraints") or {}
    merged["dietary_constraints"] = {
        "vegetarian": bool(dc.get("vegetarian", False)),
        "vegan": bool(dc.get("vegan", False)),
        "no_beef": bool(dc.get("no_beef", False)),
        "no_pork": bool(dc.get("no_pork", False)),
        "no_chicken": bool(dc.get("no_chicken", False)),
        "no_seafood": bool(dc.get("no_seafood", False)),
    }

    # servings 보정
    serv = merged.get("servings") or {}
    merged["servings"] = {
        "min": serv.get("min"),
        "max": serv.get("max"),
    }

    # spiciness 정규화
    valid_sp = {"none", "low", "medium", "high", None}
    sp = merged.get("spiciness")
    if isinstance(sp, str):
        sp = sp.lower().strip()
        if sp not in valid_sp:
            sp = None
    elif sp not in valid_sp:
        sp = None
    merged["spiciness"] = sp

    # free_text 보정
    if not isinstance(merged.get("free_text"), str) or not merged["free_text"].strip():
        merged["free_text"] = fallback_prompt

    # positive_tags → health_tags / extra_keywords 확장
    pos = merged.get("positive_tags", [])
    merged["health_tags"] = _unique_preserve_order(merged["health_tags"] + pos)
    merged["extra_keywords"] = _unique_preserve_order(merged["extra_keywords"] + pos)

    # weather_tags는 LLM 출력만 신뢰, 대신 공백/대소문자 기준 중복 제거
    merged["weather_tags"] = _dedup_by_norm_space_lower(merged["weather_tags"])

    return merged


# =========================================================
# 3. 실제 호출 함수: extract_keywords (원래 chat_template 방식 유지 → OpenAI API로 변경)
# =========================================================

def extract_keywords(user_prompt: str) -> dict:
    """
    한국어 자유 프롬프트를 입력받아 레시피 검색용 키워드를 JSON으로 추출.
    - 기존에는 4bit Qwen2.5-14B (multi-GPU) 사용
    - 이제는 OpenAI Responses API를 사용하여
      SYSTEM_PROMPT에 정의된 확장 스키마로
      health_tags / weather_tags / menu_style / extra_keywords까지 포함
    """

    # OpenAI Responses API는 instructions + input 조합으로
    # system / user 역할을 줄 수 있음.
    # - instructions: SYSTEM_PROMPT (역할/스키마 설명)
    # - input: 실제 user_prompt
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        temperature=0.0,        # JSON 포맷 유지 위해 최대한 결정적으로
        max_output_tokens=256,  # 원래 max_new_tokens=128보다는 약간 여유
    )

    # 모델이 생성한 순수 텍스트(JSON 문자열)
    output_text = response.output_text or ""

    return _postprocess_text_to_json(output_text, fallback_prompt=user_prompt)


# 파일 맨 아래 근처
__all__ = ["extract_keywords", "tokenizer", "model"]


import csv
import os

# 결과를 저장할 CSV 경로 (원하는 이름으로 바꿔도 됩니다)
CSV_PATH = "keyword_extract_log.csv"


def _join_list(x):
    """리스트를 ' | '로 이어붙여서 문자열로 변환"""
    if isinstance(x, list):
        return " | ".join(map(str, x))
    return ""


def result_to_row(user_prompt: str, result: dict) -> dict:
    """
    extract_keywords() 결과 dict를
    CSV 한 줄(row) 형태의 flat dict로 변환
    """
    dc = result.get("dietary_constraints", {}) or {}
    serv = result.get("servings", {}) or {}

    row = {
        "user_prompt": user_prompt,

        # 리스트 필드들: " | " 로 join
        "dish_type": _join_list(result.get("dish_type", [])),
        "method": _join_list(result.get("method", [])),
        "situation": _join_list(result.get("situation", [])),
        "must_ingredients": _join_list(result.get("must_ingredients", [])),
        "optional_ingredients": _join_list(result.get("optional_ingredients", [])),
        "exclude_ingredients": _join_list(result.get("exclude_ingredients", [])),
        "difficulty": _join_list(result.get("difficulty", [])),
        "health_tags": _join_list(result.get("health_tags", [])),
        "weather_tags": _join_list(result.get("weather_tags", [])),
        "menu_style": _join_list(result.get("menu_style", [])),
        "extra_keywords": _join_list(result.get("extra_keywords", [])),
        "positive_tags": _join_list(result.get("positive_tags", [])),
        "negative_tags": _join_list(result.get("negative_tags", [])),

        # 단일 값들
        "spiciness": result.get("spiciness", None),

        # diet 제약 (bool)
        "vegetarian": dc.get("vegetarian", False),
        "vegan": dc.get("vegan", False),
        "no_beef": dc.get("no_beef", False),
        "no_pork": dc.get("no_pork", False),
        "no_chicken": dc.get("no_chicken", False),
        "no_seafood": dc.get("no_seafood", False),

        # 인분 정보
        "servings_min": serv.get("min", None),
        "servings_max": serv.get("max", None),

        # 시간, free_text
        "max_cook_time_min": result.get("max_cook_time_min", None),
        "free_text": result.get("free_text", ""),
    }
    return row


def save_result_to_csv(user_prompt: str, result: dict, csv_path: str = CSV_PATH):
    """
    - user_prompt + extract_keywords 결과를 한 줄(row)로 만들어서
      csv_path에 한 줄씩 추가(append) 저장
    - 파일이 없으면 헤더를 먼저 쓰고, 있으면 헤더 없이 내용만 추가
    """
    row = result_to_row(user_prompt, result)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_and_log(user_prompt: str, csv_path: str = CSV_PATH):
    """
    편하게 쓰라고 만든 헬퍼 함수:
    - extract_keywords(user_prompt) 실행
    - 결과를 CSV에 저장
    - 결과 dict를 그대로 리턴
    """
    result = extract_keywords(user_prompt)
    save_result_to_csv(user_prompt, result, csv_path=csv_path)
    return result
