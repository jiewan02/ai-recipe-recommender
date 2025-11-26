# explanation_model.py
import json
import torch

# new_extractor_model.py 안에서 이미 로드해둔 tokenizer / model 재사용
from new_extractor_model import tokenizer, model

EXPLANATION_SYSTEM_PROMPT = """
당신은 한국어 레시피 추천 시스템의 '추천 이유 설명기'입니다.

역할:
- 이미 선택된 레시피에 대해,
  1) 사용자 요청,
  2) LLM이 추출한 키워드,
  3) 그래프 태그와 실제로 매칭된 키워드 정보,
  4) 레시피 메타데이터(이름, 조리시간, 난이도, 점수 구성)
을 입력으로 받아,
각 레시피를 왜 추천했는지 한국어로 간결하게 한 줄로 설명합니다.

출력 형식은 반드시 아래 JSON 하나만 출력하십시오:

{
  "short_reason": "한두 문장으로 정리된 추천 이유",
  "matched_keywords": [
    "캠핑",
    "채식",
    "간편식"
  ]
}

규칙:
- JSON 외 다른 텍스트는 절대 출력하지 마십시오.
- matched_keywords에는 입력으로 전달된 matched_keywords를 그대로 사용하되,
  필요시 의미가 없는 키워드는 제외할 수 있습니다.
- short_reason은 한국어로 자연스럽고 간결하게 작성하십시오.
"""


def _strip_code_fence(output_text: str) -> str:
    """
    모델이 ```json ... ``` 형태로 감싸서 출력하는 경우를 대비해
    코드 펜스를 제거해준다.
    """
    text = output_text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def generate_explanation_for_recipe(
    user_prompt: str,
    global_keywords: dict,
    recipe_info: dict,
) -> dict:
    """
    한 개 레시피에 대해 LLM 기반 설명 JSON 생성.

    Parameters
    ----------
    user_prompt : str
        원본 사용자 입력 문장.
    global_keywords : dict
        extract_keywords → build_cypher_from_keywords_relaxed 이후의 kw 딕셔너리
        (res["keywords"] 그대로 넣으면 됨).
    recipe_info : dict
        graph_rag_search_with_scoring_explanation에서 반환된 각 레시피 dict
        (matched_keywords_flat / matched_tag_dict를 포함하고 있다고 가정).
    """
    payload = {
        "user_prompt": user_prompt,
        "keywords": global_keywords,
        "recipe": {
            "recipe_id": recipe_info.get("recipe_id"),
            "title": recipe_info.get("title"),
            "name": recipe_info.get("name"),
            "time_min": recipe_info.get("time_min"),
            "difficulty": recipe_info.get("difficulty"),
            "servings": recipe_info.get("servings"),
            "score_breakdown": {
                "total": recipe_info.get("score"),
                "must_ing": recipe_info.get("score_must_ing"),
                "opt_ing": recipe_info.get("score_opt_ing"),
                "dish_type": recipe_info.get("score_dish_type"),
                "method": recipe_info.get("score_method"),
                "situation": recipe_info.get("score_situation"),
                "health": recipe_info.get("score_health"),
                "weather": recipe_info.get("score_weather"),
                "menu_style": recipe_info.get("score_menu_style"),
                "extra": recipe_info.get("score_extra"),
            },
            # 1단계에서 ipynb에 추가해둔 필드들을 그대로 사용
            "matched_keywords": recipe_info.get("matched_keywords_flat", []),
            "matched_tag_dict": recipe_info.get("matched_tag_dict", {}),
        },
    }

    messages = [
        {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    output_text = _strip_code_fence(output_text)

    try:
        data = json.loads(output_text)
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 최소 fallback
        data = {
            "short_reason": output_text,
            "matched_keywords": recipe_info.get("matched_keywords_flat", []),
        }

    # 필드 누락 보정
    if "short_reason" not in data or not isinstance(data["short_reason"], str):
        data["short_reason"] = ""
    if "matched_keywords" not in data or not isinstance(data["matched_keywords"], list):
        data["matched_keywords"] = recipe_info.get("matched_keywords_flat", [])

    return data


def add_llm_explanations(user_prompt: str, search_result: dict) -> dict:
    """
    기존 graph_rag_search_with_scoring_explanation 결과(search_result)에
    LLM 기반 설명(JSON)을 추가해서 되돌려준다.

    Parameters
    ----------
    user_prompt : str
        원본 사용자 입력
    search_result : dict
        ipynb에서 사용하는 graph_rag_search_with_scoring_explanation 리턴 값
        {
          "keywords": kw,
          "recipes": [ r_info1, r_info2, ... ]
        }

    Returns
    -------
    dict
        search_result와 같은 구조이지만, 각 recipe에 "llm_explanation" 필드가 추가된다.
    """
    kw = search_result.get("keywords", {})
    recipes = search_result.get("recipes", [])

    for r in recipes:
        expl = generate_explanation_for_recipe(user_prompt, kw, r)
        r["llm_explanation"] = expl  # short_reason, matched_keywords

    return search_result
