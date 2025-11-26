from pprint import pprint
from neo4j import GraphDatabase
import torch
from extractor_model import extract_keywords, model, tokenizer
import re
import json


URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

print("✅ Neo4j driver 생성 완료")
print("[노트북] extractor_model import 시작")
print("[노트북] extractor_model import 완료 ✅")


# --------

# Cell 2: Canonical ingredient 매핑 + 헬퍼 함수들

# =========================================
#  Canonical ingredient mapping (from CSV 분석 기반)
#  - key 가 ingredient 이름 안에 포함되면 → value 로 정규화
#  - 더 특수한 키를 위에, 더 일반적인 키를 아래에 두는 게 안전
# =========================================

CANONICAL_INGREDIENTS = {
    # -----------------------------
    # 1) 돼지고기 계열
    # -----------------------------
    "대패삼겹살": "돼지고기",
    "통삼겹살": "돼지고기",
    "냉동대패삼겹살": "돼지고기",
    "냉동삼겹살": "돼지고기",
    "생삼겹살": "돼지고기",
    "고추장삼겹살": "돼지고기",
    "삼겹살": "돼지고기",

    "돼지앞다리살": "돼지고기",
    "돼지고기앞다리살": "돼지고기",
    "앞다리살": "돼지고기",

    "돼지목살": "돼지고기",
    "돼지고기목살": "돼지고기",
    "목살": "돼지고기",

    "돼지등갈비": "돼지고기",
    "돼지갈비": "돼지고기",
    "돼지고기갈비": "돼지고기",
    "등갈비": "돼지고기",
    "갈비": "돼지고기",

    "돼지고기다짐육": "돼지고기",
    "다진돼지고기": "돼지고기",
    "간돼지고기": "돼지고기",

    "돼지고기사태": "돼지고기",
    "돼지등뼈": "돼지고기",

    "돼지고기": "돼지고기",
    "돼지": "돼지고기",

    # -----------------------------
    # 2) 소고기 계열
    # -----------------------------
    "우삼겹": "소고기",
    "채끝살": "소고기",
    "등심": "소고기",
    "안심": "소고기",
    "양지": "소고기",
    "사태": "소고기",
    "우둔살": "소고기",

    "다진소고기": "소고기",
    "간소고기": "소고기",

    "쇠고기": "소고기",
    "한우": "소고기",
    "소고기": "소고기",

    # -----------------------------
    # 3) 닭고기 / 오리고기 계열
    # -----------------------------
    "닭가슴살햄": "닭고기",
    "그릴닭가슴살": "닭고기",
    "냉동닭가슴살": "닭고기",
    "닭가슴살": "닭고기",

    "닭다리살": "닭고기",
    "닭봉": "닭고기",
    "닭날개": "닭고기",
    "닭윙": "닭고기",
    "닭안심": "닭고기",
    "닭발": "닭고기",
    "닭볶음탕용": "닭고기",
    "통닭": "닭고기",

    "닭고기": "닭고기",
    "닭": "닭고기",

    "훈제오리": "오리고기",
    "오리고기": "오리고기",
    "오리": "오리고기",

    # -----------------------------
    # 4) 가공육류 (햄/베이컨/소시지/스팸 등)
    # -----------------------------
    "스팸": "스팸",
    "베이컨": "베이컨",
    "훈제베이컨": "베이컨",
    "칵테일햄": "햄",
    "햄": "햄",
    "소시지": "소시지",
    "비엔나소세지": "소시지",
    "비엔나소시지": "소시지",
    "후랑크소시지": "소시지",

    # -----------------------------
    # 5) 계란 / 달걀 계열
    # -----------------------------
    "달걀노른자": "계란",
    "달걀흰자": "계란",
    "삶은달걀": "계란",
    "달걀프라이": "계란",
    "달걀후라이": "계란",
    "달걀지단": "계란",
    "달걀말이": "계란",
    "달걀물": "계란",
    "구운달걀": "계란",
    "달걀": "계란",

    "계란노른자": "계란",
    "계란흰자": "계란",
    "삶은계란": "계란",
    "계란후라이": "계란",
    "계란프라이": "계란",
    "계란지단": "계란",
    "계란말이": "계란",
    "계란물": "계란",
    "구운계란": "계란",
    "훈제계란": "계란",
    "계란": "계란",

    # -----------------------------
    # 6) 파 계열
    # -----------------------------
    "대파흰부분": "파",
    "대파초록부분": "파",
    "송송썬대파": "파",
    "채썬대파": "파",
    "흰대파": "파",
    "대파뿌리": "파",
    "대파잎": "파",
    "다진대파": "파",
    "다진파": "파",
    "썬파": "파",
    "파채": "파",
    "대파채": "파",
    "실파": "파",
    "쪽파": "파",
    "대파": "파",
    "파": "파",

    # -----------------------------
    # 7) 마늘 계열
    # -----------------------------
    "다진마늘": "마늘",
    "간마늘": "마늘",
    "통마늘": "마늘",
    "편마늘": "마늘",
    "깐마늘": "마늘",
    "알마늘": "마늘",
    "저민마늘": "마늘",
    "생마늘": "마늘",
    "마늘쫑": "마늘쫑",
    "마늘종": "마늘쫑",
    "마늘가루": "마늘",
    "건조마늘가루": "마늘",
    "마늘대": "마늘",
    "마늘": "마늘",

    # -----------------------------
    # 8) 양파 계열
    # -----------------------------
    "적양파": "양파",
    "홍양파": "양파",
    "자색양파": "양파",
    "양파즙": "양파",
    "양파": "양파",

    # -----------------------------
    # 9) 배추/양배추/양상추 계열
    # -----------------------------
    "알배추": "배추",
    "알배기배추": "배추",
    "절임배추": "배추",
    "배추겉절이": "배추",
    "쌈배추": "배추",
    "얼갈이배추": "배추",
    "단배추": "배추",
    "방울양배추": "양배추",
    "양배추즙": "양배추",
    "양배추": "양배추",
    "자색양배추": "양배추",
    "양상추": "양상추",

    # -----------------------------
    # 10) 고추/고춧가루
    # -----------------------------
    "청양고추": "청양고추",
    "청고추": "청양고추",
    "청량고추": "청양고추",
    "풋고추": "풋고추",
    "꽈리고추": "꽈리고추",

    "홍고추": "홍고추",
    "건고추": "건고추",
    "베트남고추": "건고추",
    "베트남건고추": "건고추",

    "고추기름": "고추기름",
    "초고추장": "고추장",
    "고추장": "고추장",
    "고추가루": "고춧가루",
    "고춧가루": "고춧가루",
    "고추": "고추",

    # -----------------------------
    # 11) 설탕/당류
    # -----------------------------
    "황설탕": "설탕",
    "흑설탕": "설탕",
    "흰설탕": "설탕",
    "백설탕": "설탕",
    "갈색설탕": "설탕",
    "자일로스설탕": "설탕",
    "비정제설탕": "설탕",
    "노란설탕": "설탕",
    "사탕수수설탕": "설탕",
    "설탕시럽": "설탕",
    "원당": "설탕",
    "스테비아": "설탕대체",
    "설탕": "설탕",

    "올리고당": "올리고당",
    "물엿": "물엿",
    "조청": "조청",
    "시럽": "시럽",

    # -----------------------------
    # 12) 소금
    # -----------------------------
    "꽃소금": "소금",
    "굵은소금": "소금",
    "함초소금": "소금",
    "구운소금": "소금",
    "트러플소금": "소금",
    "허브소금": "소금",
    "소금": "소금",

    # -----------------------------
    # 13) 간장/장류
    # -----------------------------
    "진간장": "간장",
    "국간장": "간장",
    "양조간장": "간장",
    "맛간장": "간장",
    "집간장": "간장",
    "조선간장": "간장",
    "어간장": "간장",
    "홍게간장": "간장",
    "대게백간장": "간장",
    "만능간장": "간장",
    "초간장": "간장",
    "간장소스": "간장",
    "아기간장": "간장",
    "간장": "간장",

    "된장": "된장",
    "쌈장": "쌈장",
    "고추장아찌": "고추장",
    "쌈장소스": "쌈장",

    # -----------------------------
    # 14) 액젓/발효장
    # -----------------------------
    "멸치액젓": "액젓",
    "까나리액젓": "액젓",
    "참치액젓": "액젓",
    "꽃게액젓": "액젓",
    "갈치액젓": "액젓",
    "멜치액젓": "액젓",
    "액젓": "액젓",
    "참치액": "참치액",

    # -----------------------------
    # 15) 식초/산미
    # -----------------------------
    "사과식초": "식초",
    "현미식초": "식초",
    "양조식초": "식초",
    "발사믹식초": "식초",
    "레몬식초": "식초",
    "식초": "식초",
    "레몬즙": "레몬즙",
    "유자청": "유자청",

    # -----------------------------
    # 16) 기름/오일
    # -----------------------------
    "포도씨유": "식용유",
    "카놀라유": "식용유",
    "콩기름": "식용유",
    "땅콩기름": "식용유",
    "해바라기씨유": "식용유",
    "옥수수유": "식용유",
    "식용유": "식용유",

    "올리브오일": "올리브유",
    "엑스트라버진올리브오일": "올리브유",
    "올리브유": "올리브유",

    "들기름": "들기름",
    "참기름": "참기름",

    # -----------------------------
    # 17) 버섯
    # -----------------------------
    "건표고버섯": "표고버섯",
    "생표고버섯": "표고버섯",
    "마른표고버섯": "표고버섯",
    "표고버섯가루": "표고버섯",
    "표고버섯": "표고버섯",

    "미니새송이버섯": "새송이버섯",
    "꼬마새송이버섯": "새송이버섯",
    "총알새송이버섯": "새송이버섯",
    "새송이버섯": "새송이버섯",

    "양송이": "양송이버섯",
    "양송이버섯": "양송이버섯",

    "느타리버섯": "느타리버섯",
    "애느타리버섯": "느타리버섯",

    "팽이버섯": "팽이버섯",
    "만가닥버섯": "만가닥버섯",
    "목이버섯": "목이버섯",
    "송이버섯": "송이버섯",
    "버섯": "버섯",

    # -----------------------------
    # 18) 해산물
    # -----------------------------
    "냉동새우": "새우",
    "자숙새우": "새우",
    "중새우": "새우",
    "왕새우": "새우",
    "새우살": "새우",
    "새우": "새우",
    "대하": "새우", 

    "오징어채": "오징어",
    "마른오징어": "오징어",
    "오징어": "오징어",

    "낙지": "낙지",
    "문어": "문어",
    "쭈꾸미": "쭈꾸미",

    "홍합살": "홍합",
    "홍합": "홍합",
    "바지락살": "바지락",
    "바지락": "바지락",
    "모시조개": "모시조개",
    "조개살": "조개",
    "조개": "조개",
    "가리비살": "가리비",
    "가리비관자": "가리비",
    "가리비": "가리비",

    "고등어": "고등어",
    "굴비": "조기",
    "조기": "조기",
    "갈치": "갈치",
    "삼치": "삼치",
    "꽁치": "꽁치",
    "연어": "연어",
    "코다리": "코다리",
    "명태": "명태",
    "대구": "대구",

    "전복살": "전복",
    "전복": "전복",

    # -----------------------------
    # 19) 기타
    # -----------------------------
    "다시다": "조미료",
    "미원": "조미료",
    "치킨스톡": "조미료",
    "치킨파우더": "조미료",

    "찹쌀가루": "찹쌀",
    "찹쌀": "찹쌀",
    "멥쌀": "쌀",
    "쌀밥": "밥",
    "현미밥": "밥",
    "보리밥": "밥",
    "밥": "밥",
    "쌀": "쌀",

    "라면사리": "라면",
    "라면": "라면",
    "당면": "당면",
    "소면": "소면",
    "국수": "국수",
    "우동사리": "우동",
    "우동면": "우동",

    "토마토홀": "토마토",
    "방울토마토": "토마토",
    "방울토마토즙": "토마토",
    "토마토": "토마토",
}


def canonicalize_ingredient_name(name: str) -> str:
    """
    재료 이름 하나를 canonical form 으로 변환.
    - CANONICAL_INGREDIENTS 의 key 가 name 에 포함되면 → value 로 매핑
    - 아무 매칭도 안 되면 원래 name 그대로 반환
    """
    if not isinstance(name, str):
        return name

    name = name.strip()
    if not name:
        return name

    for key, canon in CANONICAL_INGREDIENTS.items():
        if key in name:
            return canon

    return name


def canonicalize_ingredient_list(lst):
    """
    재료 리스트 전체에 canonical 적용.
    """
    if lst is None:
        return None
    result = []
    for x in lst:
        c = canonicalize_ingredient_name(str(x))
        if c:
            result.append(c)
    return result


def ensure_list(x):
    """
    문자열/None/리스트/튜플 등 들어와도 무조건 리스트로 바꿔주는 헬퍼.
    """
    if x is None:
        return []
    if isinstance(x, str):
        if not x.strip():
            return []
        return [x]
    return list(x)

#------------------------------------------------------

KNOWN_WEATHER_TAGS = [
    "더운 날", "추운날", "여름", "봄", "겨울",
    "비오는 날", "가을", "장마철", "복날", "눈오는날"
]

def infer_weather_tags_from_texts(text_list):
    """여러 텍스트 후보들에서 weather_tags를 유추"""
    detected = []
    for txt in text_list:
        if not txt:
            continue
        t_norm = str(txt).replace(" ", "").lower()
        for tag in KNOWN_WEATHER_TAGS:
            tag_norm = tag.replace(" ", "").lower()
            if tag_norm in t_norm:
                detected.append(tag)
    # 중복 제거 + 순서 유지
    return list(dict.fromkeys(detected))

def normalize_basic(text: str) -> str:
    """공백/특수문자 제거 + 소문자. 한글/영문/숫자만 남김."""
    if not isinstance(text, str):
        text = str(text)
    t = re.sub(r"[^0-9A-Za-z가-힣]", "", text)
    return t.lower()

def make_char_ngrams(text: str, min_n: int = 2, max_n: int = 4):
    """
    "건강한 국이 땡긴다" -> "건강한국이땡긴다" -> 
    2~4글자짜리 문자 n-gram 리스트.
    예: ["건강", "강한", "한국", ...]
    """
    norm = normalize_basic(text)
    grams = set()
    L = len(norm)
    if L < min_n:
        return []
    for n in range(min_n, min(max_n, L) + 1):
        for i in range(L - n + 1):
            grams.add(norm[i:i+n])
    # 너무 일반적인 (예: "하다", "하다") 같은 거 필터링하고 싶으면 여기서 길이나 빈도 조건 줄 수 있음
    return sorted(grams)

def _norm_for_match(s: str) -> str:
    """Cypher에서 했던 것처럼: 소문자 + 공백 제거."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"\s+", "", s).lower()


def analyze_match_dimension(
    kw_list,
    graph_list,
    prompt_ngrams,
    weight: int,
    symmetric_kw: bool = False,
):
    """
    한 차원(dish_type / method / situation / health / weather / menu_style / extra)에 대해
    - LLM 키워드 기반 매칭
    - prompt n-gram 기반 매칭
    을 모두 분석해서 디버깅 정보 + 점수를 계산한다.

    반환 형식 예:
    {
        "kw_values": [...],
        "graph_values": [...],
        "prompt_ngrams": [...],
        "kw_match_detail": { "국물요리": ["국물요리", "따끈한국물요리"], ... },
        "prompt_match_detail": { "국물요리": ["국물요리"], "국물": ["국물요리"], ... },
        "count_kw": 2,
        "count_prompt": 4,
        "raw_match_count": 6,
        "score": 18   # raw_match_count * weight
    }
    """
    kw_list = [k for k in (kw_list or []) if k]
    graph_list = [g for g in (graph_list or []) if g]
    prompt_ngrams = [p for p in (prompt_ngrams or []) if p]

    # 정규화
    graph_norm = [(g, _norm_for_match(g)) for g in graph_list]
    kw_norm = [(k, _norm_for_match(k)) for k in kw_list]

    kw_match_detail = {}
    prompt_match_detail = {}

    # 1) LLM 키워드 기반 매칭 (Cypher의 size([... kw ...]) 로직과 동일하게 "kw 개수"를 센다)
    count_kw = 0
    for k, kn in kw_norm:
        if not kn:
            continue
        matched_tags = []
        for g_orig, gn in graph_norm:
            cond = gn.find(kn) != -1
            if symmetric_kw:
                cond = cond or (kn.find(gn) != -1)
            if cond:
                matched_tags.append(g_orig)
        if matched_tags:
            count_kw += 1
            kw_match_detail[k] = matched_tags

    # 2) prompt n-gram 기반 매칭
    count_prompt = 0
    for pg in prompt_ngrams:
        pg_norm = _norm_for_match(pg)
        if not pg_norm:
            continue
        matched_tags = []
        for g_orig, gn in graph_norm:
            if gn.find(pg_norm) != -1:
                matched_tags.append(g_orig)
        if matched_tags:
            count_prompt += 1
            prompt_match_detail[pg] = matched_tags

    raw_match_count = count_kw + count_prompt
    score = raw_match_count * weight

    return {
        "kw_values": kw_list,
        "graph_values": graph_list,
        "prompt_ngrams": prompt_ngrams,
        "kw_match_detail": kw_match_detail,
        "prompt_match_detail": prompt_match_detail,
        "count_kw": count_kw,
        "count_prompt": count_prompt,
        "raw_match_count": raw_match_count,
        "score": score,
    }

def build_cypher_from_keywords_relaxed(kw: dict, limit: int = 50):
    """
    최종 버전 (하드코딩 최소화):

    - positive_tags → health_tags / extra_keywords 에 단순 합치기 (구조적 처리)
    - ingredient canonicalization + ensure_list
    - 프롬프트 전체 텍스트에서 char n-gram 생성 (prompt_ngrams)
    - 스코어:
        * 기존: LLM이 추출한 각 리스트(must/opt/dish/method/...)
                 와 그래프 태그 간 substring 매칭 개수 × weight
        * 추가: prompt_ngrams 와 그래프 태그 간 substring 매칭 개수 × 같은 weight
      → 즉, "건강한" vs "건강식"처럼 일부만 겹쳐도 점수 반영
    - must_ingredients 는 여전히 하드 필터 + 스코어
    - 정규화된 kw + prompt_ngrams 도 반환
    """

    # 0) kw 복사
    kw = dict(kw)

    # --- 1) positive_tags → health/extra 단순 확장 (semantic X, 구조적 처리 O) ---
    pos_tags    = ensure_list(kw.get("positive_tags"))
    base_health = ensure_list(kw.get("health_tags"))
    base_extra  = ensure_list(kw.get("extra_keywords"))

    if pos_tags:
        kw["health_tags"]    = base_health + pos_tags
        kw["extra_keywords"] = base_extra  + pos_tags
    else:
        kw["health_tags"]    = base_health
        kw["extra_keywords"] = base_extra

    # --- 2) ingredient canonicalization ---
    for key in ["must_ingredients", "optional_ingredients", "exclude_ingredients"]:
        if key in kw and kw[key]:
            kw[key] = canonicalize_ingredient_list(ensure_list(kw[key]))
        else:
            kw[key] = []

    # --- 3) 리스트 정규화 ---
    kw["dish_type"]      = ensure_list(kw.get("dish_type"))
    kw["method"]         = ensure_list(kw.get("method"))
    kw["situation"]      = ensure_list(kw.get("situation"))
    kw["health_tags"]    = ensure_list(kw.get("health_tags"))
    kw["weather_tags"]   = ensure_list(kw.get("weather_tags"))
    kw["menu_style"]     = ensure_list(kw.get("menu_style"))
    kw["extra_keywords"] = ensure_list(kw.get("extra_keywords"))

    # --- 4) 프롬프트 char n-grams 만들기 ---
    # free_text가 있다면 그걸 우선 사용 (없으면 그냥 빈 문자열)
    prompt_text = kw.get("free_text") or ""
    prompt_ngrams = make_char_ngrams(prompt_text, min_n=2, max_n=4)
    prompt_ngrams_norm = [g.lower() for g in prompt_ngrams]  # 이미 lower지만 혹시 몰라서

    kw["prompt_ngrams"] = prompt_ngrams_norm  # 디버깅/설명용

    # --- 5) Cypher 파라미터 구성 ---
    params = {
        "must_ings":       kw["must_ingredients"],
        "opt_ings":        kw["optional_ingredients"],
        "exclude_ings":    kw["exclude_ingredients"],
        "dish_type":       kw["dish_type"],
        "method_list":     kw["method"],
        "situation_list":  kw["situation"],
        "health_list":     kw["health_tags"],
        "weather_list":    kw["weather_tags"],
        "menu_style_list": kw["menu_style"],
        "extra_kw_list":   kw["extra_keywords"],
        "prompt_ngrams":   prompt_ngrams_norm,
        "max_time":        kw.get("max_cook_time_min", None),
        "limit_number":    limit,
    }

    # --- 6) Cypher 쿼리 (prompt_ngrams 까지 반영) ---
    cypher = """
MATCH (r:RecipeV2)
OPTIONAL MATCH (r)-[:HAS_INGREDIENT_V2]->(ing:IngredientV2)
OPTIONAL MATCH (r)-[:IN_CATEGORY_V2]->(cat:CategoryV2)
OPTIONAL MATCH (r)-[:COOKED_BY_V2]->(meth:MethodV2)
OPTIONAL MATCH (r)-[:FOR_SITUATION_V2]->(sit:SituationV2)
OPTIONAL MATCH (r)-[:HAS_HEALTH_TAG]->(h:HealthTag)
OPTIONAL MATCH (r)-[:HAS_WEATHER_TAG]->(w:WeatherTag)
OPTIONAL MATCH (r)-[:HAS_MENU_STYLE]->(ms:MenuStyle)
OPTIONAL MATCH (r)-[:HAS_EXTRA_KEYWORD]->(ek:ExtraKeyword)
WITH r,
     collect(DISTINCT ing.name) AS ingRaw,
     collect(DISTINCT cat.name) AS catRaw,
     collect(DISTINCT meth.name) AS methodRaw,
     collect(DISTINCT sit.name) AS sitRaw,
     collect(DISTINCT h.name) AS healthRaw,
     collect(DISTINCT w.name) AS weatherRaw,
     collect(DISTINCT ms.name) AS menuStyleRaw,
     collect(DISTINCT ek.name) AS extraRaw

// --- 공백 제거 + 소문자 리스트로 정규화 ---
WITH r,
     [x IN ingRaw       | replace(toLower(x), " ", "")] AS ingList,
     [x IN catRaw       | replace(toLower(x), " ", "")] AS catList,
     [x IN methodRaw    | replace(toLower(x), " ", "")] AS methodList,
     [x IN sitRaw       | replace(toLower(x), " ", "")] AS sitList,
     [x IN healthRaw    | replace(toLower(x), " ", "")] AS healthList,
     [x IN weatherRaw   | replace(toLower(x), " ", "")] AS weatherList,
     [x IN menuStyleRaw | replace(toLower(x), " ", "")] AS menuStyleList,
     [x IN extraRaw     | replace(toLower(x), " ", "")] AS extraList

// ----------- HARD FILTERS (must/exclude 재료, 시간) -------------
WHERE (
    size($must_ings) = 0
    OR ANY(ing IN $must_ings WHERE
            ANY(mi IN ingList WHERE mi CONTAINS replace(toLower(ing), " ", "")))
)
AND (
    size($exclude_ings) = 0
    OR NONE(ex IN $exclude_ings WHERE
            ANY(mi IN ingList WHERE mi CONTAINS replace(toLower(ex), " ", "")))
)
AND (
    $max_time IS NULL
    OR r.time_min <= $max_time
)

// ----------- SCORING (LLM 키워드 매칭 + prompt_ngrams 매칭) -------------
WITH
    r,
    ingList, catList, methodList, sitList, healthList, weatherList, menuStyleList, extraList,

    // 1) must ingredients (프롬프트 n-gram은 개입 X, 진짜 must 키워드만)
    size([
        ing IN $must_ings
        WHERE ANY(mi IN ingList
                  WHERE mi CONTAINS replace(toLower(ing), " ", ""))
    ]) * 5 AS score_must_ing,

    // 2) optional ingredients + prompt n-gram 기반 재료 매칭
    (
        size([
            ing IN $opt_ings
            WHERE ANY(mi IN ingList
                      WHERE mi CONTAINS replace(toLower(ing), " ", ""))
        ]) +
        size([
            pg IN $prompt_ngrams
            WHERE ANY(mi IN ingList
                      WHERE mi CONTAINS pg)
        ])
    ) * 2 AS score_opt_ing,

    // 3) dish_type (CategoryV2) + prompt n-gram
    (
        size([
            dt IN $dish_type
            WHERE ANY(cat IN catList
                      WHERE cat CONTAINS replace(toLower(dt), " ", ""))
        ]) +
        size([
            pg IN $prompt_ngrams
            WHERE ANY(cat IN catList
                      WHERE cat CONTAINS pg)
        ])
    ) * 3 AS score_dish_type,

    // 4) method (MethodV2) + prompt n-gram
    (
        size([
            mt IN $method_list
            WHERE ANY(m IN methodList
                      WHERE m CONTAINS replace(toLower(mt), " ", ""))
        ]) +
        size([
            pg IN $prompt_ngrams
            WHERE ANY(m IN methodList
                      WHERE m CONTAINS pg)
        ])
    ) * 2 AS score_method,

    // 5) situation (SituationV2) + prompt n-gram
    (
        size([
            st IN $situation_list
            WHERE ANY(s IN sitList
                      WHERE s CONTAINS replace(toLower(st), " ", ""))
        ]) +
        size([
            pg IN $prompt_ngrams
            WHERE ANY(s IN sitList
                      WHERE s CONTAINS pg)
        ])
    ) * 4 AS score_situation,

    // 6) health_tags (HealthTag) + prompt n-gram
    (
        size([
            ht IN $health_list
            WHERE ANY(h IN healthList
                      WHERE h CONTAINS replace(toLower(ht), " ", "")
                         OR replace(toLower(ht), " ", "") CONTAINS h)
        ]) +
        size([
            pg IN $prompt_ngrams
            WHERE ANY(h IN healthList
                      WHERE h CONTAINS pg)
        ])
    ) * 5 AS score_health,

    // 7) weather_tags (WeatherTag) + prompt n-gram
    (
        size([
            wt IN $weather_list
            WHERE ANY(w IN weatherList
                      WHERE w CONTAINS replace(toLower(wt), " ", ""))
        ]) +
        size([
            pg IN $prompt_ngrams
            WHERE ANY(w IN weatherList
                      WHERE w CONTAINS pg)
        ])
    ) * 3 AS score_weather,

    // 8) menu_style (MenuStyle) + prompt n-gram
    (
        size([
            ms IN $menu_style_list
            WHERE ANY(m IN menuStyleList
                      WHERE m CONTAINS replace(toLower(ms), " ", ""))
        ]) +
        size([
            pg IN $prompt_ngrams
            WHERE ANY(m IN menuStyleList
                      WHERE m CONTAINS pg)
        ])
    ) * 2 AS score_menu_style,

    // 9) extra_keywords (ExtraKeyword) + prompt n-gram
    (
        size([
            ek IN $extra_kw_list
            WHERE ANY(e IN extraList
                      WHERE e CONTAINS replace(toLower(ek), " ", "")
                         OR replace(toLower(ek), " ", "") CONTAINS e)
        ]) +
        size([
            pg IN $prompt_ngrams
            WHERE ANY(e IN extraList
                      WHERE e CONTAINS pg)
        ])
    ) * 3 AS score_extra

WITH
    r,
    score_must_ing,
    score_opt_ing,
    score_dish_type,
    score_method,
    score_situation,
    score_health,
    score_weather,
    score_menu_style,
    score_extra,
    (
        score_must_ing +
        score_opt_ing +
        score_dish_type +
        score_method +
        score_situation +
        score_health +
        score_weather +
        score_menu_style +
        score_extra
    ) AS score

RETURN
    r.recipe_id   AS recipe_id,
    r.title       AS title,
    r.name        AS name,
    r.views       AS views,
    r.time_min    AS time_min,
    r.difficulty  AS difficulty,
    r.servings    AS servings,
    score,
    score_must_ing,
    score_opt_ing,
    score_dish_type,
    score_method,
    score_situation,
    score_health,
    score_weather,
    score_menu_style,
    score_extra,
    r.image_url AS image_url
ORDER BY score DESC, r.views DESC
LIMIT $limit_number
    """

    return cypher, params, kw


def graph_rag_search_with_scoring_explanation(user_prompt: str, top_k: int = 5):
    """
    - user_prompt: 사용자가 입력한 자연어 프롬프트
    - top_k: 상위 몇 개 레시피를 볼지

    반환:
      {
        "keywords": 추출된 키워드 dict (정규화 포함),
        "recipes": [
            {
                "recipe_id": ...,
                "title": ...,
                "name": ...,
                "views": ...,
                "time_min": ...,
                "difficulty": ...,
                "servings": ...,
                "score": ...,
                "score_must_ing": ...,
                "score_opt_ing": ...,
                "score_dish_type": ...,
                "score_method": ...,
                "score_situation": ...,
                "score_health": ...,
                "score_weather": ...,
                "score_menu_style": ...,
                "score_extra": ...,
                "summary": "...",
                "explanation_lines": [...],
                "graph_tags": {   # 이 레시피에 실제로 달려 있는 태그들
                    "ingredients": [...],
                    "categories": [...],
                    "methods": [...],
                    "situations": [...],
                    "health_tags": [...],
                    "weather_tags": [...],
                    "menu_styles": [...],
                    "extra_keywords": [...],
                },
                "match_debug": {  # 각 차원별 매칭 상세
                    "dish_type": {...},
                    "method": {...},
                    "situation": {...},
                    "health": {...},
                    "weather": {...},
                    "menu_style": {...},
                    "extra": {...},
                },
            },
            ...
        ]
      }
    """

    print("\n" + "=" * 80)
    print("USER PROMPT:", user_prompt)

    # 1) 키워드 추출
    raw_kw = extract_keywords(user_prompt)

    # (디버깅용) JSON 스키마 전체 출력
    print("\n=== [1] Extracted Keywords (raw) ===")
    print(json.dumps(raw_kw, ensure_ascii=False, indent=2))

    # Cypher 및 정규화 kw 생성
    cypher, params, norm_kw = build_cypher_from_keywords_relaxed(raw_kw, limit=50)
    kw = norm_kw

    print("\n=== [2] Generated Cypher ===")
    print(cypher)
    print("\nParams:", params)

    # 2) Neo4j에서 후보 레시피 검색
    with driver.session() as session:
        result = session.run(cypher, **params)
        rows = list(result)

    if not rows:
        print("\n⚠️ 조건에 맞는 레시피가 없습니다.")
        return {"keywords": kw, "recipes": []}

    # 레시피별 태그 리스트를 다시 가져오기 위한 쿼리
    recipe_detail_query = """
    MATCH (r:RecipeV2 {recipe_id: $rid})
    OPTIONAL MATCH (r)-[:HAS_INGREDIENT_V2]->(ing:IngredientV2)
    OPTIONAL MATCH (r)-[:IN_CATEGORY_V2]->(cat:CategoryV2)
    OPTIONAL MATCH (r)-[:COOKED_BY_V2]->(meth:MethodV2)
    OPTIONAL MATCH (r)-[:FOR_SITUATION_V2]->(sit:SituationV2)
    OPTIONAL MATCH (r)-[:HAS_HEALTH_TAG]->(h:HealthTag)
    OPTIONAL MATCH (r)-[:HAS_WEATHER_TAG]->(w:WeatherTag)
    OPTIONAL MATCH (r)-[:HAS_MENU_STYLE]->(ms:MenuStyle)
    OPTIONAL MATCH (r)-[:HAS_EXTRA_KEYWORD]->(ek:ExtraKeyword)
    RETURN
        r.image_url AS image_url,
        collect(DISTINCT ing.name) AS ingList,
        collect(DISTINCT cat.name) AS catList,
        collect(DISTINCT meth.name) AS methodList,
        collect(DISTINCT sit.name) AS sitList,
        collect(DISTINCT h.name) AS healthList,
        collect(DISTINCT w.name) AS weatherList,
        collect(DISTINCT ms.name) AS menuStyleList,
        collect(DISTINCT ek.name) AS extraList
    """

    recipes = []
    print(f"\n=== [3] Top {top_k} results with scoring explanation ===")

    # prompt n-gram도 kw 안에 있다 (build_cypher에서 넣어둠)
    prompt_ngrams = kw.get("prompt_ngrams", [])

    for i, rec in enumerate(rows[:top_k], start=1):
        recipe_id = rec["recipe_id"]

        # 레시피별 태그 리스트 가져오기
        with driver.session() as session:
            detail = session.run(recipe_detail_query, {"rid": recipe_id}).single()

        ingList       = detail["ingList"] or []
        catList       = detail["catList"] or []
        methodList    = detail["methodList"] or []
        sitList       = detail["sitList"] or []
        healthList    = detail["healthList"] or []
        weatherList   = detail["weatherList"] or []
        menuStyleList = detail["menuStyleList"] or []
        extraList     = detail["extraList"] or []

        # 기본 정보
        r_info = {
            "recipe_id": recipe_id,
            "title": rec["title"],
            "name": rec["name"],
            "views": rec["views"],
            "time_min": rec["time_min"],
            "difficulty": rec["difficulty"],
            "servings": rec["servings"],
            "score": rec["score"],
            "score_must_ing": rec["score_must_ing"],
            "score_opt_ing": rec["score_opt_ing"],
            "score_dish_type": rec["score_dish_type"],
            "score_method": rec["score_method"],
            "score_situation": rec["score_situation"],
            "score_health": rec["score_health"],
            "score_weather": rec["score_weather"],
            "score_menu_style": rec["score_menu_style"],
            "score_extra": rec["score_extra"],
            "graph_tags": {
                "ingredients": ingList,
                "categories": catList,
                "methods": methodList,
                "situations": sitList,
                "health_tags": healthList,
                "weather_tags": weatherList,
                "menu_styles": menuStyleList,
                "extra_keywords": extraList,
            },
            "image_url": rec["image_url"]
        }

        # --- 각 차원별 매칭 디버깅 (점수 분해) ---
        match_debug = {}

        # dish_type
        match_debug["dish_type"] = analyze_match_dimension(
            kw_list=kw.get("dish_type", []),
            graph_list=catList,
            prompt_ngrams=prompt_ngrams,
            weight=3,
            symmetric_kw=False,
        )

        # method
        match_debug["method"] = analyze_match_dimension(
            kw_list=kw.get("method", []),
            graph_list=methodList,
            prompt_ngrams=prompt_ngrams,
            weight=2,
            symmetric_kw=False,
        )

        # situation
        match_debug["situation"] = analyze_match_dimension(
            kw_list=kw.get("situation", []),
            graph_list=sitList,
            prompt_ngrams=prompt_ngrams,
            weight=4,
            symmetric_kw=False,
        )

        # health_tags (양쪽 substring 허용)
        match_debug["health"] = analyze_match_dimension(
            kw_list=kw.get("health_tags", []),
            graph_list=healthList,
            prompt_ngrams=prompt_ngrams,
            weight=5,
            symmetric_kw=True,
        )

        # weather_tags
        match_debug["weather"] = analyze_match_dimension(
            kw_list=kw.get("weather_tags", []),
            graph_list=weatherList,
            prompt_ngrams=prompt_ngrams,
            weight=3,
            symmetric_kw=False,
        )

        # menu_style
        match_debug["menu_style"] = analyze_match_dimension(
            kw_list=kw.get("menu_style", []),
            graph_list=menuStyleList,
            prompt_ngrams=prompt_ngrams,
            weight=2,
            symmetric_kw=False,
        )

        # extra_keywords (양쪽 substring 허용)
        match_debug["extra"] = analyze_match_dimension(
            kw_list=kw.get("extra_keywords", []),
            graph_list=extraList,
            prompt_ngrams=prompt_ngrams,
            weight=3,
            symmetric_kw=True,
        )

        r_info["match_debug"] = match_debug

        # --- 설명 문자열 구성 ---
        expl_lines = []

        # 하드 필터 설명
        if kw.get("must_ingredients"):
            expl_lines.append(
                f"- 필수 재료(must_ingredients={kw['must_ingredients']}) 모두 포함 → 하드 필터 통과"
            )
        if kw.get("exclude_ingredients"):
            expl_lines.append(
                f"- 제외 재료(exclude_ingredients={kw['exclude_ingredients']})는 포함되지 않음 → 하드 필터 통과"
            )
        if kw.get("max_cook_time_min"):
            max_t = kw["max_cook_time_min"]
            cur_t = r_info["time_min"]
            if cur_t is not None and cur_t <= max_t:
                expl_lines.append(f"- 최대 조리시간 {max_t}분 조건 만족 (현재 {cur_t}분)")
            else:
                expl_lines.append(
                    f"- 최대 조리시간 {max_t}분 조건 미충족일 수 있음 (time_min={cur_t})"
                )

        # 각 차원별 점수/매칭 상세 (예: extra_keywords)
        def add_dim_expl(dim_name, label, score_key, debug_key):
            dbg = match_debug[debug_key]
            score = r_info[score_key]
            if score == 0 and not dbg["kw_values"] and not dbg["prompt_ngrams"]:
                # 아무 정보도 없으면 생략
                return
            expl_lines.append(
                f"- [{label}] 점수 {score}점"
                f" (LLM 키워드 매칭 {dbg['count_kw']}개, 프롬프트 n-gram 매칭 {dbg['count_prompt']}개)"
            )
            if dbg["kw_values"]:
                expl_lines.append(
                    f"   · LLM {label} 키워드: {dbg['kw_values']}"
                )
            if dbg["kw_match_detail"]:
                expl_lines.append(
                    f"   · LLM 키워드↔그래프 태그 매칭: {dbg['kw_match_detail']}"
                )
            if dbg["prompt_match_detail"]:
                expl_lines.append(
                    f"   · 프롬프트 n-gram↔그래프 태그 매칭: {dbg['prompt_match_detail']}"
                )

        add_dim_expl("dish_type", "dish_type(CategoryV2)", "score_dish_type", "dish_type")
        add_dim_expl("method", "method(MethodV2)", "score_method", "method")
        add_dim_expl("situation", "situation(SituationV2)", "score_situation", "situation")
        add_dim_expl("health", "health_tags(HealthTag)", "score_health", "health")
        add_dim_expl("weather", "weather_tags(WeatherTag)", "score_weather", "weather")
        add_dim_expl("menu_style", "menu_style(MenuStyle)", "score_menu_style", "menu_style")
        add_dim_expl("extra", "extra_keywords(ExtraKeyword)", "score_extra", "extra")

        # 요약 라인
        summary_line = (
            f"총점 {r_info['score']}점 "
            f"(must={r_info['score_must_ing']}, "
            f"opt={r_info['score_opt_ing']}, "
            f"dish={r_info['score_dish_type']}, "
            f"method={r_info['score_method']}, "
            f"situation={r_info['score_situation']}, "
            f"health={r_info['score_health']}, "
            f"weather={r_info['score_weather']}, "
            f"style={r_info['score_menu_style']}, "
            f"extra={r_info['score_extra']})"
        )

        # 콘솔 출력
        print(
            f"\n[{i}] ({r_info['recipe_id']}) {r_info['title']}  | 이름: {r_info['name']}"
        )
        print(
            f"     - 조리시간: {r_info['time_min']}분 | 난이도: {r_info['difficulty']} | 조회수: {r_info['views']}"
        )
        print("     -", summary_line)
        for line in expl_lines:
            print("       ", line)

        r_info["summary"] = summary_line
        r_info["explanation_lines"] = expl_lines
        recipes.append(r_info)

    return {
        "keywords": kw,
        "recipes": recipes,
    }