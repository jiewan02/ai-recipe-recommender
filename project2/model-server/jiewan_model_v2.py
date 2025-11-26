import re
import math
import time
import random
import json
from neo4j import GraphDatabase
from new_extractor_model import extract_keywords
# from park_extractor_model import extract_keywords

# Neo4j 연결 (네 환경에 맞게 수정)
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def normalize_basic(text: str) -> str:
    """공백/특수문자 제거 + 소문자. 한글/영문/숫자만 남김."""
    if not isinstance(text, str):
        text = str(text)
    t = re.sub(r"[^0-9A-Za-z가-힣]", "", text)
    return t.lower()

def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        if not x.strip():
            return []
        return [x]
    return list(x)


def canonicalize_ingredient_list(lst):
    """
    재료 캐노니컬라이징 
    """
    out = []
    for s in lst:
        if not s:
            continue
        s = s.strip().lower()
        if not s:
            continue
        out.append(s)
    # 중복 제거 + 순서 유지
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def softmax(scores, temperature: float = 1.0):
    """온도 조절 가능한 softmax"""
    if not scores:
        return []
    max_s = max(scores)
    exps = [math.exp((s - max_s) / temperature) for s in scores]
    Z = sum(exps)
    if Z == 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / Z for e in exps]

DIFFICULTY_MAP = {
    "쉬운": ["아무나", "초급"],
    "간단": ["아무나", "초급"],
    "쉽": ["아무나", "초급"],
    "초보": ["아무나", "초급"],
    "입문": ["아무나", "초급"],
    "초급": ["초급"],
    "중급": ["중급"],
    "고급": ["고급"],
    "어려운": ["고급", "중급"],
    "힘든": ["고급", "중급"],
}
def normalize_difficulty(raw_kw):
    """
    LLM이 difficulty를 못 잡거나 '쉬운' 같은 단어를 잡았을 때
    그래프 난이도로 자동 매핑
    """
    diffs = raw_kw.get("difficulty", [])
    text = raw_kw.get("free_text", "").lower()

    detected = set()

    # 1) LLM이 직접 추출한 난이도
    for d in diffs:
        d = d.lower()
        for key, mapped in DIFFICULTY_MAP.items():
            if key in d:
                detected.update(mapped)

    # 2) free_text 기반 자동 탐지
    for key, mapped in DIFFICULTY_MAP.items():
        if key in text:
            detected.update(mapped)

    return list(detected)



def get_all_user_keywords(kw: dict):
    """
    LLM이 추출한 모든 의미 있는 키워드를 하나의 flat 리스트로 합친다.
    (재료 · 태그 · 난이도 · 인분수 · 매운맛 등)
    """

    flat = []

    # 1) 기본 리스트 필드들
    list_fields = [
        "must_ingredients",
        "optional_ingredients",
        "exclude_ingredients",
        "dish_type",
        "method",
        "situation",
        "health_tags",
        "weather_tags",
        "menu_style",
        "extra_keywords",
        "difficulty",
        "positive_tags",
        "negative_tags"
    ]

    for lf in list_fields:
        vals = kw.get(lf, [])
        if vals:
            flat.extend([v for v in vals if v])

        
    # Servings
    serv = kw.get("servings", {})
    serv_min = serv.get("min")
    serv_max = serv.get("max")
    if serv_min is not None:
        flat.append(f"servings_min:{serv_min}")
    if serv_max is not None:
        flat.append(f"servings_max:{serv_max}")

    # 4) dietary constraints
    dc = kw.get("dietary_constraints", {})
    for key, val in dc.items():
        if val is True:
            flat.append(f"diet:{key}")

    # 5) cook time
    max_t = kw.get("max_cook_time_min")
    if max_t:
        flat.append(f"max_time:{max_t}")

    # 순서 유지 + 중복 제거
    seen = set()
    out = []
    for x in flat:
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out

# ----



#----
def build_cypher_from_keywords_relaxed(kw: dict, filterKeywords: list = [], limit: int = 50):
    """
    - difficulty, dietary constraints, servings 모두 반영
    - servings_min / servings_max property 사용 안 함 
    - servings 문자열에서 숫자 추출 후 필터/스코어링 적용
    """

    kw = dict(kw)
    ex = kw["exclude_ingredients"]
    ## filterKeywords["include"]
    for ing in ex:
        filterKeywords["include"] = list(filter(lambda x: x["name"] != ing,filterKeywords["include"]))

    ## filterKeywords["include"] -> list
        ## {name: "", field: "", status: ignore|include }
    for item in filterKeywords["include"]:
        name = item["name"]
        field = item["field"]
        state = item["state"]

        if state == "include":
            kw[field].append(name.strip())

    ## filterKeywords["exclude"] -> list
        ## {name: "", field: "", state: ignore|exclude }

    for item in filterKeywords["exclude"]:
        name = item["name"]
        field = item["field"]
        state = item["state"]

        if state == "exclude":
            # if kw[] 모든 필드 except exclude_ing 에  name이 없다면
            exclude = True
            for k, value in kw.items():
                if type(value) == list and name in value:
                    exclude = False
                    break
            if exclude:
                kw[field].append(name.strip())

    # -----------------------------
    # 리스트 정규화
    # -----------------------------
    list_keys = [
        "dish_type", "method", "situation",
        "must_ingredients", "optional_ingredients", "exclude_ingredients",
        "health_tags", "weather_tags", "menu_style",
        "extra_keywords", "positive_tags", "difficulty"
    ]

    for k in list_keys:
        kw[k] = ensure_list(kw.get(k))

    # 중복 제거
    def unique_preserve(lst):
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    for k in ["dish_type", "method", "situation", "menu_style",
              "extra_keywords", "health_tags", "difficulty"]:
        kw[k] = unique_preserve(kw[k])

    # 재료 canonicalization
    for key in ["must_ingredients", "optional_ingredients", "exclude_ingredients"]:
        if kw[key]:
            kw[key] = canonicalize_ingredient_list(kw[key])
        else:
            kw[key] = []

    # dietary constraints
    dc = kw.get("dietary_constraints", {})
    vegetarian     = bool(dc.get("vegetarian"))
    vegan          = bool(dc.get("vegan"))
    no_beef        = bool(dc.get("no_beef"))
    no_pork        = bool(dc.get("no_pork"))
    no_chicken     = bool(dc.get("no_chicken"))
    no_seafood     = bool(dc.get("no_seafood"))

    # servings
    serv_min = kw.get("servings", {}).get("min")
    serv_max = kw.get("servings", {}).get("max")

    # menu_name_list boost
    menu_names = kw.get("dish_type", []) + kw.get("extra_keywords", [])

    # -----------------------------
    # PARAMS
    # -----------------------------
    params = {
        "must_ings": kw["must_ingredients"],
        "opt_ings": kw["optional_ingredients"],
        "exclude_ings": kw["exclude_ingredients"],

        "dish_type": kw["dish_type"],
        "method_list": kw["method"],
        "situation_list": kw["situation"],
        "health_list": kw["health_tags"],
        "weather_list": kw["weather_tags"],
        "menu_style_list": kw["menu_style"],
        "extra_kw_list": kw["extra_keywords"],
        "difficulty_list": kw["difficulty"],

        "menu_name_list": menu_names,
        "serv_min": serv_min,
        "serv_max": serv_max,

        "vegetarian": vegetarian,
        "vegan": vegan,
        "no_beef": no_beef,
        "no_pork": no_pork,
        "no_chicken": no_chicken,
        "no_seafood": no_seafood,

        "max_time": kw.get("max_cook_time_min", None),
        "limit_number": limit,
    }

    # -----------------------------
    # Cypher (B안 servings 처리 적용)
    # -----------------------------
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

WITH
    r,
    [x IN ingRaw | replace(toLower(x)," ","")] AS ingList,
    [x IN catRaw | replace(toLower(x)," ","")] AS catList,
    [x IN methodRaw | replace(toLower(x)," ","")] AS methodList,
    [x IN sitRaw | replace(toLower(x)," ","")] AS sitList,
    [x IN healthRaw | replace(toLower(x)," ","")] AS healthList,
    [x IN weatherRaw | replace(toLower(x)," ","")] AS weatherList,
    [x IN menuStyleRaw | replace(toLower(x)," ","")] AS menuStyleList,
    [x IN extraRaw | replace(toLower(x)," ","")] AS extraList,

    r.servings AS servings_num
    
// HARD FILTER
WHERE (
    size($must_ings) = 0 OR
    ALL(ing IN $must_ings WHERE ANY(mi IN ingList WHERE mi CONTAINS replace(toLower(ing)," ","")))
)
AND (
    size($exclude_ings) = 0 OR
    NONE(ex IN $exclude_ings WHERE ANY(mi IN ingList WHERE mi CONTAINS replace(toLower(ex)," ","")))
)
AND ($max_time IS NULL OR r.time_min <= $max_time)
AND ($serv_min IS NULL OR servings_num >= $serv_min)
AND ($serv_max IS NULL OR servings_num <= $serv_max)



// dietary constraints
AND (NOT $vegetarian OR NOT ANY(mi IN ingList WHERE mi CONTAINS '소고기' OR mi CONTAINS '돼지고기' OR mi CONTAINS '닭고기' OR mi CONTAINS '해산물'))
AND (NOT $vegan OR NOT ANY(mi IN ingList WHERE mi CONTAINS '소고기' OR mi CONTAINS '돼지고기' OR mi CONTAINS '닭고기' OR mi CONTAINS '해산물' OR mi CONTAINS '계란' OR mi CONTAINS '우유'))
AND (NOT $no_beef OR NOT ANY(mi IN ingList WHERE mi CONTAINS '소고기'))
AND (NOT $no_pork OR NOT ANY(mi IN ingList WHERE mi CONTAINS '돼지고기'))
AND (NOT $no_chicken OR NOT ANY(mi IN ingList WHERE mi CONTAINS '닭고기'))
AND (NOT $no_seafood OR NOT ANY(mi IN ingList WHERE mi CONTAINS '해산물'))

// SCORING
WITH
    r, servings_num,
    ingList, catList, methodList, sitList, healthList, weatherList, menuStyleList, extraList,

    size([ing IN $must_ings WHERE ANY(mi IN ingList WHERE mi CONTAINS replace(toLower(ing)," ",""))]) * 5 AS score_must_ing,
    size([ing IN $opt_ings WHERE ANY(mi IN ingList WHERE mi CONTAINS replace(toLower(ing)," ",""))]) * 2 AS score_opt_ing,

    size([dt IN $dish_type WHERE ANY(cat IN catList WHERE cat CONTAINS replace(toLower(dt)," ",""))]) * 3 AS score_dish_type,
    size([mt IN $method_list WHERE ANY(m IN methodList WHERE m CONTAINS replace(toLower(mt)," ",""))]) * 2 AS score_method,
    size([st IN $situation_list WHERE ANY(s IN sitList WHERE s CONTAINS replace(toLower(st)," ",""))]) * 4 AS score_situation,
    size([ht IN $health_list WHERE ANY(h IN healthList WHERE h CONTAINS replace(toLower(ht)," ","") OR replace(toLower(ht)," ","") CONTAINS h)]) * 5 AS score_health,
    size([wt IN $weather_list WHERE ANY(w IN weatherList WHERE w CONTAINS replace(toLower(wt)," ",""))]) * 3 AS score_weather,
    size([ms IN $menu_style_list WHERE ANY(m IN menuStyleList WHERE m CONTAINS replace(toLower(ms)," ",""))]) * 2 AS score_menu_style,
    size([ek IN $extra_kw_list WHERE ANY(e IN extraList WHERE e CONTAINS replace(toLower(ek)," ","") OR replace(toLower(ek)," ","") CONTAINS e)]) * 3 AS score_extra,

    size([df IN $difficulty_list WHERE toLower(r.difficulty) CONTAINS replace(toLower(df)," ","")]) * 4 AS score_difficulty,

    size([mn IN $menu_name_list
          WHERE toLower(r.name)  CONTAINS replace(toLower(mn)," ","")
             OR toLower(r.title) CONTAINS replace(toLower(mn)," ","")
    ]) * 10 AS score_menu_name,

    CASE
        WHEN $serv_min IS NULL THEN 0
        WHEN servings_num IS NULL THEN 0
        WHEN servings_num = $serv_min THEN 5
        WHEN abs(servings_num - $serv_min) = 1 THEN 3
        ELSE 0
    END AS score_servings


WITH
    r,
    score_must_ing, score_opt_ing, score_dish_type, score_method, score_situation,
    score_health, score_weather, score_menu_style, score_extra,
    score_difficulty, score_menu_name, score_servings,

    (
        score_must_ing + score_opt_ing +
        score_dish_type + score_method + score_situation +
        score_health + score_weather + score_menu_style + score_extra +
        score_difficulty + score_menu_name + score_servings
    ) AS score

RETURN
    r.recipe_id AS recipe_id,
    r.title AS title,
    r.name AS name,
    r.views AS views,
    r.time_min AS time_min,
    r.difficulty AS difficulty,
    r.servings AS servings,
    r.image_url AS image_url,
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
    score_difficulty,
    score_menu_name,
    score_servings
ORDER BY score DESC, r.views DESC
LIMIT $limit_number
"""
    return cypher, params, kw


# ===========
def _norm_tag(s: str) -> str:
    return str(s).replace(" ", "").lower()


def _build_match_dict(llm_list, graph_list):
    result = {}
    norm_graph = [(g, _norm_tag(g)) for g in graph_list]

    for kw in llm_list:
        norm_kw = _norm_tag(kw)
        if not norm_kw:
            continue
        hits = [
            g for (g, ng) in norm_graph
            if norm_kw in ng or ng in norm_kw
        ]
        if hits:
            result[kw] = hits
    return result


# def graph_rag_search_with_scoring_explanation(
#     user_prompt: str,
#     filterKeywords: dict ={},
#     top_k: int = 5,
#     greedy_k: int = 3,          # 점수 그대로 뽑을 개수
#     temperature: float = 1.5,   # softmax 온도 (크면 다양성↑)
# ):
#     print("\n" + "=" * 80)
#     print("USER PROMPT:", user_prompt)
#     print("Filter Keywords", filterKeywords)

#     # 1) 키워드 추출
#     start = time.time()
#     raw_kw = extract_keywords(user_prompt)
#     end = time.time()
#     print(f"⏱️ 작업 소요 시간: {end - start:.4f}초")
#     cypher, params, kw = build_cypher_from_keywords_relaxed(raw_kw, filterKeywords=filterKeywords, limit=50) 
def graph_rag_search_with_scoring_explanation(
    user_prompt: str,
    top_k: int = 5,
    greedy_k: int = 3,          # 점수 그대로 뽑을 개수
    filterKeywords: dict ={},
    temperature: float = 1.5,   # softmax 온도 (크면 다양성↑)
):
    print("\n" + "=" * 80)
    print("USER PROMPT:", user_prompt)

    # 1) 키워드 추출
    start = time.time()
    raw_kw = extract_keywords(user_prompt)
    end = time.time()
    print(f"⏱️ 작업 소요 시간: {end - start:.4f}초")
    raw_kw["difficulty"] = normalize_difficulty(raw_kw)

    cypher, params, kw = build_cypher_from_keywords_relaxed(raw_kw, filterKeywords=filterKeywords, limit=50) 

    # 매칭된 키워드 모두 리스트 목록화
    matched_keywords_only = get_all_user_keywords(raw_kw)

    print("=== 사용자가 요청한 의미 키워드 목록 ===")
    print(matched_keywords_only)

    print("\n=== [2] Generated Cypher ===\n")
    print(cypher)
    print("\nParams:", params)

    # 2) Neo4j에서 상위 50개 후보 가져오기
    with driver.session() as session:
        result = session.run(cypher, **params)
        rows = list(result)

    if not rows:
        print("\n⚠️ 조건에 맞는 레시피가 없습니다.")
        return {"keywords": kw, "recipes": []}
    
    
    # 레시피 기반 프롬프트가 주어지지 않는 경우 (극단적이거나 장난스러운 프롬프트 예방)
    all_zero = all((rec["score"] or 0) == 0 for rec in rows)
    if all_zero:
        print("\n⚠️ 점수 기반으로 추천할 만한 레시피가 없습니다. (모든 후보 score=0)")
        return {
            "keywords": kw,
            "recipes": [],
            "no_result_message": "조회 가능한 메뉴가 없습니다. 프롬프트를 조금 더 구체적으로 입력해 주세요.",
        }

    # 최대 50개만 후보로 사용
    # top_candidates = rows[:50]
    top_candidates = rows

    # # 후보 개수가 top_k보다 적으면 그냥 전부 사용
    # if len(top_candidates) <= top_k:
    #     selected_rows = top_candidates
    # else:
    #     # 2-1) 상위 greedy_k개는 점수 순서 그대로
    #     greedy_k = min(greedy_k, top_k, len(top_candidates))
    #     greedy_part = top_candidates[:greedy_k]

    #     # 2-2) 나머지는 softmax로 다양성 있게 뽑기
    #     diversity_needed = top_k - greedy_k
    #     diversity_pool = top_candidates[greedy_k:]

    #     if diversity_needed <= 0 or not diversity_pool:
    #         selected_rows = greedy_part
    #     else:
    #         # softmax 확률 계산 (score 기반)
    #         scores = [rec["score"] for rec in diversity_pool]
    #         probs = softmax(scores, temperature=temperature)

    #         chosen_idx = []
    #         # 중복 없이 diversity_needed개까지 샘플링
    #         while len(chosen_idx) < diversity_needed and len(chosen_idx) < len(diversity_pool):
    #             r = random.random()
    #             cum = 0.0
    #             for i, p in enumerate(probs):
    #                 cum += p
    #                 if r <= cum:
    #                     if i not in chosen_idx:
    #                         chosen_idx.append(i)
    #                     break

    #         diverse_part = [diversity_pool[i] for i in chosen_idx]
    #         # diverse_part 조합 끝난 직후
    #         selected_rows = greedy_part + diverse_part
# 후보 개수가 top_k보다 적으면 그냥 전부 사용
    if len(top_candidates) <= top_k:
        selected_rows = top_candidates
    else:
        # 2-1) 상위 greedy_k개 (동점 처리 포함)
        greedy_k = min(greedy_k, top_k, len(top_candidates))

        # 상위 greedy_k개의 cut-off score 구하기
        # 정렬된 top_candidates 기준
        cutoff_score = top_candidates[greedy_k - 1]["score"]

        # cutoff score와 같은 문서들 찾기
        # (동점 문서가 greedy_k를 넘을 수 있음)
        tied = [rec for rec in top_candidates if rec["score"] == cutoff_score]

        # greedy_k 내에서 동점 문서가 많이 묶였는지 확인
        # greedy 구간에 속하는 문서들을 추출해야함
        greedy_zone = top_candidates[:greedy_k]

        # greedy_zone 안에서 cutoff_score와 같은 문서들만 모음
        tied_in_greedy_zone = [rec for rec in greedy_zone if rec["score"] == cutoff_score]

        # tied_in_greedy_zone의 개수가 greedy_k를 넘어가는지 검사
        if len(tied_in_greedy_zone) > 1:
            # 동점 그룹이 greedy_k범위에 여러 개 있으면
            # 전체 tied 중에서 greedy_k개를 랜덤 샘플링
            greedy_part = random.sample(tied, greedy_k)
        else:
            # 동점 문제가 없으면 기존대로 상위 greedy_k 사용
            greedy_part = greedy_zone

        # 2-2) 나머지는 softmax로 다양성 있게 뽑기
        diversity_needed = top_k - greedy_k
        diversity_pool = [rec for rec in top_candidates if rec not in greedy_part]

        if diversity_needed <= 0 or not diversity_pool:
            selected_rows = greedy_part
        else:
            # softmax 확률 계산
            scores = [rec["score"] for rec in diversity_pool]
            probs = softmax(scores, temperature=temperature)

            chosen_idx = []
            # 중복 없이 diversity_needed개까지 샘플링
            while len(chosen_idx) < diversity_needed and len(chosen_idx) < len(diversity_pool):
                r = random.random()
                cum = 0.0
                for i, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        if i not in chosen_idx:
                            chosen_idx.append(i)
                        break

            diverse_part = [diversity_pool[i] for i in chosen_idx]

            # 최종 조합
            selected_rows = greedy_part + diverse_part

            # === 메뉴명 기준 중복 제거 ===
            unique_rows = []
            seen_names = set()

            for rec in selected_rows:
                norm_name = rec["name"].replace(" ", "").lower()
                if norm_name not in seen_names:
                    seen_names.add(norm_name)
                    unique_rows.append(rec)

            if len(unique_rows) < top_k:
                needed = top_k - len(unique_rows)

                # 여기에 중복 아닌 추가 후보 채우기
                for rec in top_candidates:
                    norm_name = rec["name"].replace(" ", "").lower()
                    if norm_name not in seen_names:
                        seen_names.add(norm_name)
                        unique_rows.append(rec)
                        if len(unique_rows) == top_k:
                            break

            # top_k 크기 맞춰주기
            selected_rows = unique_rows[:top_k]

            print(f"\n=== [3] Final {len(selected_rows)} results with scoring explanation (Top-{top_k}) ===\n")


    recipes = []

    # 레시피 태그 디테일 쿼리
    recipe_detail_query = """
    MATCH (r:RecipeV2 {recipe_id: $rid})
    OPTIONAL MATCH (r)-[:HAS_HEALTH_TAG]->(h:HealthTag)
    OPTIONAL MATCH (r)-[:HAS_WEATHER_TAG]->(w:WeatherTag)
    OPTIONAL MATCH (r)-[:HAS_MENU_STYLE]->(ms:MenuStyle)
    OPTIONAL MATCH (r)-[:HAS_EXTRA_KEYWORD]->(ek:ExtraKeyword)
    OPTIONAL MATCH (r)-[:FOR_SITUATION_V2]->(sit:SituationV2)
    OPTIONAL MATCH (r)-[:COOKED_BY_V2]->(meth:MethodV2)
    OPTIONAL MATCH (r)-[:IN_CATEGORY_V2]->(cat:CategoryV2)
    RETURN
      collect(DISTINCT h.name)   AS healthList,
      collect(DISTINCT w.name)   AS weatherList,
      collect(DISTINCT ms.name)  AS menuStyleList,
      collect(DISTINCT ek.name)  AS extraList,
      collect(DISTINCT sit.name) AS situationList,
      collect(DISTINCT meth.name) AS methodList,
      collect(DISTINCT cat.name) AS categoryList
    """

    with driver.session() as session:
        for i, rec in enumerate(selected_rows, start=1):
            r_info = {
                "recipe_id": rec["recipe_id"],
                "title": rec["title"],
                "name": rec["name"],
                "views": rec["views"],
                "time_min": rec["time_min"],
                "difficulty": rec["difficulty"],
                "servings": rec.get("servings"),
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
                "score_servings": rec.get("score_servings", 0),
                "score_difficulty": rec["score_difficulty"],     
                "score_menu_name": rec["score_menu_name"],      
                "image_url": rec["image_url"]     
            }


            detail = session.run(recipe_detail_query, {"rid": rec["recipe_id"]}).single()

            categoryList  = detail["categoryList"]   or []
            methodList    = detail["methodList"]     or []
            situationList = detail["situationList"]  or []
            healthList    = detail["healthList"]     or []
            weatherList   = detail["weatherList"]    or []
            menuStyleList = detail["menuStyleList"]  or []
            extraList     = detail["extraList"]      or []

            expl_lines = []

            # ❶ 매칭 정보 구조 저장용 dict
            matched_tag_dict = {}

            # --- 하드 필터 설명 ---
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
                    expl_lines.append(
                        f"- 최대 조리시간 {max_t}분 조건 만족 (현재 {cur_t}분)"
                    )
                else:
                    expl_lines.append(
                        f"- 최대 조리시간 {max_t}분 조건 미충족일 수 있음 (time_min={cur_t})"
                    )

            # --- LLM 키워드 기반 매칭 설명 + 매칭 구조 저장 ---

            if kw.get("dish_type"):
                match_dict = _build_match_dict(kw["dish_type"], categoryList)
                matched_tag_dict["dish_type"] = match_dict
                match_cnt = len(match_dict)
                expl_lines.append(
                    f"- [dish_type(CategoryV2)] 점수 {r_info['score_dish_type']}점 (LLM 키워드 매칭 {match_cnt}개)"
                )
                expl_lines.append(f"   · LLM dish_type(CategoryV2) 키워드: {kw['dish_type']}")
                expl_lines.append(f"   · LLM 키워드↔그래프 태그 매칭: {match_dict}")

            if kw.get("method"):
                match_dict = _build_match_dict(kw["method"], methodList)
                matched_tag_dict["method"] = match_dict
                match_cnt = len(match_dict)
                expl_lines.append(
                    f"- [method(MethodV2)] 점수 {r_info['score_method']}점 (LLM 키워드 매칭 {match_cnt}개)"
                )
                expl_lines.append(f"   · LLM method(MethodV2) 키워드: {kw['method']}")
                expl_lines.append(f"   · LLM 키워드↔그래프 태그 매칭: {match_dict}")

            if kw.get("situation"):
                match_dict = _build_match_dict(kw["situation"], situationList)
                matched_tag_dict["situation"] = match_dict
                match_cnt = len(match_dict)
                expl_lines.append(
                    f"- [situation(SituationV2)] 점수 {r_info['score_situation']}점 (LLM 키워드 매칭 {match_cnt}개)"
                )
                expl_lines.append(f"   · LLM situation(SituationV2) 키워드: {kw['situation']}")
                expl_lines.append(f"   · LLM 키워드↔그래프 태그 매칭: {match_dict}")

            if kw.get("health_tags"):
                match_dict = _build_match_dict(kw["health_tags"], healthList)
                matched_tag_dict["health_tags"] = match_dict
                match_cnt = len(match_dict)
                expl_lines.append(
                    f"- [health_tags(HealthTag)] 점수 {r_info['score_health']}점 (LLM 키워드 매칭 {match_cnt}개)"
                )
                expl_lines.append(f"   · LLM health_tags(HealthTag) 키워드: {kw['health_tags']}")
                expl_lines.append(f"   · LLM 키워드↔그래프 태그 매칭: {match_dict}")

            if kw.get("weather_tags"):
                match_dict = _build_match_dict(kw["weather_tags"], weatherList)
                matched_tag_dict["weather_tags"] = match_dict
                match_cnt = len(match_dict)
                expl_lines.append(
                    f"- [weather_tags(WeatherTag)] 점수 {r_info['score_weather']}점 (LLM 키워드 매칭 {match_cnt}개)"
                )
                expl_lines.append(f"   · LLM weather_tags(WeatherTag) 키워드: {kw['weather_tags']}")
                expl_lines.append(f"   · LLM 키워드↔그래프 태그 매칭: {match_dict}")

            if kw.get("menu_style"):
                match_dict = _build_match_dict(kw["menu_style"], menuStyleList)
                matched_tag_dict["menu_style"] = match_dict
                match_cnt = len(match_dict)
                expl_lines.append(
                    f"- [menu_style(MenuStyle)] 점수 {r_info['score_menu_style']}점 (LLM 키워드 매칭 {match_cnt}개)"
                )
                expl_lines.append(f"   · LLM menu_style(MenuStyle) 키워드: {kw['menu_style']}")
                expl_lines.append(f"   · LLM 키워드↔그래프 태그 매칭: {match_dict}")

            if kw.get("extra_keywords"):
                match_dict = _build_match_dict(kw["extra_keywords"], extraList)
                matched_tag_dict["extra_keywords"] = match_dict
                match_cnt = len(match_dict)
                expl_lines.append(
                    f"- [extra_keywords(ExtraKeyword)] 점수 {r_info['score_extra']}점 (LLM 키워드 매칭 {match_cnt}개)"
                )
                expl_lines.append(f"   · LLM extra_keywords(ExtraKeyword) 키워드: {kw['extra_keywords']}")
                expl_lines.append(f"   · LLM 키워드↔그래프 태그 매칭: {match_dict}")

            if kw.get("difficulty"):
                graph_diff = [r_info["difficulty"]] if r_info["difficulty"] else []
                match_dict = _build_match_dict(kw["difficulty"], graph_diff)

                matched_tag_dict["difficulty"] = match_dict
                match_cnt = len(match_dict)

                expl_lines.append(
                    f"- [difficulty] 점수 {r_info['score_difficulty']}점 "
                    f"(LLM 키워드 매칭 {match_cnt}개)"
                )
                expl_lines.append(f"   · 요청 난이도: {kw['difficulty']}")
                expl_lines.append(f"   · 그래프 난이도: {graph_diff}")
                expl_lines.append(f"   · 매칭 결과: {match_dict}")


            # --- servings(인분수) ---
            # 사용자 요구가 있을 때만 설명 출력
            if kw.get("servings", {}).get("min") or kw.get("servings", {}).get("max"):

                requested_min = kw["servings"].get("min")
                requested_max = kw["servings"].get("max")

                # 그래프에서 실제 인분수
                graph_serv = r_info.get("servings")

                # 매칭 정보 구조 저장
                matched_tag_dict["servings"] = {
                    "requested_min": requested_min,
                    "requested_max": requested_max,
                    "graph_servings": graph_serv
                }

                expl_lines.append(
                    f"- [servings] 점수 {r_info['score_servings']}점"
                    f" (요청 인분수 min={requested_min}, max={requested_max} / 그래프 값={graph_serv})"
                )

                expl_lines.append(
                    f"   · 인분수 조건과의 거리 기반 점수: {r_info['score_servings']}"
                )

            # === ❷ 사용자가 실제로 요청한 의미 있는 모든 키워드 모아두기 ===

            user_keywords_all = (
                kw.get("must_ingredients", [])
                + kw.get("optional_ingredients", [])
                + kw.get("dish_type", [])
                + kw.get("method", [])
                + kw.get("situation", [])
                + kw.get("health_tags", [])
                + kw.get("weather_tags", [])
                + kw.get("menu_style", [])
                + kw.get("extra_keywords", [])
            )

            # 중복 제거 + 순서 유지
            seen_kw = set()
            flat_unique = []
            for k in user_keywords_all:
                if k not in seen_kw:
                    seen_kw.add(k)
                    flat_unique.append(k)

            # 결과 저장
            r_info["matched_keywords_flat"] = flat_unique


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
                f"extra={r_info['score_extra']}, "
                f"difficulty={r_info['score_difficulty']}, "
                f"menu_name={r_info['score_menu_name']}, "
                f"servings={r_info['score_servings']})"
            )



            print(f"[{i}] ({r_info['recipe_id']}) {r_info['title']}  | 이름: {r_info['name']}")
            print(f"     - 조리시간: {r_info['time_min']}분 | 난이도: {r_info['difficulty']} | 조회수: {r_info['views']}")
            print("     -", summary_line)
            for line in expl_lines:
                print("        ", line)

            # ❸ r_info에 매칭 정보 저장
            r_info["summary"] = summary_line
            r_info["explanation_lines"] = expl_lines
            r_info["matched_tag_dict"] = matched_tag_dict
            r_info["matched_keywords_flat"] = flat_unique

            recipes.append(r_info)

    return {
        "keywords": kw,
        "recipes": recipes,
    }
