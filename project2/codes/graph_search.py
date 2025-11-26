# 1. 환경 확인
import torch, transformers, bitsandbytes
from extractor_model_old import extract_keywords
from neo4j import GraphDatabase

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def build_cypher_from_keywords(kw: dict, limit: int = 50):
    where_clauses = []
    params = {}

    # 1) must_ingredients: 반드시 포함해야 하는 재료
    if kw.get("must_ingredients"):
        where_clauses.append("""
        ALL(ing IN $must_ings WHERE
            EXISTS {
                MATCH (r)-[:HAS_INGREDIENT]->(:Ingredient {name: ing})
            }
        )
        """)
        params["must_ings"] = kw["must_ingredients"]

    # 2) exclude_ingredients: 포함되면 안 되는 재료
    if kw.get("exclude_ingredients"):
        where_clauses.append("""
        ALL(ing IN $exclude_ings WHERE
            NOT EXISTS {
                MATCH (r)-[:HAS_INGREDIENT]->(:Ingredient {name: ing})
            }
        )
        """)
        params["exclude_ings"] = kw["exclude_ingredients"]

    # 3) dish_type -> Category (예: 국, 탕, 볶음)
    if kw.get("dish_type"):
        where_clauses.append("""
        EXISTS {
            MATCH (r)-[:IN_CATEGORY]->(c:Category)
            WHERE c.name IN $cats
        }
        """)
        params["cats"] = kw["dish_type"]

    # 4) method -> Method (예: 끓이기, 볶기)
    if kw.get("method"):
        where_clauses.append("""
        EXISTS {
            MATCH (r)-[:COOKED_BY]->(m:Method)
            WHERE m.name IN $methods
        }
        """)
        params["methods"] = kw["method"]

    # 5) situation -> Situation (예: 명절, 야식)
    if kw.get("situation"):
        where_clauses.append("""
        EXISTS {
            MATCH (r)-[:FOR_SITUATION]->(s:Situation)
            WHERE s.name IN $sits
        }
        """)
        params["sits"] = kw["situation"]

    # 6) 난이도
    if kw.get("difficulty"):
        where_clauses.append("r.difficulty IN $diffs")
        params["diffs"] = kw["difficulty"]

    # 7) 최대 조리 시간
    if kw.get("max_cook_time_min"):
        where_clauses.append("r.time_min IS NOT NULL AND r.time_min <= $tmax")
        params["tmax"] = kw["max_cook_time_min"]

    where_str = ""
    if where_clauses:
        where_str = "WHERE " + " AND ".join([w.strip() for w in where_clauses])

    cypher = f"""
    MATCH (r:Recipe)
    {where_str}
    RETURN
        r.recipe_id AS recipe_id,
        r.title AS title,
        r.name AS name,
        r.views AS views,
        r.time_min AS time_min,
        r.difficulty AS difficulty,
        r.servings AS servings
    ORDER BY r.views DESC
    LIMIT {limit}
    """

    return cypher, params

def build_cypher_from_keywords_relaxed(kw: dict, limit: int = 50):
    where_clauses = []
    params = {}

    # 1) must_ingredients: "이 문자열이 이름에 포함되는 재료" 있으면 OK
    ing_filters = []
    if kw.get("must_ingredients"):
        for idx, ing in enumerate(kw["must_ingredients"]):
            key = f"must_ing_{idx}"
            # LOWER 비교도 가능
            clause = f"""
            EXISTS {{
                MATCH (r)-[:HAS_INGREDIENT]->(mi:Ingredient)
                WHERE mi.name CONTAINS ${key}
            }}
            """
            ing_filters.append(clause)
            params[key] = ing  # 예: "떡", "계란"
    if ing_filters:
        where_clauses.append("(" + " AND ".join(ing_filters) + ")")

    # 2) exclude_ingredients: 이름에 포함되면 안 됨
    exc_filters = []
    if kw.get("exclude_ingredients"):
        for idx, ing in enumerate(kw["exclude_ingredients"]):
            key = f"exc_ing_{idx}"
            clause = f"""
            NOT EXISTS {{
                MATCH (r)-[:HAS_INGREDIENT]->(ei:Ingredient)
                WHERE ei.name CONTAINS ${key}
            }}
            """
            exc_filters.append(clause)
            params[key] = ing
    if exc_filters:
        where_clauses.append("(" + " AND ".join(exc_filters) + ")")

    # 3) dish_type은 일단 잠깐 빼거나, 간단 매핑으로 변환
    # 예: "국물요리" -> ["국", "탕", "찌개"]
    # 여기서는 일단 skip (나중에 mapping table로 개선)

    # 4) method/situation도 일단 빼고 재료 기반으로만 필터
    # 나중에 mapping 잘 설계하고 추가

    # 5) 시간 조건 정도는 유지해도 괜찮음
    if kw.get("max_cook_time_min"):
        where_clauses.append("r.time_min IS NOT NULL AND r.time_min <= $tmax")
        params["tmax"] = kw["max_cook_time_min"]

    where_str = ""
    if where_clauses:
        where_str = "WHERE " + " AND ".join([w.strip() for w in where_clauses])

    cypher = f"""
    MATCH (r:Recipe)
    {where_str}
    RETURN
        r.recipe_id AS recipe_id,
        r.title AS title,
        r.name AS name,
        r.views AS views,
        r.time_min AS time_min,
        r.difficulty AS difficulty,
        r.servings AS servings
    ORDER BY r.views DESC
    LIMIT {limit}
    """

    return cypher, params

def graph_rag_search(user_prompt: str, top_k: int = 10):
    # 1) 키워드 추출
    kw = extract_keywords(user_prompt)
    print("=== Extracted keywords ===")
    from pprint import pprint
    pprint(kw)

    # 2) Cypher 쿼리 생성
    cypher, params = build_cypher_from_keywords_relaxed(kw, limit=50)
    print("\n=== Generated Cypher ===")
    print(cypher)
    print("\nParams:", params)

    # 3) Neo4j에서 후보 레시피 검색
    with driver.session() as session:
        result = session.run(cypher, **params)
        rows = list(result)

    # 4) 상위 top_k개만 잘라서 보기 좋게 포맷
    recipes = []
    for rec in rows[:top_k]:
        recipes.append({
            "recipe_id": rec["recipe_id"],
            "title": rec["title"],
            "name": rec["name"],
            "views": rec["views"],
            "time_min": rec["time_min"],
            "difficulty": rec["difficulty"],
            "servings": rec["servings"],
        })

    print(f"\n=== Top {top_k} results ===")
    for i, r in enumerate(recipes, start=1):
        print(f"[{i}] ({r['recipe_id']}) {r['title']} | {r['time_min']}분 | {r['difficulty']} | 조회수 {r['views']}")

    return {
        "keywords": kw,
        "recipes": recipes,
    }

queries = [
    "따뜻하게 먹을 수 있는 국물요리 추천해줘. 냉장고에 떡, 계란이 있고 너무 매콤하지 않았으면 좋겠어.",
    # "돼지고기는 싫고, 소고기도 안 먹어. 채식 위주로 먹고 있는데 담백한 국이나 찌개 추천해줘.",
    # "30분 안에 만들 수 있는 간단한 볶음요리 추천해줘. 마늘은 알레르기라 빼줘.",
]

for q in queries:
    print("\n" + "="*80)
    print("USER PROMPT:", q)
    res = graph_rag_search(q, top_k=50)