# graph_similarity.py
from neo4j import GraphDatabase

class RecipeGraphSimilarity:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_similar_recipes(
        self,
        recipe_id: int,
        top_n: int = 3,
        min_shared_ings: int = 2,
    ):
        """
        기준 레시피 기준으로:
          - ingredients: 재료 공유 기반 top_n
          - overall: 위에서 찾은 레시피들을 제외하고, 그래프 전체 태그 기반 top_n
        """
        with self.driver.session() as session:
            # 1) 재료 기반 유사 레시피 먼저 찾기
            ingredients = session.execute_read(
                self._query_ingredient_similar,
                recipe_id,
                top_n,
                min_shared_ings,
            )

            # 2) 재료 기반으로 이미 뽑힌 recipe_id 리스트
            exclude_ids = [row["recipe_id"] for row in ingredients]

            # 3) overall에서는 위에서 뽑힌 레시피들을 제외하고 찾기
            overall = session.execute_read(
                self._query_overall_similar,
                recipe_id,
                top_n,
                exclude_ids,
            )

        return {
            "overall": overall,
            "ingredients": ingredients,
        }

    # ==========================================================
    # 1) 전체 그래프 기준 – 관계 타입별 가중치 (재료 제외) + exclude_ids 필터
    # ==========================================================
    @staticmethod
    def _query_overall_similar(tx, recipe_id, top_n, exclude_ids):
        cypher = """
        MATCH (base:RecipeV2 {recipe_id: $recipe_id})

        // base와 연결된 모든 태그 노드 (재료 제외)
        MATCH (base)-[:IN_CATEGORY_V2
                      |COOKED_BY_V2
                      |FOR_SITUATION_V2
                      |HAS_HEALTH_TAG
                      |HAS_WEATHER_TAG
                      |HAS_MENU_STYLE
                      |HAS_EXTRA_KEYWORD]->(t)

        // 같은 태그 t에 연결된 other 레시피
        MATCH (other:RecipeV2)-[r2:IN_CATEGORY_V2
                                |COOKED_BY_V2
                                |FOR_SITUATION_V2
                                |HAS_HEALTH_TAG
                                |HAS_WEATHER_TAG
                                |HAS_MENU_STYLE
                                |HAS_EXTRA_KEYWORD]->(t)
        WHERE other <> base
          AND NOT other.recipe_id IN $exclude_ids   // 재료 기반으로 뽑힌 것들 제외

        WITH other, t, type(r2) AS rel_type

        // 관계 타입별 가중치 합산
        WITH other,
             collect(DISTINCT t.name) AS shared_tags,
             sum(
               CASE rel_type
                 WHEN "FOR_SITUATION_V2"  THEN 4
                 WHEN "HAS_HEALTH_TAG"    THEN 5
                 WHEN "IN_CATEGORY_V2"    THEN 2
                 WHEN "HAS_WEATHER_TAG"   THEN 2
                 WHEN "HAS_MENU_STYLE"    THEN 2
                 WHEN "HAS_EXTRA_KEYWORD" THEN 3
                 ELSE 1
               END
             ) AS similarity_score

        RETURN
            other.recipe_id AS recipe_id,
            other.title     AS title,
            other.name      AS name,
            other.image_url AS image_url,
            similarity_score AS score,
            shared_tags
        ORDER BY score DESC, other.views DESC, title ASC
        LIMIT $top_n;
        """

        result = tx.run(
            cypher,
            recipe_id=recipe_id,
            top_n=top_n,
            exclude_ids=exclude_ids,
        )
        return [dict(record) for record in result]

    # ==========================================================
    # 2) 재료만 기준 – 공유 재료 수 기반 + 최소 재료 공유 필터
    # ==========================================================
    @staticmethod
    def _query_ingredient_similar(tx, recipe_id, top_n, min_shared_ings):
        cypher = """
        MATCH (base:RecipeV2 {recipe_id: $recipe_id})
        MATCH (base)-[:HAS_INGREDIENT_V2]->(ing:IngredientV2)

        MATCH (other:RecipeV2)-[:HAS_INGREDIENT_V2]->(ing)
        WHERE other <> base

        WITH other,
             collect(DISTINCT ing.name) AS shared_ingredients,
             count(DISTINCT ing)        AS shared_ing_count

        // 재료 공유 개수 필터
        WHERE shared_ing_count >= $min_shared_ings

        RETURN
            other.recipe_id       AS recipe_id,
            other.title           AS title,
            other.name            AS name,
            other.image_url            AS image_url,
            shared_ing_count      AS score,
            shared_ingredients
        ORDER BY score DESC, other.views DESC, title ASC
        LIMIT $top_n;
        """

        result = tx.run(
            cypher,
            recipe_id=recipe_id,
            top_n=top_n,
            min_shared_ings=min_shared_ings,
        )
        return [dict(record) for record in result]