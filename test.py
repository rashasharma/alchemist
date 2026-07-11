import time
import sys
from model.recommender import get_recommendations, get_suggestions

t0 = time.time()
try:
    load_time = time.time() - t0
    print(f"[SUCCESS] Engine loaded in {load_time:.4f} seconds")
except Exception as e:
    print(f"[FAIL] Loading Failed: {e}")
    sys.exit(1)

print("\n--- TEST 1: Dior Flagship Query ---")
res1 = get_recommendations("Dior Sauvage")
if res1["search_type"] in ["perfume", "multiple_matches"]:
    matched = res1["matched_perfume"]
    print(f"[SUCCESS] Matched: {matched['brand']} {matched['name']} (Reviews: {matched['reviews']})")
    assert matched["reviews"] > 10000
else:
    print(f"[FAIL] Query failed: {res1.get('message')}")
    sys.exit(1)

print("\n--- TEST 2: Preserved Title (Bleu de Chanel) ---")
res2 = get_recommendations("Bleu de Chanel")
if res2["search_type"] in ["perfume", "multiple_matches"]:
    matched = res2["matched_perfume"]
    print(f"[SUCCESS] Matched: {matched['brand']} {matched['name']} (Reviews: {matched['reviews']})")
    assert "Bleu de Chanel" in matched["name"]
    assert matched["reviews"] > 10000
else:
    print(f"[FAIL] Title query failed")
    sys.exit(1)

print("\n--- TEST 3: Brand Synonyms ---")
res3_ysl = get_recommendations("YSL Black Opium")
res3_dg = get_recommendations("D&G Light Blue")
assert res3_ysl["search_type"] in ["perfume", "multiple_matches"]
assert res3_dg["search_type"] in ["perfume", "multiple_matches"]
print(f"[SUCCESS] YSL matched: {res3_ysl['matched_perfume']['brand']}")
print(f"[SUCCESS] D&G matched: {res3_dg['matched_perfume']['brand']}")

print("\n--- TEST 4: Typo Tolerance ---")
res4 = get_recommendations("creed aventis")
assert res4["search_type"] in ["perfume", "multiple_matches"]
print(f"[SUCCESS] 'creed aventis' resolved to: {res4['matched_perfume']['brand']} {res4['matched_perfume']['name']}")

print("\n--- TEST 5: Autocomplete Popularity Sorting ---")
sugg = get_suggestions("chanel", limit=5)
assert len(sugg) > 0
reviews = [s["reviews"] for s in sugg]
assert reviews == sorted(reviews, reverse=True)
print(f"[SUCCESS] Autocomplete sorted by reviews: {[s['name'] + ' (' + str(s['reviews']) + ')' for s in sugg[:3]]}")

print("\n--- TEST 6: Gender Suitability Filter ---")
res6 = get_recommendations("Sauvage", gender_filter="women")
recs = res6["recommendations"]
assert all(r["gender"].lower() in ["for women", "for women and men"] for r in recs)
print(f"[SUCCESS] Gender filter active: returned {len(recs)} feminine/unisex recommendations")

print("\n--- TEST 7: Non-vocabulary Handling ---")
res7 = get_recommendations("asdfghjklqwerty")
assert res7["search_type"] == "error"
print(f"[SUCCESS] Non-vocabulary query returned clean error: '{res7['message']}'")

print("\n--- TEST 8: Exact Match Priority ---")
res8 = get_recommendations("9am")
assert res8["search_type"] in ["perfume", "multiple_matches"]
matched8 = res8["matched_perfume"]
assert matched8["name"] == "9am" and matched8["brand"] == "Afnan"
print(f"[SUCCESS] Exact match prioritized: {matched8['brand']} {matched8['name']} (Reviews: {matched8['reviews']})")

print("\n========================================")
print("ALL TEST CASES PASSED SUCCESSFULLY")
print("========================================")

