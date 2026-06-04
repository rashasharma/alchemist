import time
import sys
import os

print("========================================")
print("ALCHEMIST RECOMMENDATION ENGINE TEST SUITE")
print("========================================\n")

# Measure loading time of the recommendation engine
t0 = time.time()
try:
    from model.recommender import get_recommendations, get_suggestions
    load_time = time.time() - t0
    print(f"[SUCCESS] Loading Success: Engine loaded in {load_time:.4f} seconds.")
    if load_time < 0.3:
        print("   [INFO] Performance Check: EXCELLENT! Startup is under 300ms (Pickled load is working).")
    else:
        print("   [WARNING] Performance Check: Slow startup. Verify serialization loading.")
except Exception as e:
    print(f"[FAIL] Loading Failed: {e}")
    sys.exit(1)

print("\n----------------------------------------")
print("TEST CASE 1: Token-Intersection Matching & Brand De-hyphenation")
print("----------------------------------------")
# Test a query where order is reversed (e.g., "[Brand] [Name]" format which failed previously)
query1 = "Yves Saint Laurent Black Opium"
res1 = get_recommendations(query1)

if res1["search_type"] in ["perfume", "multiple_matches"]:
    print(f"[SUCCESS] Query '{query1}' successfully matched a perfume in the database!")
    print(f"   Matched Seed: {res1['matched_perfume']['name']} by {res1['matched_perfume']['brand']}")
    print(f"   Recommendations returned: {len(res1['recommendations'])}")
else:
    print(f"[FAIL] Search engine could not match the perfume. Type: {res1['search_type']}. Message: {res1.get('message', '')}")

print("\n----------------------------------------")
print("TEST CASE 2: Abbreviation/Synonym Mapping")
print("----------------------------------------")
# Test an abbreviated brand query (YSL)
query2 = "YSL Black Opium"
res2 = get_recommendations(query2)

if res2["search_type"] in ["perfume", "multiple_matches"] and res2["matched_perfume"]["brand"] == "Yves Saint Laurent":
    print(f"[SUCCESS] Abbreviation 'YSL' correctly mapped to 'Yves Saint Laurent'!")
    print(f"   Matched Seed: {res2['matched_perfume']['name']} by {res2['matched_perfume']['brand']}")
else:
    print(f"[FAIL] Synonym mapping failed. Brand matched: {res2.get('matched_perfume', {}).get('brand', 'None')}")

print("\n----------------------------------------")
print("TEST CASE 3: Popularity Autocomplete Suggestions")
print("----------------------------------------")
# Test typing a partial string like "sauvage"
query3 = "sauvage"
suggestions = get_suggestions(query3, limit=5)

if suggestions:
    print(f"[SUCCESS] Retrieved {len(suggestions)} suggestions for '{query3}'.")
    for idx, s in enumerate(suggestions):
        print(f"   {idx+1}. {s['brand']} - {s['name']} (Rating: {s['rating']}, Reviews: {s['reviews']})")
    
    # Check popularity sorting (most reviews should be first)
    reviews_list = [s['reviews'] for s in suggestions]
    if reviews_list == sorted(reviews_list, reverse=True):
        print("   [INFO] Sorting Check: EXCELLENT! Suggestions are correctly ordered by review count popularity.")
    else:
        print("   [WARNING] Sorting Check: Suggestions are not ordered by popularity.")
else:
    print(f"[FAIL] No suggestions returned for '{query3}'.")

print("\n----------------------------------------")
print("TEST CASE 4: Scent Suitability Gender Filtering")
print("----------------------------------------")
# Search recommendations for "Sauvage" with gender filter set to "women"
query4 = "Sauvage"
res4 = get_recommendations(query4, gender_filter="women")

if res4["search_type"] in ["perfume", "multiple_matches"]:
    recs = res4["recommendations"]
    women_only = all(r["gender"].lower() in ["for women", "for women and men"] for r in recs)
    if women_only:
        print(f"[SUCCESS] Gender suitability filter active! Recommendations only contain feminine/unisex scents.")
        print(f"   Sample Recommendation: {recs[0]['brand']} - {recs[0]['name']} ({recs[0]['gender']})")
    else:
        print("[FAIL] Non-feminine scents found in recommendations with women filter active:")
        for r in recs:
            print(f"   - {r['name']} ({r['gender']})")
else:
    print(f"[FAIL] Could not match seed for gender test.")

print("\n========================================")
print("TEST SUITE COMPLETED!")
print("========================================")