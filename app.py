from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from model.recommender import get_recommendations, get_suggestions

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/suggest', methods=['GET'])
def api_suggest():
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify([])
    
    suggestions = get_suggestions(query, limit=10)
    return jsonify(suggestions)

@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    perfume_name = request.args.get('perfume')
    gender_filter = request.args.get('gender')
    brand_filter = request.args.get('brand')
    
    if not perfume_name:
        return jsonify({"error": "Please provide a perfume name or ingredient"}), 400

    results = get_recommendations(
        perfume_name, 
        gender_filter=gender_filter, 
        brand_filter=brand_filter
    )
    
    if results["search_type"] == "error":
        return jsonify({"error": results["message"]}), 404
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)