from flask import Flask, jsonify, request
from flask_cors import CORS
from model.recommender import get_recommendations
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    perfume_name = request.args.get('perfume')
    
    if not perfume_name:
        return jsonify({"error": "Please provide a perfume name"}), 400

    raw_results = get_recommendations(perfume_name)
    
    if isinstance(raw_results, list) and len(raw_results) > 0 and isinstance(raw_results[0], str):
        return jsonify({"error": raw_results[0]}), 404

    formatted_results = []
    for row in raw_results:
        formatted_results.append({
            "name": row[0],
            "brand": row[1],
            "image": row[2]
        })
    
    return jsonify({
        "searched_for": perfume_name,
        "recommendations": formatted_results
    })

if __name__ == '__main__':
    app.run(debug=True)