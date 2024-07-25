from flask import Flask, request, jsonify, render_template,  redirect, session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS, ALSModel
from pymongo import MongoClient



app = Flask(__name__)
 
# Replace 'mongodb_uri' with your MongoDB connection string
mongodb_uri = 'mongodb://localhost:27017/Amazon_Office'

# Create a MongoClient object
client = MongoClient(mongodb_uri)

# Access the database
db = client['Amazon']

# Access a specific collections
meta = db['meta']
predictions = db['user_predictions']
products = db['products']
users = db['new_users']




@app.route('/')
def index():
    documents = meta.find({}).limit(10)
    top5 = top10 = list(documents)
    return render_template("index.html", top5=top5)


@app.route('/products')
def get_products():
    pipeline = [
        {"$match": {"rating_number": {"$gt": 100}}},
        {"$sort": {"average_rating": -1}},
        {"$limit": 10},
        {"$project": {"_id": 0, "parent_asin": 1, "details": 1, "average_rating": 1, "rating_number": 1}}
    ]

    top_items = list(meta.aggregate(pipeline))
    return jsonify(top_items)



@app.route('/<parent_asin>', methods=['GET', 'POST'])
def show_prudact(parent_asin):
    if request.method == 'GET':
        product = meta.find_one({'parent_asin': parent_asin}, {'_id': 0})
        return render_template("product.html", product=product)
    '''else:
        user_id = User.query.filter_by(name=session['user_name']).first().id
        comment = request.form.get('text').strip()
        team_id = Team.query.filter_by(name=team_name).first().id
        rating = request.form.get('rate')
        review_obj = Review(user_id=user_id, team_id=team_id, rating=rating, comment=comment)
        db.session.add(review_obj)
        db.session.commit()'''
    return redirect("/Homepage")




@app.route("/", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user_name = request.form.get("user_name")
        password = request.form.get("password")
        user = users.find_one({'user_name': user_name, 'password': password})
        if user:
            session["user_name"] = user_name
            return redirect("/Homepage")
        else:
            return "Incorrect password or username!"
    else:
        return render_template("Loginpage.html")


if __name__ == '__main__':
    app.run(debug=True, port=5001)
