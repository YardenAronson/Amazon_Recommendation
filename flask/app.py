from flask import Flask, request, jsonify, render_template,  redirect, session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS, ALSModel
from pymongo import MongoClient
import os



app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Replace 'mongodb_uri' with your MongoDB connection string
mongodb_uri = 'mongodb://localhost:27017/'

# Create a MongoClient object
client = MongoClient(mongodb_uri)

# Access the database
db = client['Amazon']

# Access a specific collections
meta = db['meta']
user_recommendations = db['user_recommendations']
product_recommendations = db['product_recommendations']
users = db['users']
top_products = db['top_products']
products = db['products']
stats = db['stats']
top10 = []

for p in top_products.find():
    parent_asin = p.get('parent_asin')
    if parent_asin:
        meta_doc = meta.find_one({'parent_asin': parent_asin})
        if meta_doc:
            top10.append(meta_doc)


@app.route('/index')
def index():
    my_list = []

    # Find recommendations for the product
    recommendations = user_recommendations.find_one({'num_id': session["user_num"]}, {'_id': 0, 'recommendations': 1})
    
    # if not recommendations or 'recommendations' not in recommendations:
    #     return render_template("product.html", product=product,top10=[])
    
    
    for p in recommendations['recommendations']:
        recommended_asin = p.get('num_product')
        if recommended_asin:
            asin_doc = products.find_one({'num': recommended_asin}, {'parent_asin': 1})
            if asin_doc:
                parent_asin = asin_doc.get('parent_asin')
                if parent_asin:
                    meta_doc = meta.find_one({'parent_asin': parent_asin}, {'_id': 0})
                    if meta_doc:
                        my_list.append(meta_doc)


    return render_template("index.html", top10=top10, my_list=my_list)



@app.route('/<parent_asin>', methods=['GET', 'POST'])
def show_product(parent_asin):
    if request.method == 'GET':
        # Find the main product by parent_asin
        product = meta.find_one({'parent_asin': parent_asin}, {'_id': 0})

        if not product:
            return "Product not found", 404

        # Retrieve the num field from products
        product_details = products.find_one({'parent_asin': parent_asin}, {'_id': 0, 'num': 1})
        if not product_details or 'num' not in product_details:
            return "Product details not found", 404

        num = product_details['num']

        # Find recommendations for the product
        recommendations = product_recommendations.find_one({'product1': num}, {'_id': 0, 'recommendations': 1})
        
        if not recommendations or 'recommendations' not in recommendations:
            return render_template("product.html", product=product, top10=[])

        product_pred = []

        for p in recommendations['recommendations']:
            if p:
                asin_doc = products.find_one({'num': p}, {'parent_asin': 1})
                if asin_doc:
                    recommended_asin = asin_doc.get('parent_asin')
                    if recommended_asin:
                        meta_doc = meta.find_one({'parent_asin': recommended_asin}, {'_id': 0})
                        if meta_doc:
                            product_pred.append(meta_doc)

        return render_template("product.html", product=product, top10=product_pred)

    # Handle POST requests if necessary
    '''else:
        user_id = User.query.filter_by(name=session['user_name']).first().id
        comment = request.form.get('text').strip()
        team_id = Team.query.filter_by(name=team_name).first().id
        rating = request.form.get('rate')
        review_obj = Review(user_id=user_id, team_id=team_id, rating=rating, comment=comment)
        db.session.add(review_obj)
        db.session.commit()
    '''
    return redirect("/index")



@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        user_name = int(request.form.get("user_name"))
        print(user_name)
        user = users.find_one({'num': user_name})
        print(user)
        if user:
            session["user_num"] = user['num']
            return redirect("/index")
        else:
            return "Incorrect username!"
    else:
        return render_template("login.html")


@app.route("/signup/", methods=["POST", "GET"])
def signup():
    if request.method == "POST":
        user_name = request.form.get("user_name")
        in_users = users.find_one({"user_id":user_name})
        if not in_users:
            num = stats.find_one().get('user_max')
            print(num)
            return redirect("/")
        else:
            return "User name already exist!"
       

    else:
        return render_template("Signup.html")


if __name__ == '__main__':
    app.run(debug=True, port=5001)
