from flask import Flask, render_template
import os
import random

app = Flask(__name__)

@app.route('/')
def home():
    image_files = os.listdir('static/generated')
    random_image = random.choice(image_files) if image_files else None
    return render_template('index.html', image_file=random_image)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
