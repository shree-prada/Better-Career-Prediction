from flask import Flask, render_template, request

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
@app.route("/")
def home():
    return render_template('index.html');

@app.route("/predict")
def predict():
    return render_template('prediction.html')

@app.route("/skills")
def skills():
    return render_template('main.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    sslc= request.form["sslc"]
    hsc= request.form["hsc"]
    cgpa= request.form["cgpa"]
    school_type= request.form["school_type"]
    no_of_miniprojects= request.form["no_of_miniprojects"]
    no_of_projects= request.form["no_of_projects"]
    coresub_skill= request.form["coresub_skill"]
    aptitude_skill= request.form["aptitude_skill"]
    problemsolving_skill= request.form["problemsolving_skill"]
    programming_skill= request.form["programming_skill"]
    abstractthink_skill= request.form["abstractthink_skill"]
    design_skill= request.form["design_skill"]
    first_computer= request.form["first_computer"]
    first_program= request.form["first_program"]
    lab_programs= request.form["lab_programs"]
    ds_coding= request.form["ds_coding"]
    technology_used= request.form["technology_used"]
    sympos_attend= request.form["sympos_attend"]
    sympos_won= request.form["sympos_won"]
    extracurricular= request.form["extracurricular"]
    learning_style= request.form["learning_style"]
    college_bench= request.form["college_bench"]
    clg_teachers_know= request.form["clg_teachers_know"]
    college_performence= request.form["college_performence"]
    college_skills= request.form["college_skills"]

    dataset = pd.read_csv("career_compute_dataset.csv")
    labelencoder = LabelEncoder()
    df = dataset
    label = df.iloc[:49, -1]
    original = label.unique()
    label = label.values
    label2 = labelencoder.fit_transform(label)
    y = pd.DataFrame(label2, columns=["ROLE"])
    numeric = y["ROLE"].unique()
    y1 = pd.DataFrame({'ROLE': original, 'Associated Number': numeric})
    print(y1)

    import pickle
    loaded_model = pickle.load(open("career.pickle.dat", "rb"))
    x_new =[sslc,hsc,cgpa,school_type,no_of_miniprojects,no_of_projects,coresub_skill,aptitude_skill,problemsolving_skill,programming_skill,abstractthink_skill,design_skill,first_computer,first_program,lab_programs,ds_coding,technology_used,sympos_attend,sympos_won,extracurricular,learning_style,college_bench,clg_teachers_know,college_performence,college_skills]
    new_pred = loaded_model.predict([x_new])
    print("Prediction : {}".format(y1[y1['Associated Number'] == new_pred[0]]['ROLE']))
    return render_template('prediction.html',result="Suggested Job Role : {}".format(y1.loc[y1['Associated Number'] ==new_pred[0]].values.tolist()[0][0]))


if __name__ == "__main__":
    app.run(debug=False)


