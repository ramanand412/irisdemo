import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# sample Iris prediction app
This app predicts the **iris flower*** type!
""")
st.sidebar.header("User input parameters")

def user_input_features():

    sepel_length=st.sidebar.slider("sepal length",4.3,7.9,5.4)
    sepel_width=st.sidebar.slider("sepal width",2.0,4.4,3.4)
    petel_length=st.sidebar.slider("petal length", 1.0,6.9,1.3)
    petel_width=st.sidebar.slider("petel width",0.1,2.5,0.2)

    data={
         "sepal_length":sepel_length,
         "sepel_width":sepel_width,
         "petel_length":petel_length,
         "petel_width":petel_width

    }
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_features()
st.subheader("usr input parameters")
st.write(df)

iris=datasets.load_iris()
X=iris.data
Y=iris.target

clf=RandomForestClassifier()
clf.fit(X,Y)

prediction=clf.predict(df)
prediction_proba=clf.predict_proba(df)
st.subheader("class label and their corresponding")
st.write(iris.target_names)
st.subheader("prediction")
st.write(iris.target_names[prediction])

st.subheader("prediction probability")
st.write(prediction_proba)