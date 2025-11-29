import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

# load the diabetes dataset
diabetes_df = pd.read_csv(
    r"C:\Users\jayan\Downloads\SWAROO\OneDrive\Desktop\projects\diabetes (1).csv"
)

# group the data by outcome to get a sense of the distribution
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

# split the data into input and target variables
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# scale the input variables using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create an SVM model
model = svm.SVC(
    kernel='rbf',          # a bit more flexible than linear
    probability=True,      # so we can see risk %
    class_weight='balanced',  # handle class imbalance
    random_state=1
)

# train the model on the training set
model.fit(X_train, y_train)

# make predictions on the training and testing sets
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

# calculate the accuracy of the model on the training and testing sets
train_acc = accuracy_score(y_train, train_y_pred)
test_acc = accuracy_score(y_test, test_y_pred)

# create the Streamlit app
def app():

    # show image
    img = Image.open(
        r"C:\Users\jayan\Downloads\SWAROO\OneDrive\Desktop\projects\img (1).jpeg"
    )
    img = img.resize((200, 200))
    st.image(img, caption="Diabetes Image", width=200)

    st.title('Diabetes Prediction')

    # create the input form for the user to input new data
    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # make a prediction based on the user input
    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    input_data_nparray = np.asarray(input_data, dtype=float).reshape(1, -1)

    # scale user input with the same scaler used for training
    scaled_input_data = scaler.transform(input_data_nparray)

    # predict
    prediction = model.predict(scaled_input_data)[0]   # 0 or 1
    proba = model.predict_proba(scaled_input_data)[0][1]  # probability of diabetes (class 1)

    # debug info (helps you see what's happening)
    st.subheader("Model Output")
    st.write(f"Raw prediction (0=no diabetes, 1=diabetes): **{prediction}**")
    st.write(f"Estimated diabetes risk: **{proba * 100:.1f}%**")

    # display the prediction to the user
    st.write('Based on the input features, the model predicts:')
    if prediction == 1:
        st.warning('This person has diabetes.')
    else:
        st.success('This person does not have diabetes.')

    # display some summary statistics about the dataset
    st.header('Dataset Summary')
    st.write(diabetes_df.describe())

    st.header('Distribution by Outcome')
    st.write(diabetes_mean_df)

    # display the model accuracy
    st.header('Model Accuracy')
    st.write(f'Train set accuracy: {train_acc:.2f}')
    st.write(f'Test set accuracy: {test_acc:.2f}')

if __name__ == '__main__':
    app()
