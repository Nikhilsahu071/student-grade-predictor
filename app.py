import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page Title
st.title("🎓 Student Grade Predictor")

st.write("""
This application predicts a student's final grade
based on attendance and assignment marks using
Machine Learning (Linear Regression).
""")

# Upload Dataset
uploaded_file = st.file_uploader(
    "Upload Student Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    # Show Dataset
    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

    # Train Model
    X = df[["Attendance", "Assignment"]]
    y = df["FinalGrade"]

    model = LinearRegression()
    model.fit(X, y)

    st.success("✅ Model Trained Successfully!")

    # User Input
    st.subheader("📝 Enter Student Details")

    attendance = st.slider(
        "Attendance (%)",
        0,
        100,
        80
    )

    assignment = st.slider(
        "Assignment Marks",
        0,
        100,
        75
    )

    # Predict Button
    if st.button("Predict Grade"):

        prediction = model.predict(
            [[attendance, assignment]]
        )

        predicted_grade = prediction[0]

        # Display Prediction
        st.subheader("🎯 Prediction Result")

        st.success(
            f"Predicted Grade: {predicted_grade:.2f}"
        )

        # Grade Classification
        if predicted_grade >= 90:
            grade = "A+"
        elif predicted_grade >= 80:
            grade = "A"
        elif predicted_grade >= 70:
            grade = "B"
        elif predicted_grade >= 60:
            grade = "C"
        else:
            grade = "D"

        st.info(f"Grade Classification: {grade}")

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Attendance",
                f"{attendance}%"
            )

        with col2:
            st.metric(
                "Assignment",
                assignment
            )

        with col3:
            st.metric(
                "Predicted Grade",
                f"{predicted_grade:.2f}"
            )

        # Chart
        st.subheader("📈 Performance Analysis")

        fig, ax = plt.subplots()

        ax.bar(
            ["Attendance", "Assignment", "Predicted Grade"],
            [attendance, assignment, predicted_grade]
        )

        ax.set_ylabel("Score")
        ax.set_title("Student Performance")

        st.pyplot(fig)

else:
    st.warning("Please upload a CSV dataset to continue.")