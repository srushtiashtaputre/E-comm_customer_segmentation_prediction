import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("scalar.pkl", "rb") as scalar_file:
    scalar = pickle.load(scalar_file)

# Define cluster labels
cluster_labels = {
    0: "Weak Customer, May like to Churn",
    1: "Potential Loyal Customer, churn probability is less",
    2: "High Risk customer, High probability to churn",
    3: "Best Customer, These are good customers"
}

business_strategies = {
    0: "**Weak Customer**:\n- Offer special discounts\n- Improve customer engagement\n- Personalized email campaigns",
    1: "**Potential Loyal Customer**:\n- Provide loyalty rewards\n- Exclusive early access to products\n- Encourage referrals",
    2: "**High Risk to Churn**:\n- Conduct customer feedback surveys\n- Offer limited-time special deals\n- Improve customer support",
    3: "**Best Customer**:\n- VIP perks and premium benefits\n- Priority customer service\n- Personalized thank-you offers"
}

def predict_customer_segment(recency, frequency, monetary):
    input_data = np.array([[recency, frequency, monetary]])
    input_df = pd.DataFrame(input_data)
    input_df = scalar.transform(input_df)

    predicted_cluster = model.predict(input_df)[0]
    cluster_label = cluster_labels.get(predicted_cluster, "Unknown Cluster")
    business_strategy = business_strategies.get(predicted_cluster, "No strategy available")

    return predicted_cluster, cluster_label, business_strategy

def show_business_strategies():
  strategies = "\n\n".join([f"**Cluster {k}**: {business_strategies[k]}" for k in business_strategies])
  return strategies

# Streamlit UI
st.title("Customer Segmentation and Prediction ")
st.write("Enter Recency, Frequency, and Monetary Value to predict the customer segment.")

# User Inputs
recency = st.number_input("Recency (days since last purchase)")
frequency = st.number_input("Frequency (number of purchases)")
monetary = st.number_input("Monetary Value (total spent)")

# Prediction Button
if st.button("Predict Customer Segment"):
    predicted_cluster, cluster_label, business_strategy = predict_customer_segment(recency, frequency, monetary)
    st.success(f"Predicted Cluster: {predicted_cluster} ({cluster_label})")
    
    # Display the business strategy for the predicted cluster
    st.markdown("### Recommended Business Strategy:")
    st.markdown(business_strategy)

# Show Business Strategies Button
if st.button("Show All Business Strategies"):
  st.markdown(show_business_strategies())
