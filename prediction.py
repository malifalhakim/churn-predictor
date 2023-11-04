import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def predict_churn (data):
    clf = joblib.load('churn_model.sav')
    return clf.predict(data)

def show_shap_value(data):
    clf = joblib.load('churn_model.sav')
    explainer = shap.TreeExplainer(clf)
    data_df = pd.Series(data,index=['TenureMonths', 'Location', 'DeviceClass', 'GamesProduct',
       'MusicProduct', 'EducationProduct', 'CallCenter', 'VideoProduct',
       'UseMyApp', 'PaymentMethod', 'MonthlyPurchase',
       'EstimatedTotalPurchase'])
    shap_values = explainer.shap_values([data_df])

    fig = shap.force_plot(explainer.expected_value, shap_values, data_df,matplotlib=True)
    st.pyplot(fig)
    plt.clf()

def predict_cltv(data):
    reg = joblib.load('cltv_model.sav')
    return reg.predict([data])

def show_shap_value_reg(data):
    reg = joblib.load('cltv_model.sav')
    data_df =  pd.Series(data,index=['TenureMonths', 'Location', 'DeviceClass', 'GamesProduct',
       'MusicProduct', 'EducationProduct', 'CallCenter', 'VideoProduct',
       'UseMyApp', 'PaymentMethod', 'MonthlyPurchase'])
    data_array = data_df.values.reshape(1,-1)
    explainer = shap.TreeExplainer(reg)
    shap_values = explainer.shap_values(data_array)
    fig = shap.force_plot(explainer.expected_value, shap_values, data_df,matplotlib=True)
    st.pyplot(fig)
    plt.clf()