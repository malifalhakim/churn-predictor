import joblib
import pandas as pd
import shap
import streamlit as st
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import numpy as np

def predict_churn (data):
    clf = joblib.load('churn_model.sav')
    return clf.predict_proba(data)

def show_shap_value(data):
    clf = joblib.load('churn_model.sav')

    explainer = shap.TreeExplainer(clf)
    
    data_df = pd.Series(data,index=['TenureMonths', 'Location', 'DeviceClass', 'GamesProduct',
       'MusicProduct', 'EducationProduct', 'CallCenter', 'VideoProduct',
       'UseMyApp', 'PaymentMethod', 'MonthlyPurchase'])
    shap_values = explainer.shap_values([data_df])

    return st_shap(shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0],data_df,max_display=16),width=700,height=500)