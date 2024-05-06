import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

url_c = "https://raw.githubusercontent.com/cakirogl/tiles/main/tiles.csv"
url_t = "https://raw.githubusercontent.com/cakirogl/tiles/main/tiles_tensile.csv"
model_selector = st.selectbox('**Predictive model**', ["XGBoost", "LightGBM", "CatBoost", "Extra Trees"])
df_c = pd.read_csv(url_c);df_t=pd.read_csv(url_t);
x_c, y_c = df_c.iloc[:, :-1], df_c.iloc[:, -1]
x_t, y_t = df_t.iloc[:, :-1], df_t.iloc[:, -1]
scaler_c = MinMaxScaler();scaler_t = MinMaxScaler();
#x_c=scaler_c.fit_transform(x_c);x_t=scaler_t.fit_transform(x_t);
input_container = st.container()
ic1,ic2,ic3 = input_container.columns(3)
with ic1:
    age = st.number_input("**Hydration age [days]:**",min_value=14.0,max_value=60.0,step=1.0,value=28.0)
    nca = st.number_input("**Natural coarse aggregate [kg/m$^3$]]:**",min_value=32.0,max_value=65.0,step = 1.0,value=32.0)
    w_c = st.number_input("**Water to cement ratio**", min_value=0.4, max_value=0.45, step=0.01, value=0.4)
    cfa = st.number_input("**Ceramic fine aggregate [kg\/m$^3$]:**", min_value=0.0, max_value=9.13, step=1.0, value=4.5)
    cca_fine_mod = st.number_input("**Ceramic coarse aggregate fineness modulus:**", min_value=0.0, max_value=5.5, step=0.5, value=5.5)
with ic2:
    cfa_mean_part_size = st.number_input("**Ceramic fine aggregate mean part. size [$\mu$ m]:**", min_value=0.2, max_value=3.0, step=0.1, value=0.3)
    nfa=st.number_input("**Natural fine aggregate [kg/m$^3$]:**", min_value=30.0, max_value=50.0, step=1.0, value=40.0)
    cfa_rep_perc = st.number_input("**Ceramic fine aggregate replacement percentage [%]**", min_value=0.0, max_value=20.0, step=1.0, value=10.0)
    cca_rep_perc = st.number_input("**Ceramic coarse aggregate replacement percentage [%]**", min_value=0.0, max_value=50.0, value=5.0, step=10.0)
with ic3:
    cca = st.number_input("**Ceramic coarse aggregate [kg/m$^3$]:**", min_value=0.0, max_value=32.13, step=1.0, value=12.85)
    cca_spec_gr = st.number_input("**Ceramic coarse aggregate specific gravity:**", min_value = 0.0, max_value = 2.0, step = 0.1, value = 1.9)
    cca_abs_cap = st.number_input("**Ceramic coarse aggregate Absorption Capacity [%]:**", min_value=0.0, max_value = 14.23, step = 0.2, value = 14.0)
    cca_dens = st.number_input("**Ceramic coarse aggregate density [kg/m$^3$]:**", min_value=0.0, max_value = 1114.15, step = 100.0, value = 1114.0)

new_sample=np.array([[w_c, nca, cca_fine_mod, cca_dens, cca_abs_cap, cca_spec_gr, cca, cca_rep_perc, cfa_mean_part_size, cfa, cfa_rep_perc, nfa, age]],dtype=object)

if model_selector=="LightGBM":
    model_c=LGBMRegressor(random_state=0, verbose=-1)
    model_c.fit(x_c,y_c)
    model_t=LGBMRegressor(random_state=0, verbose=-1)
    model_t.fit(x_t,y_t)
elif model_selector=="XGBoost":
    model_c=XGBRegressor(random_state=0)
    model_c.fit(x_c, y_c)
    model_t=XGBRegressor(random_state=0)
    model_t.fit(x_t, y_t)
if model_selector=="CatBoost":
    model_c=CatBoostRegressor(random_state=0, verbose=-1)
    model_c.fit(x_c,y_c)
    model_t=CatBoostRegressor(random_state=0, verbose=-1)
    model_t.fit(x_t,y_t)
elif model_selector=="Extra Trees":
    model_c=ExtraTreesRegressor(random_state=0)
    model_c.fit(x_c, y_c)
    model_t=ExtraTreesRegressor(random_state=0)
    model_t.fit(x_t, y_t)

with ic2:
    st.write(f":blue[**Compressive strength = **{model_c.predict(new_sample)[0]:.3f}** MPa**]\n")
    
with ic3:
    st.write(f"**Tensile strength = **{model_t.predict(new_sample)[0]:.3f}** MPa**\n")