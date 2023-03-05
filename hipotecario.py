import streamlit as st
import numpy as np
import pandas as pd
import pickle
import math
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Colores en el texto:
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


#Establece origen de datos
os.chdir('C:\\Users\\dhcamarg\\.apphipo')
# workdirectory=os.getcwd()

#rutas hacia archivos binarios
# rutapoisson=os.path.join(workdirectory,'modelos\\modelo_poisson.sav')
# rutaarimapoisson=os.path.join(workdirectory,'modelos\\arima_poisson.sav')
# rutadatapoisson=os.path.join(workdirectory,'modelos\\data_poisson.sav')
# rutaprop=os.path.join(workdirectory,'modelos\\modelo_prop.sav')
# rutadataprop=os.path.join(workdirectory,'modelos\\data_prop.sav')


#cargar archivos
poisson_model = pickle.load(open("modelo_poisson.sav", 'rb'))
arimapoisson_model = pickle.load(open("arima_poisson.sav", 'rb'))
poisson_data = pickle.load(open("data_poisson.sav", 'rb'))
prop_model=pickle.load(open("modelo_prop.sav", 'rb'))
prop_data=pickle.load(open("data_prop.sav", 'rb'))
arimaprop_model=pickle.load(open("arima_prop.sav", 'rb'))

# Imagen
icon=Image.open("LogoBancolombia.png")
st.set_page_config(
    page_title="BANSIMEF",
    page_icon=icon,
    layout="wide"
)

#funcion poisson
def predice_poisson(tasa_pesos):
    index_of_fc=pd.date_range(poisson_data.index[-1] + pd.DateOffset(weeks=1), periods = 13, freq='W')
    semanas=index_of_fc.isocalendar().week
    semana_baja=semanas.apply(lambda x: 1 if (x>=49) | (x<=2) else 0)
    semana_alta=semanas.apply(lambda x: 1 if (x>=3) & (x<=11) else 0)
    agregado_real=poisson_data.desembolso.resample("M").sum()
    inflacion=poisson_data.inflacion[-1:][0]
    if inflacion>6:
        intervencion=1
    else:
        intervencion=0
    if max(pd.concat([poisson_data.tasapesos[-13:],pd.Series(tasa_pesos)]))==tasa_pesos:
        X_test={'const':list(np.repeat(1,13)),
                'semana_baja':semana_baja,
                'semana_alta':list(semana_alta),
                'covid':list(np.repeat(0,13)),
                'reactivacion':list(np.repeat(0,13)),
                'tasapesos':list(np.repeat(tasa_pesos,13)),
                'tendencia':list(np.repeat(0,13))               
                }
        X_test=pd.DataFrame.from_dict(X_test)
        X_test.index=index_of_fc
        pred_media=pd.Series(poisson_model.predict(X_test),index=index_of_fc)
        # make series for plotting purpose
        fitted,confint = arimapoisson_model.predict(n_periods=13,return_conf_int=True)
        fitted_series = pd.Series(fitted, index=index_of_fc)+pred_media
        agregado_pred=fitted_series.resample("M").sum()
        nivel_agregado_pred=agregado_pred[:-1]/(1+0.20)
        int_sup=nivel_agregado_pred*(1+0.25)
        int_inf=nivel_agregado_pred/(1+0.25)
        #grafico
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=agregado_real.index, y=agregado_real[:],
                        mode='lines',
                                name='Conteo'))
        fig.add_trace(go.Scatter(x=nivel_agregado_pred.index, y=nivel_agregado_pred[:],
                        mode='lines',
                        name='prediccion',
                        line_color="#0a0a0a"))
        fig.add_trace(go.Scatter(x=int_sup.index, y=int_sup[:],
                        mode='lines',
                        name='Lim superior',
                        line_color="#e3fc3f"))
        fig.add_trace(go.Scatter(x=int_inf.index, y=int_inf[:],
                        mode='lines',
                        name='Lim inferior',
                        line_color="#e3fc3f"))
        fig.update_layout(title_text="Número de solicitudes banco -pronostico 2 meses",
                    title_font_size=30)
        st.plotly_chart(fig)
        return nivel_agregado_pred
    else:
        tendencia=tasa_pesos-max(poisson_data.tasapesos[-13:])
        X_test={'const':list(np.repeat(1,13)),
                'semana_baja':semana_baja,
                'semana_alta':list(semana_alta),
                'covid':list(np.repeat(0,13)),
                'reactivacion':list(np.repeat(0,13)),
                'tasapesos':list(np.repeat(tasa_pesos,13)),
                'tendencia':list(np.repeat(0,13))               
                }
        X_test=pd.DataFrame.from_dict(X_test)
        X_test.index=index_of_fc
        pred_media=pd.Series(poisson_model.predict(X_test),index=index_of_fc)
        # make series for plotting purpose
        fitted,confint = arimapoisson_model.predict(n_periods=13,return_conf_int=True)
        fitted_series = pd.Series(fitted, index=index_of_fc)+pred_media
        agregado_pred=fitted_series.resample("M").sum()
        nivel_agregado_pred=agregado_pred[:-1]/(1+0.20)
        int_sup=nivel_agregado_pred*(1+0.25)
        int_inf=nivel_agregado_pred/(1+0.25)
        #grafico
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=agregado_real.index, y=agregado_real[:],
                        mode='lines',
                                name='Conteo'))
        fig.add_trace(go.Scatter(x=nivel_agregado_pred.index, y=nivel_agregado_pred[:],
                        mode='lines',
                        name='prediccion',
                        line_color="#0a0a0a"))
        fig.add_trace(go.Scatter(x=int_sup.index, y=int_sup[:],
                        mode='lines',
                        name='Lim superior',
                        line_color="#e3fc3f"))
        fig.add_trace(go.Scatter(x=int_inf.index, y=int_inf[:],
                        mode='lines',
                        name='Lim inferior',
                        line_color="#e3fc3f"))
        fig.update_layout(title_text="Número de solicitudes banco -pronostico 2 meses",
                    title_font_size=30)
        st.plotly_chart(fig)
        return nivel_agregado_pred
   

def predice_prop(tasacomp,tasabanrep,reactivacion=0):
    index_of_fc=pd.date_range(poisson_data.index[-1] + pd.DateOffset(weeks=1), periods = 13, freq='W')
    X_test={'const':list(np.repeat(1,13)),
            'reactivacion':list(np.repeat(reactivacion,13)),
            'competencia':list(np.repeat(tasacomp,13)),
            'tasabanrep':list(np.repeat(tasabanrep,13))
            }
    X_test=pd.DataFrame.from_dict(X_test)
    pred_media=prop_model.predict(X_test)
    pred_media.index=index_of_fc
    fitted,confint = arimaprop_model.predict(n_periods=13,return_conf_int=True)
    fitted_series = pd.Series(fitted, index=index_of_fc)+pred_media
    agregado_pred=fitted_series.resample("M").mean()
    agregado_pred=agregado_pred[:-1]
    return agregado_pred

def plot_prediccion(conteo,proporcion):
    col1,col2=st.columns([4,2])
    with col1:
        estimado=conteo*proporcion
        int_sup=estimado*(1+0.26)
        int_inf=estimado/(1+0.26)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=estimado.index, y=estimado[:],
                                mode='lines',
                                name='Créditos',
                                line_color="#0a0a0a"))
        fig.add_trace(go.Scatter(x=int_sup.index, y=int_sup[:],
                                mode='lines',
                                name='lim superior',
                                line_color="#e3fc3f"))
        fig.add_trace(go.Scatter(x=int_inf.index, y=int_inf[:],
                                mode='lines',
                                name='lim inferior',
                                line_color="#e3fc3f"))
        fig.update_layout(title_text="Solicitudes desembolsadas <br><sup>-pronostico 2 meses</sup>",
                        title_font_size=30,
                        autosize=False,
                        width=450)
        st.plotly_chart(fig)
    with col2:
        st.subheader("Predicciones")
        st.write(f"""
        Se espera para el siguiente mes un total de {math.floor(conteo[0])} nuevas solicitudes, de las cuales
        se espera que al final del proceso se hayan generado {math.floor(estimado[0])} nuevos créditos
        hipotecarios. Adicionalmente se espera que máximo se generen {math.floor(int_sup[0])} nuevos 
        créditos y mínimo se generen {math.floor(int_inf[0])} nuevos créditos.

        Se espera para el segundo mes un total de {math.floor(conteo[1])} nuevas solicitudes, de las cuales 
        se espera que al final del proceso se hayan generado {math.floor(estimado[1])} nuevos créditos
        hipotecarios. Adicionalmente se espera que máximo se generen {math.floor(int_sup[1])} nuevos 
        créditosy mínimo se generen {math.floor(int_inf[1])} nuevos créditos.
        """)


def main():
   st.title("Simulador de efectos-segmento hipotecario")
   st.sidebar.header("Parámetros del usario")
   def user_inputs():
        #inputs usuarios
        banco=st.sidebar.number_input("Tasa bancolombia pesos",min_value=2.0,max_value=30.0,
                                value=float(poisson_data.tasapesos[-1:][0]),
                               step=0.01)
        otros=st.sidebar.number_input("Tasa competencia",min_value=2.0,max_value=30.0,
                                value=float(poisson_data.competencia[-1:][0]),
                               step=0.01)
        banrep=st.sidebar.number_input("Tasa BRC",min_value=2.0,max_value=30.0,
                                value=float(poisson_data.tasabanrep[-1:][0]),
                               step=0.01)
        
        return banco,otros,banrep
   
   tasab,tasac,BRC=user_inputs()
   with st.sidebar:
        st.write("El MAPE de predicción del numero de créditos desembolsados para dos meses es de 14.37% y el MAE es de 233 créditos")
   #subtitulo
   st.subheader("Parámetros elegidos por el usuario")
   st.write(pd.DataFrame({"Tasa banco":[tasab],
                          "Tasa competencia":[tasac],
                          "Tasa BanRep":[BRC],
                          }
                          ,index=["Parámetros"]))
   #arancar las estimaciones
   #if st.button("RUN"):
   prediccion_poisson=predice_poisson(tasab)
   prediccion_prop=predice_prop(tasac,BRC)
   plot_prediccion(prediccion_poisson,prediccion_prop)


if __name__=='__main__':
    main()