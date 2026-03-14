import streamlit as st
import pandas as pd
import xgboost as xgb
import datetime
import pytz
from entsoe import EntsoePandasClient

# RENSEIGNEZ VOTRE CLÉ API ENTSO-E ICI
API_KEY = "1efc1e14-c733-4915-bd4f-cf2e11e6750f"

st.set_page_config(page_title="Prediction Prix Electricite", layout="wide")
st.title("Tableau de Bord : Alertes Prix Negatifs")

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    try:
        model.load_model("modele_xgboost.json")
        return model
    except Exception:
        return None

model = load_model()

st.sidebar.header("Configuration")
pays_code = st.sidebar.selectbox("Zone de Bidding (Pays)", ["FR", "DE", "BE", "NL", "ES"], index=0)
seuil_alerte = st.sidebar.number_input("Seuil de declenchement d'alerte", min_value=0.0, max_value=1.0, value=0.73, step=0.01)

if st.sidebar.button("Recuperer les donnees et Predire"):
    if API_KEY == "VOTRE_CLE_API_ICI":
        st.error("Veuillez renseigner votre cle API dans le code à la ligne 9.")
    elif model is None:
        st.error("Modele introuvable. Verifiez la presence du fichier 'modele_xgboost.json'.")
    else:
        with st.spinner("Recuperation des previsions..."):
            try:
                client = EntsoePandasClient(api_key=API_KEY)
                
                timezone = pytz.timezone('Europe/Paris')
                demain = pd.Timestamp.now(tz=timezone).normalize()
                fin_demain = demain + pd.Timedelta(days=1)
                
                load_forecast = client.query_load_forecast(pays_code, start=demain, end=fin_demain)
                wind_solar_forecast = client.query_wind_and_solar_forecast(pays_code, start=demain, end=fin_demain)
                
                df_api = pd.DataFrame(index=load_forecast.index)
                df_api['Load'] = load_forecast
                
                wind_cols = [col for col in wind_solar_forecast.columns if 'Wind' in col]
                df_api['Forecast_Wind_MW'] = wind_solar_forecast[wind_cols].sum(axis=1) if wind_cols else 0
                df_api['Forecast_Solar_MW'] = wind_solar_forecast['Solar'] if 'Solar' in wind_solar_forecast.columns else 0
                
                df_api['Forecast_Residual_Load_MW'] = df_api['Load'] - (df_api['Forecast_Wind_MW'] + df_api['Forecast_Solar_MW'])
                df_api['Hour'] = df_api.index.hour
                df_api['DayOfWeek'] = df_api.index.dayofweek
                df_api['Month'] = df_api.index.month
                
                features = ['Forecast_Residual_Load_MW', 'Forecast_Solar_MW', 'Forecast_Wind_MW', 'Hour', 'DayOfWeek', 'Month']
                df_demain = df_api[features].copy()
                
                df_demain = df_demain.groupby('Hour').mean().reset_index()
                df_demain['DayOfWeek'] = demain.dayofweek
                df_demain['Month'] = demain.month

                df_pour_prediction = df_demain[features]
                probabilites = model.predict_proba(df_pour_prediction)[:, 1]
                df_demain['Probabilite Prix Negatif'] = probabilites
                
                st.subheader(f"Previsions horaires pour le {demain.strftime('%d/%m/%Y')} ({pays_code})")
                
                alertes = df_demain[df_demain['Probabilite Prix Negatif'] >= seuil_alerte]
                if not alertes.empty:
                    st.error(f"Risque de prix negatifs detecte pour {len(alertes)} heure(s) demain.")
                    heures_alerte = ", ".join([f"{int(h)}h" for h in alertes['Hour']])
                    st.write(f"Heures critiques : {heures_alerte}")
                else:
                    st.success("Aucun prix negatif prevu pour demain selon le seuil actuel.")

                chart_data = df_demain[['Hour', 'Probabilite Prix Negatif']].set_index('Hour')
                st.area_chart(chart_data)
                
                with st.expander("Voir les donnees brutes ENTSO-E et predictions"):
                    df_affichage = df_demain.copy()
                    df_affichage['Probabilite Prix Negatif'] = (df_affichage['Probabilite Prix Negatif'] * 100).round(1).astype(str) + ' %'
                    df_affichage[['Forecast_Residual_Load_MW', 'Forecast_Solar_MW', 'Forecast_Wind_MW']] = df_affichage[['Forecast_Residual_Load_MW', 'Forecast_Solar_MW', 'Forecast_Wind_MW']].round(0)
                    st.dataframe(df_affichage, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur lors de la recuperation des donnees ENTSO-E : {e}")

                # --- NOUVEAU BLOC : COMPARAISON AVEC LES VRAIS PRIX ---
                st.subheader(f"Comparaison : Prédictions vs Marché (Day-Ahead)")
                
                try:
                    # On tente de récupérer les vrais prix (publiés à 13h00)
                    vrais_prix = client.query_day_ahead_prices(pays_code, start=demain, end=fin_demain)
                    
                    # On s'assure que les données sont alignées
                    df_demain['Vrai Prix (EUR/MWh)'] = vrais_prix.values
                    # On crée une colonne binaire pour savoir si le vrai prix était négatif ou non
                    df_demain['Realite Prix Negatif'] = (df_demain['Vrai Prix (EUR/MWh)'] <= 0).astype(int)
                    
                    st.success("Les prix Day-Ahead officiels sont disponibles et ont ete recuperes !")
                    
                    # Graphique comparatif : Probabilité prédite vs Réalité (0 ou 1)
                    chart_comparaison = df_demain[['Hour', 'Probabilite Prix Negatif', 'Realite Prix Negatif']].set_index('Hour')
                    st.bar_chart(chart_comparaison)
                    
                    # Affichage des vrais prix dans le tableau détaillé
                    with st.expander("Voir les donnees brutes ENTSO-E et predictions"):
                        df_affichage = df_demain.copy()
                        df_affichage['Probabilite Prix Negatif'] = (df_affichage['Probabilite Prix Negatif'] * 100).round(1).astype(str) + ' %'
                        df_affichage[['Forecast_Residual_Load_MW', 'Forecast_Solar_MW', 'Forecast_Wind_MW']] = df_affichage[['Forecast_Residual_Load_MW', 'Forecast_Solar_MW', 'Forecast_Wind_MW']].round(0)
                        st.dataframe(df_affichage, use_container_width=True)

                except Exception as e:
                    # Si on est avant 13h00, l'API renverra une erreur car les prix n'existent pas encore
                    st.info("Heure actuelle : Les prix Day-Ahead pour demain ne sont pas encore publies par la bourse (publication vers 13h00). Affichage de vos predictions seules en attendant.")
                    
                    # On affiche le graphique standard avec juste vos probabilités
                    chart_data = df_demain[['Hour', 'Probabilite Prix Negatif']].set_index('Hour')
                    st.area_chart(chart_data)
                    
                    with st.expander("Voir les donnees brutes ENTSO-E et predictions"):
                        df_affichage = df_demain.copy()
                        df_affichage['Probabilite Prix Negatif'] = (df_affichage['Probabilite Prix Negatif'] * 100).round(1).astype(str) + ' %'
                        df_affichage[['Forecast_Residual_Load_MW', 'Forecast_Solar_MW', 'Forecast_Wind_MW']] = df_affichage[['Forecast_Residual_Load_MW', 'Forecast_Solar_MW', 'Forecast_Wind_MW']].round(0)
                        st.dataframe(df_affichage, use_container_width=True)