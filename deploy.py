import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import matplotlib.pyplot as plt

@st.cache_resource
def get_pg_models():
    pga_model= joblib.load('models/mlp_PGA_scx.pkl')
    pgv_model= joblib.load('models/mlp_PGV_scx.pkl')
    return pga_model, pgv_model

@st.cache_resource
def get_sa_models(model):
    models = []
    periods = []
    filenames = []
    pattern = re.compile(fr'{model}_Sa(\d+_\d+)_scx.pkl')
    for filename in os.listdir('models/'):
        if filename.startswith(model) and filename.endswith('.pkl'):
            match = pattern.match(filename)
            if match:
                period = float(match.group(1).replace('_', '.'))
                periods.append(period)
                filenames.append(filename)
                trained_model = joblib.load(os.path.join('models', filename))
                models.append(trained_model)
    return models, periods, filenames

@st.cache_data
def get_data_download(sa_periods, predicted_sa):
    csv_df = pd.DataFrame({'Periods': sa_periods, 'Predicted_SA': predicted_sa})
    return csv_df.to_csv(index=False).encode('utf-8')

def main():
    apptitle = 'Tabriz GMM'
    st.set_page_config(page_title=apptitle, page_icon="s4h-logo.svg")
    st.title('Tabriz Ground Motion Model')
    col_img, col_txt = st.columns([2, 7])
    col_img.image("s4h-logo.svg", width=150)
    col_txt.markdown("""
                * **Mixed effect regression** using **Artificial Neural Networks**
                * Reference: *Simulation of Seismic Scenarios and Construction of an ANN-based Ground Motion Model: A Case Study on the North Tabriz Fault in Northwest Iran*
                """)
    st.write('#### Please Define Input Parameters')
    col_mw, col_rjb, col_fd = st.columns(3)
    mw = col_mw.number_input('**Magnitude** (Mw)', min_value=0.0, max_value=10.0, value=7.0, step=0.1, help="Moment magnitude of the earthquake")
    rjb = col_rjb.number_input('**Distance** (RJB) in km', min_value=0.0, max_value=1000.0, value=20.0, step=1.0, help="Joyner-Boore distance in kilometers")
    fd = col_fd.number_input('**Focal Depth** (Fd) in km', min_value=0.0, max_value=100.0, value=10.0, step=1.0, help="Focal depth of the earthquake in kilometers")
    input_data = pd.DataFrame({'Mw': [mw], 'Rjb': [rjb], 'Focal depth': [fd]})

    try:
        if mw <= 0 or rjb <= 0 or fd <= 0:
            st.error("Inputs must be positive values.")
        else:
            (pga_model, pga_scaler), (pgv_model, pgv_scaler) = get_pg_models()
            predicted_pga = np.exp(pga_model.predict(pga_scaler.transform(input_data)))[0]
            predicted_pgv = np.exp(pgv_model.predict(pgv_scaler.transform(input_data)))[0]
            sa_models_scalers, sa_periods, sa_filenames = get_sa_models('mlp')
            predicted_sa = []
            for (sa_model, sa_scaler) in sa_models_scalers:
                   predicted_sa.append(np.exp(sa_model.predict(sa_scaler.transform(input_data)))[0])
            sa_periods, predicted_sa = zip(*sorted(zip(sa_periods, predicted_sa)))

            st.write('#### Peak Ground Motion Estimates')
            col_pga, col_pgv = st.columns(2)
            col_pga.latex(f'PGA = {predicted_pga:.2f}\\ \\frac{{cm}}{{s^2}}')
            col_pgv.latex(f'PGV = {predicted_pgv:.2f}\\ \\frac{{cm}}{{s}}')

            st.write('#### Spectral Acceleraton Graph')
            fig, ax = plt.subplots(figsize=(9, 2), dpi=600)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.plot(sa_periods, predicted_sa, color='tab:blue')
            ax.set_xlabel('Periods (s)')
            ax.set_ylabel(r'SA ($\frac{cm}{s^2}$)')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            st.pyplot(fig)

            st.download_button(label='Download SA as .csv',
                               data=get_data_download(sa_periods, predicted_sa),
                               file_name='Sa_Tabriz.csv', mime='text/csv')
    except ValueError as ve:
        st.error(str(ve))
    st.markdown("Developed by [S.M. Sajad Hussaini](https://linkedin.com/in/sajadhussaini)")

if __name__ == '__main__':
    main()
