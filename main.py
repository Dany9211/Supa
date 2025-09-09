import streamlit as st
import pandas as pd
import io

def pulisci_e_formatta_df(df):
    """
    Funzione per pulire e formattare un DataFrame di pandas.
    - Rimuove le righe duplicate.
    - Estrae e riposiziona le colonne 'giorno', 'mese', 'anno' da 'date_GMT'.
    - Ordina il DataFrame per 'timestamp' se la colonna esiste.
    """
    
    # 1. Rimozione righe duplicate
    righe_iniziali = len(df)
    df.drop_duplicates(inplace=True)
    duplicati_rimossi = righe_iniziali - len(df)
    
    st.write(f"✅ Rimosse **{duplicati_rimossi}** righe doppione.")
    
    # 2. Estrazione e riposizionamento di Giorno, Mese, Anno
    if 'date_GMT' in df.columns:
        try:
            df['date_GMT'] = df['date_GMT'].astype(str).str.split(' - ').str[0]
            parts = df['date_GMT'].str.split(' ', expand=True)
            df['giorno'] = parts[1]
            df['mese'] = parts[0]
            df['anno'] = parts[2]
            
            # Riordina le colonne
            if 'attendance' in df.columns:
                idx_attendance = df.columns.get_loc('attendance')
                cols = df.columns.tolist()
                new_cols = cols[:idx_attendance] + ['giorno', 'mese', 'anno'] + cols[idx_attendance:]
                df = df.reindex(columns=new_cols)
            
            df.drop(columns=['date_GMT'], inplace=True)
            st.write("✅ Colonne 'giorno', 'mese' e 'anno' create e 'date_GMT' rimossa.")
        except Exception as e:
            st.warning(f"⚠️ ERRORE durante l'elaborazione della colonna 'date_GMT': {e}")
    else:
        st.info("ℹ️ AVVISO: Colonna 'date_GMT' non trovata.")

    # 3. Ordinamento per timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df.sort_values(by='timestamp', inplace=True)
        st.write("✅ Dataset ordinato cronologicamente.")
        
    return df

def to_csv(df):
    """Converte un DataFrame in un formato CSV in memoria per il download."""
    output = io.StringIO()
    df.to_csv(output, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    return output.getvalue().encode('utf-8-sig')

# --- Struttura dell'App Streamlit ---
st.set_page_config(page_title="CSV Cleaner", page_icon="🧹")
st.title("🧹 Pulitore e Formattatore di File CSV")
st.markdown("Carica un file CSV per pulirlo, rimuovere i duplicati, e formattare la colonna 'date_GMT'.")

# Componente per caricare il file
uploaded_file = st.file_uploader(
    "Carica un file CSV",
    type=["csv"],
    help="Supporta i delimitatori ';' e ',' e la codifica UTF-8 o Latin-1."
)

if uploaded_file is not None:
    # Mostra l'indicatore di caricamento
    with st.spinner('Caricamento e lettura del file...'):
        try:
            # Tenta di leggere il file con diversi delimitatori e codifiche
            try:
                # Prova il punto e virgola (formato italiano)
                df = pd.read_csv(uploaded_file, sep=';', low_memory=False, on_bad_lines='skip')
                if df.shape[1] == 1:
                    # Se non funziona, prova la virgola
                    uploaded_file.seek(0) # Riporta il "puntatore" all'inizio del file
                    df = pd.read_csv(uploaded_file, sep=',', low_memory=False, on_bad_lines='skip')
            except UnicodeDecodeError:
                # Prova la codifica Latin-1 se la prima fallisce
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1', low_memory=False, on_bad_lines='skip')
                if df.shape[1] == 1:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=',', encoding='latin-1', low_memory=False, on_bad_lines='skip')

            st.success("🎉 File letto con successo!")
            st.write("### Anteprima del file originale")
            st.dataframe(df.head())

            # Elabora il DataFrame
            df_pulito = pulisci_e_formatta_df(df.copy())
            
            st.write("---")
            st.write("### Anteprima del file pulito e formattato")
            st.dataframe(df_pulito.head())

            # Pulsante per il download del file elaborato
            csv_data = to_csv(df_pulito)
            
            st.download_button(
                label="📥 Scarica il file CSV pulito",
                data=csv_data,
                file_name=f'pulito_{uploaded_file.name}',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"❌ Si è verificato un errore durante l'elaborazione del file: {e}")
            st.help("Verifica che il file sia un CSV valido e che non sia corrotto.")
