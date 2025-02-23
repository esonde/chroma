import streamlit as st
import os
import json
import random
import math
import base64
import numpy as np
import pandas as pd
import altair as alt

# Configurazione iniziale
st.set_page_config(page_title="Beauty Comparison App", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Parametri globali
USE_IMGS = 100  # Numero massimo di immagini attese
DATA_FILE = 'comparisons.json'
PHOTO_FOLDER = 'photos'

# Funzioni di utilità
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            content = f.read().strip()
            return json.loads(content) if content else {}
    return {}

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)

def get_random_pair():
    img1 = random.randint(0, USE_IMGS - 1)
    img2 = random.randint(0, USE_IMGS - 1)
    while img2 == img1:
        img2 = random.randint(0, USE_IMGS - 1)
    return img1, img2

def load_image_as_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"
    except Exception as e:
        st.error(f"Errore nel caricamento dell'immagine {image_path}: {e}")
        return None

# Modello Bradley-Terry
def compute_bt(comparisons):
    epsilon = 1e-3
    # Raccogli tutti gli ID unici dai confronti
    all_ids = set()
    for w, l in comparisons:
        all_ids.add(str(w))
        all_ids.add(str(l))
    
    # Inizializza n_matrix con tutti gli ID trovati
    n_matrix = {id_: {} for id_ in all_ids}
    wins = {}
    
    # Popola la matrice dei confronti
    for w, l in comparisons:
        w, l = str(w), str(l)
        wins[w] = wins.get(w, 0) + 1
        n_matrix[w][l] = n_matrix[w].get(l, 0) + 1
        n_matrix[l][w] = n_matrix[w][l]
    
    images = list(n_matrix.keys())
    theta = {img: 1.0 for img in images}
    for _ in range(1000):
        theta_new = {}
        max_diff = 0
        for i in images:
            w_i = wins.get(i, 0) + epsilon
            denom = sum((n_ij + epsilon) / (theta[i] + theta[j]) for j, n_ij in n_matrix[i].items())
            theta_new[i] = w_i / denom if denom else theta[i]
            max_diff = max(max_diff, abs(theta_new[i] - theta[i]))
        theta = theta_new
        if max_diff < 1e-6:
            break
    elo = {img: 400 * math.log10(max(min(t, 1e6), 1e-6)) for img, t in theta.items()}
    # Ritorna Elo per tutti gli ID, default a 0 per quelli non presenti
    return {str(i): elo.get(str(i), 0) for i in range(max(map(int, all_ids)) + 1)}

def compute_std(elo):
    return np.std(list(elo.values())) if elo else 0

def compute_correlation(elo1, elo2):
    common = set(elo1.keys()) & set(elo2.keys())
    if len(common) < 2:
        return None
    arr1, arr2 = [elo1[k] for k in common], [elo2[k] for k in common]
    if np.std(arr1) == 0 or np.std(arr2) == 0:
        return None
    return np.corrcoef(arr1, arr2)[0, 1]

# App principale
def main():
    data = load_data()

    # Login
    if "username" not in st.session_state:
        st.title("Beauty Comparison App")
        st.subheader("Login")
        if "temp_username" not in st.session_state:
            st.session_state.temp_username = f"User{random.randint(10000, 99999)}"
        username = st.text_input("Nome utente", value=st.session_state.temp_username)
        gender = st.selectbox("A chi sei attratto?", ["Uomo", "Donna", "Entrambi", "Nessuno"], index=2)
        if st.button("Accedi"):
            if not username:
                st.error("Inserisci un nome utente.")
            elif username in data:
                st.error("Nome utente già esistente.")
            else:
                st.session_state.username = username
                data[username] = {"comparisons": [], "gender": gender}
                save_data(data)
                st.rerun()
        return

    # Schermata principale
    current_user = st.session_state.username
    st.title("Beauty Comparison App")
    st.markdown(f"<h3>Benvenuto, <span style='color:#4CAF50'>{current_user}</span>!</h3>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Confronto immagini
    st.header("Confronto tra Immagini")
    st.write("Seleziona cliccando sul bottone sotto l'immagine che preferisci.")
    
    if "current_pair" not in st.session_state:
        st.session_state.current_pair = get_random_pair()
    
    img1_id, img2_id = st.session_state.current_pair
    img1_path = os.path.join(PHOTO_FOLDER, f"{img1_id:05d}.png")
    img2_path = os.path.join(PHOTO_FOLDER, f"{img2_id:05d}.png")
    img1_base64 = load_image_as_base64(img1_path)
    img2_base64 = load_image_as_base64(img2_path)
    
    if img1_base64 and img2_base64:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1_base64, use_container_width=True)
            if st.button("Seleziona", key=f"btn_img1_{img1_id}_{img2_id}"):
                data[current_user]["comparisons"].append([img1_id, img2_id])
                save_data(data)
                st.session_state.current_pair = get_random_pair()
                st.rerun()
        with col2:
            st.image(img2_base64, use_container_width=True)
            if st.button("Seleziona", key=f"btn_img2_{img1_id}_{img2_id}"):
                data[current_user]["comparisons"].append([img2_id, img1_id])
                save_data(data)
                st.session_state.current_pair = get_random_pair()
                st.rerun()

    # Statistiche
    st.header("Classifica Utenti")
    stats = []
    elo_cache = {}
    for user, info in data.items():
        comps = info.get("comparisons", [])
        elo = compute_bt(comps) if comps else {str(i): 0 for i in range(USE_IMGS)}
        elo_cache[user] = elo
        corr = 1.0 if user == current_user else compute_correlation(elo_cache.get(current_user, {}), elo)
        stats.append({
            "Utente": user,
            "Preferenze": len(comps),
            "Deviazione Std Elo": round(compute_std(elo), 2),
            "Accordo con te": round(corr, 2) if corr is not None else None,
            "Attrazione": info.get("gender", "Entrambi")
        })
    
    df = pd.DataFrame(stats)
    df.sort_values(by="Deviazione Std Elo", ascending=False, inplace=True)
    
    # Evidenzia la riga dell'utente corrente con il colore
    def highlight_current(row):
        if row["Utente"] == current_user:
            return ['background-color: #d4f1c5'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = df.style.apply(highlight_current, axis=1)
    st.dataframe(styled_df)
    
    st.subheader("Grafico Statistiche")
    if not df.empty:
        chart = alt.Chart(df).mark_bar().encode(
            x="Utente:N",
            y="Preferenze:Q",
            tooltip=["Utente", "Preferenze", "Deviazione Std Elo", "Accordo con te", "Attrazione"]
        ).properties(width=600)
        st.altair_chart(chart, use_container_width=True)
    
    if st.button("Aggiorna statistiche"):
        st.rerun()

if __name__ == "__main__":
    main()
