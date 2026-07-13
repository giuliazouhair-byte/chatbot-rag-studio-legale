"""
app.py
Interfaccia chat in Streamlit per il chatbot RAG dello studio legale.
"""

import streamlit as st
from rag import create_index_from_scratch, generate_answer

st.set_page_config(page_title="Assistente Studio Legale", page_icon="⚖️")

st.title("⚖️ Assistente Studio Legale (RAG)")
st.caption(
    "Assistente didattico basato sui documenti dello studio. "
    "Non fornisce consulenza legale professionale."
)


@st.cache_resource(show_spinner="Indicizzazione dei documenti in corso...")
def get_index():
    return create_index_from_scratch()

try:
    index = get_index()
except Exception as e:
    st.error(f"Errore nella costruzione dell'indice: {e}")
    st.stop()

st.success(f"Indice pronto: {len(index)} passaggi indicizzati.")

if st.button("🗑️ Nuova conversazione"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Fonti: " + ", ".join(msg["sources"]))


user_question = st.chat_input("Scrivi la tua domanda sui documenti dello studio...")

if user_question:
   
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

# Genera la risposta con il RAG
    with st.chat_message("assistant"):
        with st.spinner("Sto cercando nei documenti..."):
            result = None
            try:
                result = generate_answer(user_question, index)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = f"Si è verificato un errore durante la generazione della risposta: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            st.caption("Fonti: " + ", ".join(sources))

        # Pannello opzionale: mostra i passaggi recuperati (trasparenza)
        if result and "chunks" in result:
            with st.expander("Mostra i passaggi recuperati (debug/trasparenza)"):
                for c in result["chunks"]:
                    st.markdown(f"**{c['source']}** (score: {c['score']:.3f})")
                    st.text(c["text"])
                    st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
