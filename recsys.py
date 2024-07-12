import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Определение устройства (GPU или CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"


st.set_page_config(
    page_title="Умный поиск сериалов",
    page_icon=":tv:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Умный поиск сериалов")


def load_data():
    df = pd.read_csv("/home/artemiy/recsys_project/data/dataset_new.csv")
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    model.to(device)
    return df, tokenizer, model


# Функция для создания эмбеддингов
def embed_bert_cls(text, model, tokenizer):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,  # Примерная максимальная длина описания
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        model_output = model(input_ids, attention_mask=attention_mask)

    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


def build_faiss_index(_df, _model, _tokenizer):
    _df["description"] = _df["description"].fillna("").astype(str)
    description_embeddings = np.vstack(
        _df["description"].apply(lambda x: embed_bert_cls(x, _model, _tokenizer)).values
    )
    index = faiss.IndexFlatL2(description_embeddings.shape[1])
    index.add(description_embeddings.astype(np.float32))
    return index, description_embeddings


# Функция поиска шоу
def find_shows(
    query, df, model, tokenizer, index, description_embeddings, filters, top_k=5
):
    query_embedding = embed_bert_cls(query, model, tokenizer).reshape(1, -1)
    _, top_k_indices = index.search(query_embedding.astype(np.float32), len(df))

    result = df.iloc[top_k_indices[0]].copy()
    similarities = cosine_similarity(
        query_embedding, description_embeddings[top_k_indices[0]]
    ).flatten()
    result["similarity"] = similarities

    # Фильтрация результатов
    result = result[result["rating"] >= filters["rating"]]

    if filters["year_from"]:
        result = result[result["year"] >= filters["year_from"]]
    if filters["year_to"]:
        result = result[result["year"] <= filters["year_to"]]
    if filters["genres"]:
        genres_selected = filters["genres"]
        result = result[
            result["genres"].apply(
                lambda x: any(genre in x for genre in genres_selected)
            )
        ]
    if filters["country"]:
        countries_selected = filters["country"]
        result = result[
            result["country"].apply(
                lambda x: any(country in x for country in countries_selected)
            )
        ]
    if filters["cast"]:
        result = result[result["cast"].str.contains(filters["cast"], case=False)]
    if filters["age"]:
        result = result[result["age"] >= filters["age"]]

    return result.head(top_k).sort_values("similarity", ascending=False)


# Загружаем данные и модель
df, tokenizer, model = load_data()
index, description_embeddings = build_faiss_index(df, model, tokenizer)

# Создание списка уникальных стран
all_countries = set()
for countries_list in df["country"].dropna().str.split(", "):
    all_countries.update([country.strip() for country in countries_list])
all_countries = sorted(all_countries)
# Создание списка уникальных жанров
all_genres = set()
for genres_list in df["genres"].dropna().str.split(", "):
    all_genres.update([genre.strip() for genre in genres_list])
all_genres = sorted(all_genres)
query = st.text_area("Введите описание сериала", height=150)
# Фильтры
st.sidebar.title("Фильтры")
with st.sidebar:
    st.markdown(
        "<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<style>div.Widget.row-widget.stRadio > div > label{background-color: #f0f0f0; padding: 5px; margin-right: 5px;}</style>",
        unsafe_allow_html=True,
    )

filters = {
    "rating": st.sidebar.slider(
        "Рейтинг",
        float(df["rating"].min()),
        float(df["rating"].max()),
        float(df["rating"].min()),
    ),
    "year_from": st.sidebar.slider(
        "Год выпуска от",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=int(df["year"].min()),
    ),
    "year_to": st.sidebar.slider(
        "Год выпуска до",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=int(df["year"].max()),
    ),
    "age": st.sidebar.slider(
        "Возрастное ограничение", min_value=int(df["age"].min()), max_value=21, value=21
    ),
    "genres": st.sidebar.multiselect("Жанры", options=all_genres),
    "country": st.sidebar.multiselect("Страны", options=all_countries),
    "cast": st.sidebar.text_input("Актеры"),
}

top_k = st.slider("Количество результатов", 1, 20, 5)

if st.button("Найти"):
    if query:
        results = find_shows(
            query,
            df,
            model,
            tokenizer,
            index,
            description_embeddings,
            filters,
            top_k=top_k,
        )
        st.write("Найденные сериалы:")
        for _, row in results.iterrows():
            st.image(row["poster"], width=150)
            st.markdown(f"### {row['title']}")
            st.markdown(
                f"**Рейтинг: {row['rating']} | Год: {row['year']} | Возраст: {row['age']}"
            )
            st.markdown(f"Жанры: {row['genres']}")
            st.markdown(f"Актеры: {row['cast']}")
            st.markdown(f"Страна: {row['country']}")
            st.markdown(f"Описание: {row['description']}")
            st.markdown(f"Косинусное сходство: {row['similarity']:.4f}")
            st.markdown("---")
    else:
        st.write("Введите описание для поиска.")
