import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm

# Определение устройства (GPU или CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Умный поиск сериалов",
    page_icon=":tv:",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Загрузка данных и модели BERT


def load_data():
    df = pd.read_csv("/home/artemiy/recsys_project/data/dataset_new.csv")
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    model.to(device)
    return df, tokenizer, model


# Функция для создания эмбеддингов текста с помощью BERT
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


# Функция для построения индекса FAISS на основе эмбеддингов описаний
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


# Функция для парсинга данных о сериалах
def scrape_series_data(
    start_page=1,
    end_page=316,
    base_url="https://kino.mail.ru/series/all/?order=rate_count&year=1916&year=2024",
):
    items_urls = []

    for i in tqdm(range(start_page, end_page + 1)):
        end = "/" if i == 1 else f"&page={i}"
        link = f"{base_url}{end}"
        content = requests.get(link).content
        soup = BeautifulSoup(content, "lxml")
        series = soup.find_all(
            class_="link link_inline link-holder link-holder_itemevent link-holder_itemevent_small"
        )

        for item in series:
            item_page_url = item.get("href")
            full_url = "https://kino.mail.ru" + item_page_url
            if full_url not in items_urls:
                items_urls.append(full_url)

    titles, sources, descriptions, genres, years, posters, casts, ratings = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for url in tqdm(items_urls):
        try:
            text = requests.get(url).text
            soup = BeautifulSoup(text, "lxml")
            title = soup.find(
                "h1", class_="text text_bold_giant color_white"
            ).text.split(" (сери")[0]
            picture_url = soup.find("meta", itemprop="image")["content"]
            genre = [i.text for i in soup.find_all("span", class_="badge__text")]
            cast = [
                act.text
                for act in soup.find_all(
                    "span", class_="p-truncate__inner js-toggle__truncate-inner"
                )[1:4]
            ]
            description = soup.find(
                "div",
                class_="text text_inline text_light_medium text_fixed valign_baseline p-movie-info__description-text",
            ).text.replace("\xa0", " ")
            year = soup.find("span", class_="nowrap").text.split()[2]
            rating = float(
                soup.find("span", class_="text text_bold_huge text_fixed").text
            )

            # Appending to lists
            titles.append(title)
            sources.append(url)
            descriptions.append(description)
            genres.append(genre)
            years.append(year)
            posters.append(picture_url)
            casts.append(cast)
            ratings.append(rating)

        except Exception as e:
            print(f"Error scraping {url}: {e}")

    data = pd.DataFrame(
        {
            "url": sources,
            "poster": posters,
            "title": titles,
            "genres": genres,
            "description": descriptions,
            "year": years,
            "rating": ratings,
            "cast": casts,
        }
    )

    return data


# Streamlit UI
def main():
    st.title("Умный поиск сериалов")

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Выберите страницу", ["Поиск сериалов", "Парсинг данных"]
    )

    if page == "Поиск сериалов":
        st.header("Поиск сериалов")

        # Загрузка данных и модели
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
                "Возрастное ограничение",
                min_value=int(df["age"].min()),
                max_value=21,
                value=21,
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

    elif page == "Парсинг данных":
        st.header("Сбор данных о сериалах")
        start_page = st.number_input(
            "Начальная страница", min_value=1, max_value=316, value=1
        )
        end_page = st.number_input(
            "Конечная страница", min_value=1, max_value=316, value=316
        )

        if st.button("Собрать данные"):
            data = scrape_series_data(start_page=start_page, end_page=end_page)
            st.write(data)
            st.download_button(
                label="Скачать данные в CSV",
                data=data.to_csv(index=False),
                file_name="series_data.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
