import streamlit as st
import pandas as pd
import random

# Путь к CSV-файлу
CSV_FILE = "/home/artemiy/recsys_project/data/dataset_itog.csv"


# Загрузка данных из CSV
@st.cache
def load_data(file):
    return pd.read_csv(file)


# Основная функция для отображения случайных 10 позиций
def show_random_shows(data):
    # Выбираем случайные 10 позиций
    random_indices = random.sample(range(len(data)), 10)
    random_shows = data.iloc[random_indices]

    # Отображаем данные о сериалах
    st.subheader("Случайные 10 сериалов")
    st.dataframe(random_shows[["title", "description", "rating", "year", "genres"]])


# Основная часть приложения Streamlit
def main():
    st.title("Случайные сериалы")

    # Загружаем данные
    data = load_data(CSV_FILE)

    # Добавляем кнопку для отображения случайных сериалов
    if st.button("Показать случайные сериалы"):
        show_random_shows(data)


if __name__ == "__main__":
    main()
