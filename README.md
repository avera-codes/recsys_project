# Умный поиск сериалов
Данный сервис рекомендует сериалы на основе короткого текстового запроса от пользователя.


Над сервисом работали [Аверченков Артемий](https://github.com/avera-codes) и [Наданьян Сергей](https://github.com/Sergik1994) 

---


## Содержание

- #### [Библиотеки](#библиотеки)
- #### [Работа с данными](#работа-с-данными)
  - ##### [Пояснение к данным](#пояснение-к-данным)
  - ##### [Парсинг](#парсинг)
- #### [Модель](#модель)
- Streamlit

---

## Библиотеки
Для реализации сервиса мы использовали следующие блиблиотеки:
| Import   | Описание                                                    | Использование в проекте                                                                                   |
|--------------|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `pandas`       | Библиотека для манипуляций и анализа табличных (и не только) данных на Python.| Чтение CSV-файлов, манипуляции с DataFrame и фильтрации/сортировки данных.                  |
| `numpy`        | Библиотека для вычислений на Python.             | Обработка числовых операций, включая создание и манипуляцию массивами.                             |
| `sklearn`      | Библиотека для машинного обучения на Python.                 | Вычисление косинусного сходства между эмбеддингами запроса пользователя и описания сериала.                                       |
| `transformers` | Библиотека для современных методов обработки естественного языка (NLP). | Создание эмбеддингов текстовых описаний с помощью модели `rubert-tiny2`.                            |
| `pytorch`        | Фреймворк для глубокого обучения и работы с нейросетями на Python.                  | Загрузка и использование нейросетевой модели.                                          |
| `streamlit`    | Фреймворк для создания веб-приложений для проектов машинного обучения и анализа данных. | Создание веб-интерфейса сервиса умного поиска сериалов                                             |

---

## Работа с данными
Для проекта мы использовали даные с сайта [kino.mail.ru](https://kino.mail.ru/series/all/?order=rate_count&year=1916&year=2024). Количество уникальных обьектов составило 15750. Структура датасета выглядит следующим образом:

| url | poster | title | description | rating | year | genres | cast | country | age |
|-----|--------|-------|-------------|--------|------|--------|------|---------|-----|
| "https://kino.mail.ru/series_885164_zhizn_v_det..." | "https://resizer.mail.ru/p/0e126721-3ca5-5a6d-b..." | "Жизнь в деталях" | "Комедийный сериал, рассказывающий забавные истории..." | 7.4 | 2015 | "Комедия" | "Колин Хэнкс, Зои Листер Джонс, Томас Садоски, ..." | "США" | "21 сентября 2015 (РФ)" |


### Пояснение к данным
| Столбец                 | Описание                                                                                     | Тип данных  |
|-------------------------|----------------------------------------------------------------------------------------------|-------------|
| url                 | URL-адрес страницы с описанием сериала.                                                       | object      |
| poster              | URL-адрес постера сериала.                                                                    | object      |
| title               | Название сериала.                                                                             | object      |
| description         | Описание сериала.                                                                             | object      |
| rating              | Рейтинг сериала. Если отсутствует, значения могут быть пропущены.                              | float64     |
| year                | Год выпуска сериала.                                                                          | int64       |
| genres              | Жанры сериала, разделенные запятыми.                                                          | object      |
| cast                | Актерский состав сериала, разделенный запятыми.                                               | object      |
| country             | Страна производства сериала.                                                                  | object      |
| age                 | Рейтинг возрастного ограничения     | object      |


### Парсинг 

Для парсинга данных с сайта была использована библиотека `BeautifulSoup` и парсер `lxml`.

На выходе мы получили датасет такого содержания:

| #  | Column       | Non-Null Count  | Dtype   |
|----|--------------|-----------------|---------|
| 0  | url          | 15750 non-null  | object  |
| 1  | poster       | 15750 non-null  | object  |
| 2  | title        | 15750 non-null  | object  |
| 3  | description  | 15750 non-null  | object  |
| 4  | rating       | 5737 non-null   | float64 |
| 5  | year         | 15750 non-null  | int64   |
| 6  | genres       | 15744 non-null  | object  |
| 7  | cast         | 15653 non-null  | object  |
| 8  | country      | 15647 non-null  | object  |
| 9  | age          | 15617 non-null  | object  |

---

## Модель

Для реализации сервиса мы взяли предобученную модель **[`rubert-tiny2`](https://huggingface.co/cointegrated/rubert-tiny2)**.

`rubert-tiny2` имеет словарь токенов размеров 80000 и максимальную длину обработки текста 2048 токенов. Более подробно про модель можно прочитать [на хабре](https://habr.com/ru/articles/669674/) или ознакомиться с кратким описанием на английском языке на [hugging face](https://huggingface.co/cointegrated/rubert-tiny2).

