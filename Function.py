import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import streamlit as st

def scrape_series_data(start_page=1, end_page=316, base_url='https://kino.mail.ru/series/all/?order=rate_count&year=1916&year=2024'):
    items_urls = []

    for i in tqdm(range(start_page, end_page + 1)):
        end = '/' if i == 1 else f'&page={i}'
        link = f'{base_url}{end}'
        content = requests.get(link).content
        soup = BeautifulSoup(content, 'lxml')
        series = soup.find_all(class_='link link_inline link-holder link-holder_itemevent link-holder_itemevent_small')

        for item in series:
            item_page_url = item.get('href')
            full_url = 'https://kino.mail.ru' + item_page_url
            if full_url not in items_urls:
                items_urls.append(full_url)

    titles, sources, descriptions, ganres, years, posters, casts, ratings = [], [], [], [], [], [], [], []

    for url in tqdm(items_urls):
        try:
            text = requests.get(url).text
            soup = BeautifulSoup(text, 'lxml')
            try:
                title = soup.find('h1', class_='text text_bold_giant color_white').text
                titles.append(title[:title.find(' (сери')])
            except:
                titles.append(None)
            try:
                picture_url = soup.find('meta', itemprop='image')['content']
                posters.append(picture_url)
            except:
                posters.append(None)
            try:
                ganre = soup.find_all('span', class_='badge__text')
                helper = [i.text for i in ganre]
                ganres.append(helper)
            except:
                ganres.append(None)
            try:
                cast = soup.find_all('span', class_='p-truncate__inner js-toggle__truncate-inner')
                a = [act.text for act in cast]
                casts.append(a[1:4])
            except:
                casts.append(None)
            try:
                description = soup.find('div', class_='text text_inline text_light_medium text_fixed valign_baseline p-movie-info__description-text').text
                descriptions.append(description.replace('\xa0', ' '))
            except:
                descriptions.append(None)
            try:
                year = soup.find('span', class_='nowrap').text
                years.append(year.split()[:3])
            except:
                years.append(None)
            try:
                rating = float(soup.find('span', class_='text text_bold_huge text_fixed').text)
                ratings.append(rating)
            except:
                ratings.append(None)

            sources.append(url)

        except requests.exceptions.RequestException as e:
            print(f"ссылка номер: {items_urls.index(url)}, ссылка: {url}, ошибка: {e}")

    data = pd.DataFrame({
        'url': sources,
        'poster': posters,
        'title': titles,
        'ganres': ganres,
        'description': descriptions,
        'year': years,
        'rating': ratings,
        'cast': casts
    })

    return data

# Пример использования функции в Streamlit
st.title('Сбор данных о сериалах')
start_page = st.number_input('Начальная страница', min_value=1, max_value=316, value=1)
end_page = st.number_input('Конечная страница', min_value=1, max_value=316, value=316)

if st.button('Собрать данные'):
    data = scrape_series_data(start_page=start_page, end_page=end_page)
    st.write(data)
    st.download_button(label='Скачать данные в CSV', data=data.to_csv(index=False), file_name='series_data.csv', mime='text/csv')