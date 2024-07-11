import requests
from bs4 import BeautifulSoup

base_url = "https://myshows.me"
search_url = base_url + "/search/all/"

response = requests.get(search_url)
data = BeautifulSoup(response.text, "lxml")

film_info_list = []

for film_data in data.find_all("a", class_="ShowCatalogCard"):
    tvshow_title = film_data.find("div", class_="ShowCatalogCard__title").text

    image_tag = film_data.find("img", class_="ShowCatalogCard__poster-image")
    image_url = image_tag["src"]

    page_url = base_url + film_data["href"]

    # Заходим на страницу фильма и извлекаем описание
    film_response = requests.get(page_url)
    film_page = BeautifulSoup(film_response.text, "lxml")

    try:
        description = (
            film_page.find("div", class_="Container")
            .find("div", class_="HtmlContent")
            .find("p")
            .text
        )
    except AttributeError:
        description = "Описание не найдено"

    film_info_list.append(
        {
            "title": tvshow_title,
            "image_url": image_url,
            "page_url": page_url,
            "description": description,
        }
    )

    print(tvshow_title, "-----", image_url, "-----", page_url, "-----", description)

# Вы можете сохранить film_info_list в CSV или вывести как HTML, если необходимо
