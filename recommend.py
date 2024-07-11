import pandas as pd
import torch
import numpy as np


import torch
import pandas as pd


def recommend_for_user_id_from_valid(
    user_id: int, valid_df: pd.DataFrame, model: torch.nn.Module, n: int = 10
) -> pd.DataFrame:
    """Returns dataframe top n movies, true and pred ratings

    Args:
        user_id (int): user_id for predictions
        valid_df (pd.DataFrame): valid dataframe
        model (torch.nn.Module): neural net
        n (int, optional): top_n movies will be returned. Defaults to 10.

    Returns:
        pd.DataFrame: with 4 cols: movie_id, title, pred_rating, true_rating
        shape(n, 4)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Получим id и оценки фильмов из валидационной части, которые пользователь как будто бы не видел
    unviewed_movies_id = torch.tensor(
        valid_df[valid_df["user_id"] == user_id]["item_id"].values, device=device
    )
    true_ratings = valid_df[valid_df["user_id"] == user_id]["rating"].values

    # Сделаем батч для модели
    user_for_rec_batch = torch.repeat_interleave(
        torch.tensor([user_id], device=device).long(), repeats=len(unviewed_movies_id)
    )

    # Получим названия фильмов по id
    movie_id = pd.read_csv(
        "aux/ml-100k/u.item", sep="|", encoding="iso-8859-1", header=None
    ).iloc[:, 0:2]
    movie_id.columns = ["id", "title"]

    # Получим названия непросмотренных фильмов и предсказанные рейтинги
    recommended_movie_titles = movie_id[
        movie_id["id"].isin(unviewed_movies_id.detach().cpu().numpy())
    ]["title"].values

    # Получаем предсказанные рейтинги
    with torch.inference_mode():
        pred_ratings = (
            model(user_for_rec_batch, unviewed_movies_id).detach().cpu().numpy()
        )

    # Преобразуем pred_ratings в одномерный массив
    pred_ratings = pred_ratings.flatten()

    # Делаем DataFrame и возвращаем n фильмов, отсортированных по предсказанным оценкам
    recommended_movies = pd.DataFrame(
        {
            "movie_id": unviewed_movies_id.cpu().numpy(),
            "title": recommended_movie_titles,
            "pred_rating": pred_ratings,
            "true_rating": true_ratings,
        }
    )

    return recommended_movies.sort_values("pred_rating", ascending=False).iloc[:n, :]


def recommend_for_user_id_unwatched(
    user_id: int, full_df: pd.DataFrame, id_to_movie: dict, model: torch.nn.Module
) -> pd.DataFrame:
    """Return DataFrame with sorted by pred_rating among unwatched films

    Args:
        user_id (int): user_id for recommendations
        full_df (pd.DataFrame): full dataframe to filter watched items
        id_to_movie (dict): dict with id and corresponding movie title
        model (torch.nn.Module): neural network model

    Returns:
        pd.DataFrame: shape (n_unwatched_movies, 2) with predicted rating and movie title
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Вычисляем число фильмов
    n_unique_movies = full_df["item_id"].nunique()

    # Создаем массив всех идентификаторов фильмов
    all_movies_ids = np.arange(1, n_unique_movies + 1)  # +1 чтобы учитывать id фильмов

    # Идентификаторы фильмов, просмотренных пользователем
    user_watched_ids = full_df[full_df["user_id"] == user_id]["item_id"].unique()

    # Идентификаторы непросмотренных фильмов
    unwatched_items = list(set(all_movies_ids) - set(user_watched_ids))

    # Создаем тензоры для модели и перемещаем их на устройство
    user_tensor = torch.repeat_interleave(
        torch.tensor(user_id).long(), len(unwatched_items)
    ).to(device)
    items_tensor = torch.tensor(unwatched_items).long().to(device)

    # Получаем предсказания оценок
    with torch.inference_mode():
        preds = model(user_tensor, items_tensor).detach().cpu().numpy()

    # Преобразуем идентификаторы в названия фильмов
    movie_titles = [id_to_movie[i] for i in unwatched_items]

    # Проверяем формы массивов перед созданием DataFrame
    print(f"Shape of preds: {preds.shape}")
    print(f"Length of movie_titles: {len(movie_titles)}")

    # Убедимся, что preds является одномерным массивом
    if preds.ndim > 1:
        preds = preds.flatten()

    # Проверим еще раз длины массивов перед созданием DataFrame
    assert len(preds) == len(
        movie_titles
    ), "Lengths of preds and movie_titles do not match."

    # Создаем финальную таблицу с предсказаниями
    result = pd.DataFrame(
        {"pred_rating": preds, "movie_title": movie_titles}
    ).sort_values("pred_rating", ascending=False)

    return result.iloc[:10, :]
