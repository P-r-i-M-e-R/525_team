from typing import Any

import numpy as np
import spacy

nlp = spacy.load("ru_core_news_sm")


def sort_boxes_into_reading_order(
    boxes: list,
    col_threshold: float = None
) -> list:
    
    # 0) Предварительные расчёты
    # Центры по X, Y и ширины
    x_mins = np.array([b[0] for b in boxes])
    x_maxs = np.array([b[2] for b in boxes])
    widths = x_maxs - x_mins
    
    # Порог на «ширину» колонки
    if col_threshold is None:
        col_threshold = float(np.median(widths)) * 0.8
    
    # Группируем по колонкам
    sorted_idx = np.argsort(x_mins)
    columns = []
    columns_x = []  # средний x_min каждой колонки
    
    for idx in sorted_idx:
        box = boxes[idx]
        x0, y0, x1, y1, text = box
        cx = (x0 + x1) / 2
        
        # ищем колонку, куда этот бокс «вписывается»
        assigned = False
        for col_i, col_x in enumerate(columns_x):
            # считае 
            if abs(cx - col_x) < col_threshold:
                columns[col_i].append(box)
                # обновляем средний центр колонки
                columns_x[col_i] = np.mean([(b[0] + b[2]) / 2 for b in columns[col_i]])
                assigned = True
                break
        
        # если не подошёл ни к одной — создаём новую колонку
        if not assigned:
            columns.append([box])
            columns_x.append(cx)
    
    # Сортируем колонки слева-направо по среднему x
    cols_sorted = [c for _, c in sorted(zip(columns_x, columns), key=lambda x: x[0])]
    
    # Внутри каждой колонки сортируем боксы по y_min (сверху→вниз)
    for i, col in enumerate(cols_sorted):
        cols_sorted[i] = sorted(col, key=lambda b: b[1])
    
    # Склеиваем всё в один список
    ordered = []
    for col in cols_sorted:
        ordered.extend(col)
    
    return ordered, cols_sorted


def preprocess(text: str, nlp) -> list[str]:
    text = text.lower()
    text = ' '.join(text.replace('-\n', '').split('\n'))
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc
              if not token.is_stop and not token.is_punct and len(token)>2]
    return tokens


def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def matching_boxes(
    columns: list[list[list[Any]]],
    ordered: list[list[Any]],
    sim_threshold: float = 0.7,
) -> list[list[str]]:

    sim_threshold = sim_threshold
    articles = []
    embeddings = [
        [nlp(text[-1]).vector for text in columns[col]] for col in range(len(columns))
    ]
    used = {i: [False] * len(columns[i]) for i in range(len(columns))}

    idx = 0
    cur_article = []
    cur_col = 0

    while sum(sum(used[i]) for i in range(len(columns))) != len(ordered):
        print(cur_col, idx)
        if used[cur_col][idx]:
            if idx == len(used[cur_col]) - 1:
                cur_col += 1
                idx = 0
                continue
            if cur_col == len(columns) - 1:
                idx += 1
                continue
            idx += 1
            continue
        used[cur_col][idx] = True

        if cur_article == []:
            cur_article = [f'{cur_col}_{idx}']

        # пара заголовок + текс сразу мапим, тут сразу гарантируем то, что какой-то кусок текста будет под заголовком
        if len(columns[cur_col][idx][-1].split()) <= 7 and idx != len(columns[cur_col]) - 1:
            cur_article.append(f'{cur_col}_{idx + 1}')
            idx += 1
            continue

        if idx == len(used[cur_col]) - 1 and cur_col == 6:
            break


        if cur_col == len(columns) - 1:
            while idx < len(used[cur_col]) - 1:
                # в конце нет окончания предл + начало с маленькой след абзаца
                if (all(punkt not in columns[cur_col][idx][-1][-2:] for punkt in '.!?') or columns[cur_col][idx][-1][-1:] == '-') and columns[cur_col][idx + 1][-1][0].islower():
                    cur_article.append(f'{cur_col}_{idx + 1}')
                    used[cur_col][idx + 1] = True
                    idx += 1
                elif max(
                    cosine_similarity(embeddings[cur_col][idx], embeddings[cur_col][idx + 1]),
                    cosine_similarity(embeddings[cur_col][idx], embeddings[cur_col][idx + 1])
                ) > sim_threshold:
                    cur_article.append(f'{cur_col}_{idx + 1}')
                    used[cur_col][idx + 1] = True
                    idx += 1
                else:
                    break
            if idx != len(used[cur_col]) - 1:
                # переход на левый верхний угол, которые не использован
                flag = False
                for i, col in enumerate(list(used.keys())[:-1]):
                    if sum(used[col]) != len(columns[col]):
                        cur_col = i
                        idx = sorted(used[col]).index(False)
                        articles.append(cur_article)
                        cur_article = []
                        flag = True
                        break
                if flag:
                    continue
                else:
                    idx += 1
                    articles.append(cur_article)
                    cur_article = []
                    continue
            else:
                # подразумеваем, что все прошли
                articles.append(cur_article)
                cur_article = []
                break

        # когда мы внизу страницы
        if idx == len(used[cur_col]) - 1:
            right_index = -1
            for i, _ in enumerate(used[cur_col + 1]):
                cur_idxs = cur_article[0].split('_')
                idx1 = int(cur_idxs[0])
                idx2 = int(cur_idxs[1])
                print(columns[cur_col + 1][i][1], int(columns[idx1][idx2][1]))
                if not used[cur_col + 1][i] and abs(columns[cur_col + 1][i][1] - int(columns[idx1][idx2][1])) < 500:
                    right_index = i
                    break

            if right_index != -1:
                extra_rule = 0
                if (all(punkt not in columns[cur_col][idx][-1][-2:] for punkt in '.!?') or columns[cur_col][idx][-1][-1:] == '-')\
                    and columns[cur_col + 1][right_index][-1][0].islower():
                    cur_article.append(f'{cur_col + 1}_{right_index}')
                    idx = right_index
                    cur_col += 1
                    continue
                elif max(
                    cosine_similarity(embeddings[cur_col][idx], embeddings[cur_col + 1][right_index]),
                    cosine_similarity(embeddings[cur_col][idx], embeddings[cur_col + 1][right_index])
                ) + extra_rule * 0.1 > sim_threshold:
                    cur_article.append(f'{cur_col + 1}_{right_index}')
                    idx = right_index
                    cur_col += 1
                    continue
                else:
                    articles.append(cur_article)
                    cur_article = []
                    cur_col += 1
                    try:
                        idx = sorted(used[cur_col]).index(False)
                    except ValueError:
                        idx = 0
                    continue
            else:
                articles.append(cur_article)
                cur_article = []
                cur_col += 1
                try:
                    idx = sorted(used[cur_col]).index(False)
                except ValueError:
                    idx = 0
                continue


        extra_score = 0
        if len(columns[cur_col][idx + 1][-1].split()) > 7:
            extra_score = 0.03

        if (all(punkt not in columns[cur_col][idx][-1][-1:] for punkt in '.!?') or columns[cur_col][idx][-1][-1:] == '-') and columns[cur_col][idx + 1][-1][0].islower():
            cur_article.append(f'{cur_col}_{idx + 1}')
            used[cur_col][idx + 1] = True
            idx += 1
        elif max(
            cosine_similarity(embeddings[cur_col][idx], embeddings[cur_col][idx + 1]),
            cosine_similarity(embeddings[cur_col][idx], embeddings[cur_col][idx + 1])
        ) + extra_score > sim_threshold and abs(columns[cur_col][idx][3] - int(columns[cur_col][idx + 1][1])) < 1000:
            cur_article.append(f'{cur_col}_{idx + 1}')
            idx += 1
        else:
            articles.append(cur_article)
            cur_article = []
            idx += 1
        
    if cur_article != []:
        articles.append(cur_article)
    return articles
