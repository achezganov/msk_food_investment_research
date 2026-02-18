import pandas as pd
import numpy as np
import re

# Константы
WEEK_DAYS = {"пн", "вт", "ср", "чт", "пт"}
WEEKEND_DAYS = {"сб", "вс"}
ALL_DAYS = WEEK_DAYS | WEEKEND_DAYS

DAY_START = 6
DAY_END = 22


# Вспомогательные функции для hours_parser 
def parse_time_range(time_range: str):
    """
    Парсит строку временного диапазона и возвращает время начала и конца
    в виде чисел с плавающей точкой (в часах).

    Поддерживаемые форматы:
        "HH:MM–HH:MM"
        "HH:MM-HH:MM"

    Логика:
    1. Приводит длинное тире к стандартному дефису.
    2. Извлекает часы и минуты через регулярное выражение.
    3. Переводит время в float-формат:
           10:30 -> 10.5
    4. Если конец меньше или равен началу —
       считается переход через полночь, и к концу прибавляется 24 часа.

    Примеры:
        "22:00-02:00" -> (22.0, 26.0)
        "10:00-18:30" -> (10.0, 18.5)

    Возвращает:
        tuple(float, float) | None
        (start, end) — если формат корректен
        None — если строка не распознана
    """

    time_range = time_range.replace("–", "-")

    match = re.match(r"(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})", time_range)
    if not match:
        return None

    h1, m1, h2, m2 = map(int, match.groups())

    start = h1 + m1 / 60
    end = h2 + m2 / 60

    if end <= start:
        end += 24

    return start, end


def intersects_day(start: float, end: float):
    """
    Проверяет, пересекается ли интервал работы с дневным временем.

    Дневное время определяется как:
        06:00–22:00

    Логика:
    Проверяется пересечение двух интервалов:
        [start, end] и [6, 22]

    Возвращает:
        True  — если есть хотя бы частичное пересечение
        False — если пересечения нет
    """
    return max(start, DAY_START) < min(end, DAY_END)


def intersects_night(start: float, end: float):
    """
    Проверяет, пересекается ли интервал работы с ночным временем.

    Ночное время определяется как:
        22:00–24:00
        00:00–06:00

    Логика:
    Проверяется пересечение с двумя интервалами.
    Если есть пересечение хотя бы с одним —
    считается, что заведение работает ночью.

    Возвращает:
        True  — если есть ночная работа
        False — если нет
    """

    night_ranges = [(22, 24), (0, 6)]

    for ns, ne in night_ranges:
        if max(start, ns) < min(end, ne):
            return True

    return False


def expand_days(days_part: str):
    """
    Раскрывает строку с днями недели в множество конкретных дней.

    Поддерживаемые форматы:
        1. Диапазон:
           "пн-пт"
        2. Перечисление:
           "сб,вс"

    Логика:
    1. Приводит строку к нижнему регистру.
    2. Если указан диапазон — возвращает срез из упорядоченного списка дней.
    3. Если перечисление — разбивает по запятой.
    4. Отбрасывает некорректные значения.

    Возвращает:
        set[str] — множество дней недели
    """

    days = set()
    days_part = days_part.lower()

    order = ["пн","вт","ср","чт","пт","сб","вс"]

    range_match = re.match(
        r"(пн|вт|ср|чт|пт|сб|вс)-(пн|вт|ср|чт|пт|сб|вс)",
        days_part
    )

    if range_match:
        start, end = range_match.groups()
        i1 = order.index(start)
        i2 = order.index(end)
        days.update(order[i1:i2+1])
    else:
        for d in re.split(r",\s*", days_part):
            if d in ALL_DAYS:
                days.add(d)

    return days

# Основная функция
def hours_parser(hours_str: str):
    """
    Основная функция анализа поля 'hours'.

    Принимает строку с расписанием работы заведения
    и возвращает структурированные признаки.

    Логика:
    is_24_7 = True, если выполняется хотя бы одно условие:

    1) В строке присутствуют одновременно:
       - "круглосуточно"
       - "ежедневно"

    2) В строке присутствуют одновременно:
       - "круглосуточно"
       - "пн-вс"

    3) Указано время фактически 24 часа ежедневно:
       - "ежедневно 00:00-23:59"
       - "ежедневно 00:00-00:00"
       - "ежедневно 00:00-24:00"

    Во всех остальных случаях is_24_7 = False.

    Возвращаемые признаки:
    is_24_7 : bool
        Работает ли заведение 24/7.

    is_night : bool
        Есть ли пересечение с ночным временем (22:00–06:00).

    is_day : bool
        Есть ли пересечение с дневным временем (06:00–22:00).

    on_week : bool
        Работает ли в будни.

    on_weekend : bool
        Работает ли в выходные.

    hours_on_week : float
        Суммарные часы работы в будни.

    hours_on_weekend : float
        Суммарные часы работы в выходные.

    Поддерживаемые форматы:
        - "ежедневно 10:00-22:00"
        - "пн-пт 09:00-18:00; сб,вс 10:00-16:00"
        - "ежедневно, круглосуточно"
        - "пн-вс круглосуточно"

    Возвращает:
        pandas.Series с рассчитанными признаками.
    """

    if pd.isna(hours_str):
        return pd.Series({
            "is_24_7": False,
            "is_night": False,
            "is_day": False,
            "on_week": False,
            "on_weekend": False,
            "hours_on_week": 0.0,
            "hours_on_weekend": 0.0
        })

    text = str(hours_str).lower().replace("–", "-")

    is_24_7 = False
    is_night = False
    is_day = False
    on_week = False
    on_weekend = False
    hours_on_week = 0.0
    hours_on_weekend = 0.0

    # Обработка 24/7
    if (
        ("круглосуточно" in text and "ежедневно" in text)
        or ("круглосуточно" in text and "пн-вс" in text)
    ):
        is_24_7 = True

    if not is_24_7 and text.startswith("ежедневно"):
        match = re.search(r"\d{1,2}:\d{2}-\d{1,2}:\d{2}", text)
        if match:
            parsed = parse_time_range(match.group())
            if parsed:
                start, end = parsed
                if start == 0 and (end - start) >= 23.99:
                    is_24_7 = True

    if is_24_7:
        return pd.Series({
            "is_24_7": True,
            "is_night": True,
            "is_day": True,
            "on_week": True,
            "on_weekend": True,
            "hours_on_week": 24 * 5,
            "hours_on_weekend": 24 * 2
        })

    # Обработка "ежедневно" + время
    if text.startswith("ежедневно"):
        match = re.search(r"\d{1,2}:\d{2}-\d{1,2}:\d{2}", text)
        if match:
            parsed = parse_time_range(match.group())
            if parsed:
                start, end = parsed
                hours = end - start

                is_day = intersects_day(start, end)
                is_night = intersects_night(start, end)

                return pd.Series({
                    "is_24_7": False,
                    "is_night": is_night,
                    "is_day": is_day,
                    "on_week": True,
                    "on_weekend": True,
                    "hours_on_week": hours * 5,
                    "hours_on_weekend": hours * 2
                })

    # Обработка сегментированного формата
    segments = text.split(";")

    for segment in segments:
        segment = segment.strip()

        match = re.match(
            r"([а-я,\-]+)\s+(\d{1,2}:\d{2}-\d{1,2}:\d{2})",
            segment
        )

        if not match:
            continue

        days_part, time_part = match.groups()
        days = expand_days(days_part)

        parsed = parse_time_range(time_part)
        if parsed is None:
            continue

        start, end = parsed
        hours = end - start

        if intersects_day(start, end):
            is_day = True

        if intersects_night(start, end):
            is_night = True

        for d in days:
            if d in WEEK_DAYS:
                on_week = True
                hours_on_week += hours
            elif d in WEEKEND_DAYS:
                on_weekend = True
                hours_on_weekend += hours

    return pd.Series({
        "is_24_7": False,
        "is_night": is_night,
        "is_day": is_day,
        "on_week": on_week,
        "on_weekend": on_weekend,
        "hours_on_week": hours_on_week,
        "hours_on_weekend": hours_on_weekend
    })

# Основная функция avg_bill parser 
def parse_middle_beer_cup(value: str):
    """
    Парсит колонку avg_bill и извлекает цену бокала пива.

    Логика:
    - Если строка содержит 'цена бокала пива:'
        - если найдено два числа → возвращается их медиана (среднее)
        - если найдено одно число → возвращается это число
    - В остальных случаях возвращается -1

    Параметры:
        value (str): значение из колонки avg_bill

    Возвращает:
        float: рассчитанное значение или np.nan
    """

    if pd.isna(value):
        return np.nan

    value = str(value).lower().strip()

    # Проверяем ключевую фразу
    if "цена бокала пива" not in value:
        return np.nan

    # Извлекаем все числа
    numbers = re.findall(r'\d+', value)

    if not numbers:
        return np.nan

    numbers = list(map(int, numbers))

    # Если одно число
    if len(numbers) == 1:
        return float(numbers[0])

    # Если два и более чисел -> берём медиану первых двух
    if len(numbers) >= 2:
        return float(np.median(numbers[:2]))

    return np.nan