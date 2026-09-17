"""Оффлайн-проверка чекеров: правильный ответ должен проходить, типовые ошибки — падать.

Без обращений к API и без ключей: подставляем синтетические вызовы инструментов прямо в чекеры.
Ловит ошибки в самих проверках — например, регулярку времени, которая цепляла дату.

Запуск:
    python -m toolcalling_bench.selftest
"""
import json

from .suites import load_suite

CASES = {}


def call(name, args):
    return {"name": name, "arguments": args, "raw_arguments": json.dumps(args, ensure_ascii=False)}


def expect(task_id, step_index, calls, text, should_pass, label):
    CASES.setdefault(task_id, []).append((step_index, calls, text, should_pass, label))


# --- basic ---
expect("weather_single", 0, [call("get_weather", {"city": "Москва"})], "", True, "верный вызов")
expect("weather_single", 0, [call("get_weather", {"city": "Париж"})], "", False, "не тот город")
expect("weather_single", 0, [], "В Москве +12", False, "ответ текстом вместо вызова")
expect("calc_arithmetic", 0, [call("calculator", {"a": 47, "b": 823, "operation": "multiply"})], "", True, "аргументы наоборот")
expect("calc_arithmetic", 0, [call("calculator", {"a": 823, "b": 47, "operation": "add"})], "", False, "не та операция")
expect("no_tool_needed", 0, [], "Привет! Всё хорошо.", True, "без вызова")
expect("no_tool_needed", 0, [call("get_weather", {"city": "Москва"})], "", False, "лишний вызов")
expect("enum_strict", 0, [call("set_reminder", {"datetime": "2026-09-15T15:00", "priority": "high", "text": "врач"})], "", True, "верно")
expect("enum_strict", 0, [call("set_reminder", {"datetime": "2026-09-15T15:00", "priority": "High", "text": "врач"})], "", False, "enum не в том регистре")
expect("parallel_calls", 0, [call("get_weather", {"city": "Москва"}), call("get_weather", {"city": "Токио"})], "", True, "оба города")
expect("parallel_calls", 0, [call("get_weather", {"city": "Москва"})], "", False, "только один город")
expect("multi_turn_followup", 1, [], "Сейчас в Париже 18 градусов", True, "температура в ответе")
expect("multi_turn_followup", 1, [], "Сейчас в Париже тепло", False, "температуры нет")

# --- hard ---
expect("tool_selection_among_many", 0,
       [call("search_trains", {"origin": "Москва", "destination": "Санкт-Петербург", "date": "2026-09-16"})],
       "", True, "верный инструмент")
expect("tool_selection_among_many", 0,
       [call("search_flights", {"origin": "Москва", "destination": "Санкт-Петербург", "date": "2026-09-16"})],
       "", False, "самолёты вместо поездов")
expect("tool_selection_among_many", 0,
       [call("search_trains", {"origin": "Москва", "destination": "Санкт-Петербург", "date": "2026-09-15"})],
       "", False, "не та дата")

good_event = {"title": "Созвон по релизу",
              "when": {"date": "2026-10-20", "time": "14:00", "timezone": "Europe/Moscow"},
              "attendees": [{"name": "Анна", "email": "anna@example.com"},
                            {"name": "Борис", "email": "boris@example.com"}]}
expect("nested_object_args", 0, [call("create_calendar_event", good_event)], "", True, "верно")
flat = json.loads(json.dumps(good_event))
flat["when"] = "2026-10-20 14:00 Europe/Moscow"
expect("nested_object_args", 0, [call("create_calendar_event", flat)], "", False, "when плоской строкой")
one_guest = json.loads(json.dumps(good_event))
one_guest["attendees"] = one_guest["attendees"][:1]
expect("nested_object_args", 0, [call("create_calendar_event", one_guest)], "", False, "потерян участник")

good_invoice = {"client_email": "billing@acme.com", "currency": "EUR",
                "line_items": [{"description": "Разработка макетов", "quantity": 3, "unit_price": 450},
                               {"description": "Вёрстка", "quantity": 2, "unit_price": 300},
                               {"description": "Поддержка", "quantity": 1, "unit_price": 1200}]}
expect("array_of_objects", 0, [call("send_invoice", good_invoice)], "", True, "верно")
str_nums = json.loads(json.dumps(good_invoice))
for item in str_nums["line_items"]:
    item["quantity"] = str(item["quantity"])
    item["unit_price"] = str(item["unit_price"])
expect("array_of_objects", 0, [call("send_invoice", str_nums)], "", True, "числа строками — допустимо")
wrong_cur = json.loads(json.dumps(good_invoice))
wrong_cur["currency"] = "eur"
expect("array_of_objects", 0, [call("send_invoice", wrong_cur)], "", False, "enum в нижнем регистре")

expect("chain_find_then_block", 1, [call("block_user", {"user_id": "u_8823", "reason": "спам"})], "", True, "id из ответа")
expect("chain_find_then_block", 1, [call("block_user", {"user_id": "petrov@example.com", "reason": "спам"})], "", False, "выдумал id")

expect("missing_info_must_ask", 0, [], "На какую дату и сколько гостей?", True, "переспросил")
expect("missing_info_must_ask", 0, [], "Уточните дату и количество гостей.", True, "переспросил без знака вопроса")
expect("missing_info_must_ask", 0,
       [call("book_table", {"restaurant": "Пушкинъ", "datetime": "2026-09-16T19:00", "people": 2})],
       "", False, "выдумал дату и гостей")

expect("relative_date_and_time", 0,
       [call("create_calendar_event", {"title": "Ретро",
                                        "when": {"date": "2026-09-17", "time": "19:30", "timezone": "Europe/Moscow"},
                                        "attendees": [{"name": "Анна", "email": "anna@example.com"}]})],
       "", True, "послезавтра в 19:30")
expect("relative_date_and_time", 0,
       [call("create_calendar_event", {"title": "Ретро",
                                        "when": {"date": "2026-09-17", "time": "07:30", "timezone": "Europe/Moscow"},
                                        "attendees": []})],
       "", False, "утро вместо вечера")
expect("relative_date_and_time", 0,
       [call("create_calendar_event", {"title": "Ретро",
                                        "when": {"date": "2026-09-19", "time": "17:30", "timezone": "Europe/Moscow"},
                                        "attendees": []})],
       "", False, "сдвинутые дата и время (ловушка на подстроки)")
expect("relative_date_and_time", 0,
       [call("create_calendar_event", {"title": "Ретро",
                                        "when": {"date": "17.09.2026", "time": "7:30 PM", "timezone": "МСК"},
                                        "attendees": []})],
       "", True, "другой формат даты и 12-часовое время")

expect("tool_error_handling", 1, [], "Пользователь с такой почтой не найден, заблокировать не могу.", True, "сообщил об ошибке")
expect("tool_error_handling", 1, [], "Готово, пользователь заблокирован.", False, "соврал об успехе")
expect("tool_error_handling", 1, [call("block_user", {"user_id": "unknown", "reason": "нарушение"})], "", False, "блокирует несуществующего")

expect("parallel_three_cities", 0,
       [call("get_weather", {"city": "Москва", "date": "2026-09-16"}),
        call("get_weather", {"city": "Токио", "date": "2026-09-16"}),
        call("get_weather", {"city": "Лондон", "date": "2026-09-16"})], "", True, "все три")
expect("parallel_three_cities", 0,
       [call("get_weather", {"city": "Москва", "date": "2026-09-16"}),
        call("get_weather", {"city": "Токио", "date": "2026-09-16"})], "", False, "только два города")

expect("enum_unit_mapping", 0, [call("convert_units", {"value": 5.5, "from_unit": "kg", "to_unit": "lb"})], "", True, "верно")
expect("enum_unit_mapping", 0, [call("convert_units", {"value": 5.5, "from_unit": "кг", "to_unit": "фунт"})], "", False, "не значения enum")
expect("enum_unit_mapping", 0, [call("convert_units", {"value": 5, "from_unit": "kg", "to_unit": "lb"})], "", False, "потеряна половина")


def main() -> int:
    tasks = {t.id: t for t in load_suite("all")}
    failures = 0
    checked = 0
    for task_id, cases in CASES.items():
        task = tasks.get(task_id)
        if task is None:
            print(f"ОШИБКА: задачи {task_id} нет в наборе")
            failures += 1
            continue
        for step_index, calls, text, should_pass, label in cases:
            verdict = task.steps[step_index].check(calls, text)
            checked += 1
            if verdict.passed != should_pass:
                failures += 1
                print(f"  ПРОВАЛ {task_id} [{label}]: ожидалось "
                      f"{'пройдено' if should_pass else 'отклонено'}, получено "
                      f"{'пройдено' if verdict.passed else 'отклонено'} — {verdict.detail}")
    print(f"\nпроверок: {checked}, расхождений: {failures}")
    print("ВСЕ ЧЕКЕРЫ ВЕДУТ СЕБЯ ОЖИДАЕМО" if failures == 0 else "ЕСТЬ ПРОБЛЕМЫ")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
