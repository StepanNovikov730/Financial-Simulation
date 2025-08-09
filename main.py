import random
import numpy as np
import time
import os
import datetime

# Импорты из наших модулей
from config import (
    RANDOM_SEED, PLANS, N_SCENARIOS, N_MONTHS, HORIZONS,
    SAVINGS_RETURN_RATE, IDEAL_RETURN_RATE, TAX_RATE, CUSHION_AMOUNT,
    DEBT_INTEREST_RATE, MINOR_EMERGENCY_PROB, MINOR_EMERGENCY_COST,
    MEDIUM_EMERGENCY_PROB, MEDIUM_EMERGENCY_COST, MAJOR_EMERGENCY_PROB, MAJOR_EMERGENCY_COST,
    MINOR_CLUSTER_PROB, MAJOR_CLUSTER_LAMBDA,
    PARTIAL_LOSS_PROB, PARTIAL_LOSS_RATE, PARTIAL_LOSS_DURATION,
    FULL_LOSS_PROB, FULL_LOSS_DURATION_MEAN
)
from simulation_core import run_simulation, initialize_validation_log, finalize_validation_log
from reporting import (
    print_comparative_results, save_results_to_text, save_shock_analysis_to_text, 
    save_planned_expenses_analysis, save_debt_analysis, save_key_scenarios_analysis, 
    save_wealth_distribution_analysis, save_simulation_parameters
)
import config  # Импортируем модуль для установки ANOMALY_LOG_FILE

# ===== ВОСПРОИЗВОДИМОСТЬ =====
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def main():
    """Основная функция запуска симуляции"""
    
    # Вывод информации о симуляции
    print("="*70)
    print(" ВЕКТОРИЗОВАННАЯ ФИНАНСОВАЯ СИМУЛЯЦИЯ С ПЛАНАМИ (ТРАЕКТОРИЯМИ) ")
    print("="*70)
    print(f"Сценариев: {N_SCENARIOS}, Месяцев: {N_MONTHS} (30 лет)")
    print("Базовые параметры:")
    print(f"- Доходность сбережений: {SAVINGS_RETURN_RATE*100:.1f}%/мес (~{SAVINGS_RETURN_RATE*12*100:.0f}% годовых)")
    print(f"- Налог на инвестиционный доход: {TAX_RATE*100:.0f}%")
    print(f"- Идеальная доходность: {IDEAL_RETURN_RATE*100:.1f}%/мес (~{IDEAL_RETURN_RATE*12*100:.0f}% годовых)")
    print(f"- Подушка безопасности: {CUSHION_AMOUNT:,}₽ (0% доходности)")
    print(f"- Ставка по долгу: {DEBT_INTEREST_RATE*100:.1f}%/мес (~{DEBT_INTEREST_RATE*12*100:.0f}% годовых)")
    print(f"- НОВОЕ: Динамические планы с изменением дохода/расходов во времени")
    print(f"- ИСПРАВЛЕНО: Виртуальный сценарий для правильного расчета потерь компаундинга")
    print(f"- ИСПРАВЛЕНО: Единая система долгов во всех сценариях")
    print(f"- ИСПРАВЛЕНО: Запланированные расходы типа 'time' выполняются только при достаточности средств")
    print(f"- ВЕКТОРИЗОВАНО: Батчевая генерация случайных чисел для ускорения")
    print(f"- ОПТИМИЗИРОВАНО: Количество сценариев снижено до {N_SCENARIOS} для веб-версии")
    print(f"- НОВОЕ: Детальная валидация финансовых состояний с логированием")

    print("\nПланы (траектории):")
    for plan_id, plan_data in PLANS.items():
        plan_expenses = plan_data.get('planned_expenses', [])
        print(f"- План {plan_id}: {plan_data['initial_income']:,}₽ → {plan_data['initial_expenses']:,}₽ (стартовый капитал: {plan_data.get('initial_capital', 0):,}₽, расходов: {len(plan_expenses)})")
        if plan_data['income_changes']:
            print(f"  Изменения дохода: {len(plan_data['income_changes'])} событий")
            for change in plan_data['income_changes']:
                month = change['month']
                year = (month - 1) // 12 + 1
                print(f"    ├── Месяц {month} (год {year}): {change['new_income']:,}₽")
        if plan_data['expense_changes']:
            print(f"  Изменения расходов: {len(plan_data['expense_changes'])} событий")
            for change in plan_data['expense_changes']:
                month = change['month']
                year = (month - 1) // 12 + 1
                print(f"    ├── Месяц {month} (год {year}): {change['new_expenses']:,}₽")
        if plan_expenses:
            print(f"  Запланированные расходы: {len(plan_expenses)} событий")
            for i, expense in enumerate(plan_expenses):
                if expense['type'] == 'time':
                    print(f"    ├── {expense['name']}: {expense['amount']:,}₽ (начиная с года {expense['condition']}, при наличии средств)")
                else:
                    print(f"    ├── {expense['name']}: {expense['amount']:,}₽ (при накоплении {expense['condition']:,}₽)")

    print("\nРиски ЧП (Москва 2025):")
    print(f"- Мелкие траты ({MINOR_EMERGENCY_PROB*100:.1f}%/мес): {MINOR_EMERGENCY_COST}₽")
    print(f"- Средние траты ({MEDIUM_EMERGENCY_PROB*100:.1f}%/мес): {MEDIUM_EMERGENCY_COST}₽")
    print(f"- Крупные траты ({MAJOR_EMERGENCY_PROB*100:.1f}%/мес): {MAJOR_EMERGENCY_COST}₽")
    print(f"  Кластеризация мелких/средних: {MINOR_CLUSTER_PROB*100:.0f}% вероятность продолжения")
    print(f"  Кластеризация крупных: распределение Пуассона (λ={MAJOR_CLUSTER_LAMBDA})")
    print(f"- Частичная потеря дохода: {PARTIAL_LOSS_PROB*100:.2f}%/мес ({PARTIAL_LOSS_RATE*100:.0f}% от дохода)")
    print(f"  Средняя длительность: {PARTIAL_LOSS_DURATION:.1f} мес")
    print(f"- Полная потеря дохода: {FULL_LOSS_PROB*100:.3f}%/мес")
    print(f"  Средняя длительность: {FULL_LOSS_DURATION_MEAN:.1f} мес")
    print("="*70)

    # Сохранение результатов в файл
    base_results_dir = r"Y:\code\monte carlo\results"
    try:
        os.makedirs(base_results_dir, exist_ok=True)
        print(f"✓ Базовая директория создана/существует: {base_results_dir}")
    except Exception as e:
        print(f"✗ Ошибка создания базовой директории {base_results_dir}: {e}")
        # Fallback на текущую директорию
        base_results_dir = "results"
        os.makedirs(base_results_dir, exist_ok=True)
        print(f"✓ Используем fallback директорию: {base_results_dir}")

    # Генерация уникального имени папки с датой и временем
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_folder = f"simulation_vectorized_{timestamp}"  # Изменено название папки
    results_dir = os.path.join(base_results_dir, unique_folder)
    
    try:
        os.makedirs(results_dir, exist_ok=True)
        print(f"✓ Директория результатов создана: {results_dir}")
    except Exception as e:
        print(f"✗ Ошибка создания директории результатов {results_dir}: {e}")
        return

    # Настройка логирования аномалий
    anomaly_log_filename = "debug_anomalies.log"
    anomaly_log_filepath = os.path.join(results_dir, anomaly_log_filename)

    # ИСПРАВЛЕНО: Устанавливаем глобальный путь для логирования аномалий
    config.ANOMALY_LOG_FILE = anomaly_log_filepath
    print(f"✓ Путь к логу валидации установлен: {config.ANOMALY_LOG_FILE}")

    # НОВОЕ: Инициализация лога валидации
    print(f"\nИнициализация системы валидации...")
    initialize_validation_log()

    # Имена файлов в уникальной папке
    txt_filename = "main_results.txt"
    txt_filepath = os.path.join(results_dir, txt_filename)

    shock_filename = "shock_analysis.txt"
    shock_filepath = os.path.join(results_dir, shock_filename)

    planned_filename = "planned_expenses_analysis.txt"
    planned_filepath = os.path.join(results_dir, planned_filename)

    debt_filename = "debt_analysis.txt"
    debt_filepath = os.path.join(results_dir, debt_filename)

    # НОВОЕ: Файлы для ключевых сценариев и распределения активов
    key_scenarios_filename = "key_scenarios_analysis.txt"
    key_scenarios_filepath = os.path.join(results_dir, key_scenarios_filename)

    wealth_distribution_filename = "wealth_distribution_analysis.txt"
    wealth_distribution_filepath = os.path.join(results_dir, wealth_distribution_filename)

    # НОВОЕ: Файл с параметрами симуляции
    params_filename = "simulation_parameters.txt"
    params_filepath = os.path.join(results_dir, params_filename)

    print(f"\nСохранение результатов в папку: {results_dir}")
    print(f"  ├── {txt_filename}")
    print(f"  ├── {shock_filename}")
    print(f"  ├── {planned_filename}")
    print(f"  ├── {debt_filename}")
    print(f"  ├── {key_scenarios_filename}")
    print(f"  ├── {wealth_distribution_filename}")
    print(f"  ├── {params_filename}")
    print(f"  └── {anomaly_log_filename} (лог валидации - создается всегда)")

    # Запуск симуляций
    start_total = time.time()
    all_results = {}

    for plan_id, plan_data in PLANS.items():
        all_results[plan_id] = run_simulation(plan_id, plan_data)

    total_time = time.time() - start_total

    # НОВОЕ: Финализация лога валидации
    print(f"\nФинализация валидации...")
    finalize_validation_log()

    # Вывод результатов
    print_comparative_results(all_results)

    print(f"\nСохранение результатов в файлы...")
    try:
        save_results_to_text(all_results, txt_filepath)
        save_shock_analysis_to_text(all_results, shock_filepath)
        save_planned_expenses_analysis(all_results, planned_filepath)
        save_debt_analysis(all_results, debt_filepath)
        save_key_scenarios_analysis(all_results, key_scenarios_filepath)
        save_wealth_distribution_analysis(all_results, wealth_distribution_filepath)
        save_simulation_parameters(params_filepath)
        print("✓ Результаты успешно сохранены!")
        print(f"✓ Путь к папке: {results_dir}")
        
        # ОБНОВЛЕНО: Проверка лога валидации (теперь создается всегда)
        if os.path.exists(anomaly_log_filepath):
            file_size = os.path.getsize(anomaly_log_filepath)
            if file_size > 0:
                # Импортируем статистику валидации
                from config import VALIDATION_STATS
                if VALIDATION_STATS['total_anomalies'] > 0:
                    print(f"⚠️  Обнаружено {VALIDATION_STATS['total_anomalies']:,} финансовых аномалий из {VALIDATION_STATS['total_checks']:,} проверок!")
                    print(f"📋 Детали в файле: {anomaly_log_filename}")
                else:
                    print(f"✅ Валидация пройдена: {VALIDATION_STATS['total_checks']:,} проверок, аномалий не найдено")
                    print(f"📋 Отчет валидации: {anomaly_log_filename}")
            else:
                print(f"⚠️  Лог валидации пуст: {anomaly_log_filename}")
        else:
            print(f"✗ Лог валидации не создан: {anomaly_log_filename}")
            print(f"    Проверьте права доступа к директории: {results_dir}")
            
    except Exception as e:
        print(f"✗ Ошибка при сохранении: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nОбщее время расчета: {total_time/60:.1f} минут")
    print(f"Скорость: {N_SCENARIOS/total_time:.0f} сценариев/сек")
    print(f"ВЕКТОРИЗОВАНО: Использованы батчи случайных чисел для ускорения")


if __name__ == "__main__":
    main()