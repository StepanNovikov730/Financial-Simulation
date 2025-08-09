import numpy as np
import random
import time
import datetime
import os
from scipy.stats import gaussian_kde

# Импорты из config.py (будут доступны после создания config.py) слово
from config import (
    N_SCENARIOS, N_MONTHS, HORIZONS,
    CUSHION_AMOUNT, SAVINGS_RETURN_RATE, IDEAL_RETURN_RATE, TAX_RATE,
    DEBT_INTEREST_RATE, RESTRUCTURING_THRESHOLD_RATIO, BANKRUPTCY_THRESHOLD_RATIO,
    MINOR_EMERGENCY_PROB, MINOR_EMERGENCY_COST,
    MEDIUM_EMERGENCY_PROB, MEDIUM_EMERGENCY_COST,
    MAJOR_EMERGENCY_PROB, MAJOR_EMERGENCY_COST,
    MINOR_CLUSTER_PROB, MAJOR_CLUSTER_LAMBDA,
    PARTIAL_LOSS_PROB, PARTIAL_LOSS_RATE, PARTIAL_LOSS_DURATION,
    FULL_LOSS_PROB, FULL_LOSS_DURATION_MEAN, FULL_LOSS_DURATION_SD,
    VALIDATION_STATS, RANDOM_SEED
)
import config  # Импортируем модуль целиком для доступа к ANOMALY_LOG_FILE

# Включение/выключение валидации (для отладки)
DEBUG_VALIDATION = True  # Установите True для включения валидации


def initialize_validation_log():
    """
    НОВАЯ ФУНКЦИЯ: Инициализирует лог валидации с заголовком
    """
    if not config.ANOMALY_LOG_FILE:
        print("✗ Путь к логу валидации не установлен")
        return
    
    try:
        # Создаем папку если не существует
        log_dir = os.path.dirname(config.ANOMALY_LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            print(f"✓ Папка создана: {log_dir}")
        
        # Сбрасываем статистику
        VALIDATION_STATS['total_checks'] = 0
        VALIDATION_STATS['total_anomalies'] = 0
        VALIDATION_STATS['anomaly_details'] = []
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Создаем новый лог с заголовком
        with open(config.ANOMALY_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(" ОТЧЕТ ВАЛИДАЦИИ ФИНАНСОВЫХ СОСТОЯНИЙ \n")
            f.write("="*80 + "\n")
            f.write(f"Дата и время запуска: {timestamp}\n")
            f.write(f"Random seed: {RANDOM_SEED}\n")
            f.write(f"Количество сценариев: {N_SCENARIOS:,}\n")
            f.write(f"Количество месяцев: {N_MONTHS} (30 лет)\n")
            f.write(f"Горизонты анализа: {HORIZONS} лет\n")
            f.write(f"Валидация включена: {DEBUG_VALIDATION}\n")
            f.write("="*80 + "\n\n")
            f.write("ОПИСАНИЕ ВАЛИДАЦИИ:\n")
            f.write("Проверяется логическая консистентность финансового состояния:\n")
            f.write("- АНОМАЛИЯ: savings <= 0 И annual_growth > 0\n")
            f.write("- Это указывает на ошибку в расчете налогооблагаемой базы\n")
            f.write("- При отсутствии сбережений не должно быть накопленного роста\n\n")
            f.write("НАЙДЕННЫЕ АНОМАЛИИ:\n")
            f.write("-" * 80 + "\n")
        
        print(f"✓ Лог валидации инициализирован: {config.ANOMALY_LOG_FILE}")
        
    except Exception as e:
        print(f"✗ Ошибка инициализации лога валидации {config.ANOMALY_LOG_FILE}: {e}")
        import traceback
        traceback.print_exc()


def finalize_validation_log():
    """
    НОВАЯ ФУНКЦИЯ: Финализирует лог валидации с итоговой статистикой
    """
    if not config.ANOMALY_LOG_FILE:
        print("✗ Путь к логу валидации не установлен")
        return
    
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(config.ANOMALY_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(" ИТОГОВАЯ СТАТИСТИКА ВАЛИДАЦИИ \n")
            f.write("="*80 + "\n")
            f.write(f"Время завершения: {timestamp}\n")
            f.write(f"Общее количество проверок: {VALIDATION_STATS['total_checks']:,}\n")
            f.write(f"Обнаружено аномалий: {VALIDATION_STATS['total_anomalies']:,}\n")
            
            if VALIDATION_STATS['total_checks'] > 0:
                anomaly_rate = (VALIDATION_STATS['total_anomalies'] / VALIDATION_STATS['total_checks']) * 100
                f.write(f"Частота аномалий: {anomaly_rate:.6f}%\n")
            else:
                f.write("Частота аномалий: 0% (проверки не выполнялись)\n")
            
            f.write("\n")
            
            if VALIDATION_STATS['total_anomalies'] == 0:
                f.write("🎉 УСПЕХ: Все финансовые состояния прошли валидацию!\n")
                f.write("✓ Логическая консистентность подтверждена\n")
                f.write("✓ Ошибок в расчете налогооблагаемой базы не обнаружено\n")
                f.write("✓ Система корректно обрабатывает изъятия из сбережений\n")
            else:
                f.write("⚠️  ОБНАРУЖЕНЫ ПРОБЛЕМЫ:\n")
                f.write(f"Найдено {VALIDATION_STATS['total_anomalies']} финансовых аномалий\n")
                f.write("Рекомендуется проанализировать детали выше\n")
                f.write("Возможные причины:\n")
                f.write("- Ошибка в функции handle_savings_withdrawal\n")
                f.write("- Некорректная обработка налогообложения\n")
                f.write("- Проблемы с синхронизацией savings и annual_growth\n")
            
            f.write("\n" + "="*80 + "\n")
        
        # Выводим статистику в консоль
        if VALIDATION_STATS['total_anomalies'] == 0:
            print(f"✓ Валидация завершена успешно: {VALIDATION_STATS['total_checks']:,} проверок, аномалий не найдено")
        else:
            print(f"⚠️  Валидация завершена: {VALIDATION_STATS['total_checks']:,} проверок, найдено {VALIDATION_STATS['total_anomalies']:,} аномалий")
        
    except Exception as e:
        print(f"✗ Ошибка финализации лога валидации {config.ANOMALY_LOG_FILE}: {e}")
        import traceback
        traceback.print_exc()


class RandomBatchManager:
    """
    Класс для эффективного управления батчами случайных чисел
    Ускоряет симуляцию за счет предгенерации случайных чисел
    """
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        self.batch_idx = 0
        self.current_batch = None
        # Индексы для разных типов событий
        self.MINOR_EM_IDX = 0
        self.MEDIUM_EM_IDX = 1  
        self.MAJOR_EM_IDX = 2
        self.CLUSTER_CONTINUE_IDX = 3
        self.CLUSTER_TYPE_IDX = 4
        self.PARTIAL_LOSS_IDX = 5
        self.FULL_LOSS_IDX = 6
        self._generate_new_batch()
    
    def _generate_new_batch(self):
        """Генерирует новый батч случайных чисел"""
        # 7 колонок для основных событий + дополнительные для нормальных/экспоненциальных распределений
        self.current_batch = np.random.random((self.batch_size, 7))
        # Предгенерируем также специальные распределения
        self.poisson_batch = np.random.poisson(MAJOR_CLUSTER_LAMBDA, self.batch_size)
        self.exponential_batch = np.random.exponential(PARTIAL_LOSS_DURATION, self.batch_size)
        self.normal_batch = np.random.normal(FULL_LOSS_DURATION_MEAN, FULL_LOSS_DURATION_SD, self.batch_size)
        self.batch_idx = 0
    
    def get_randoms(self):
        """Возвращает следующий набор случайных чисел"""
        if self.batch_idx >= self.batch_size:
            self._generate_new_batch()
        
        result = {
            'uniform': self.current_batch[self.batch_idx],
            'poisson': self.poisson_batch[self.batch_idx],
            'exponential': self.exponential_batch[self.batch_idx],
            'normal': self.normal_batch[self.batch_idx]
        }
        self.batch_idx += 1
        return result


def validate_financial_state(savings, annual_growth, context=""):
    """
    ОБНОВЛЕНО: Проверяет логическую консистентность финансового состояния
    Записывает аномалии в файл для анализа редких багов в production
    Ведет подробную статистику проверок
    """
    # Увеличиваем счетчик проверок
    VALIDATION_STATS['total_checks'] += 1
    
    if not DEBUG_VALIDATION:
        return True
        
    # Проверяем основную аномалию: нет сбережений, но есть накопленный рост
    if savings <= 0 and annual_growth > 0:
        VALIDATION_STATS['total_anomalies'] += 1
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        anomaly_msg = f"[{timestamp}] ФИНАНСОВАЯ АНОМАЛИЯ #{VALIDATION_STATS['total_anomalies']}: {context}"
        anomaly_details = f"  ├── savings: {savings:.6f} ₽"
        anomaly_details += f"\n  ├── annual_growth: {annual_growth:.6f} ₽"
        anomaly_details += f"\n  └── Проблема: накопленный рост при отсутствии сбережений"
        
        full_anomaly = anomaly_msg + "\n" + anomaly_details
        
        # Сохраняем детали для статистики
        VALIDATION_STATS['anomaly_details'].append({
            'timestamp': timestamp,
            'context': context,
            'savings': savings,
            'annual_growth': annual_growth,
            'anomaly_number': VALIDATION_STATS['total_anomalies']
        })
        
        # Вывод в консоль
        print(f"WARNING: ФИНАНСОВАЯ АНОМАЛИЯ #{VALIDATION_STATS['total_anomalies']}: {context}")
        print(f"         savings={savings:.6f}, annual_growth={annual_growth:.6f}")
        
        # Запись в файл аномалий (если путь установлен)
        if config.ANOMALY_LOG_FILE:
            try:
                with open(config.ANOMALY_LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(full_anomaly + "\n\n")
            except Exception as e:
                print(f"Ошибка записи в лог аномалий {config.ANOMALY_LOG_FILE}: {e}")
        
        return False
    return True


def handle_savings_withdrawal(savings, annual_growth, withdrawal_amount):
    """
    ИСПРАВЛЕНО: Корректная обработка изъятия из savings с учетом налогов
    Устранена аномалия с annual_growth > 0 при savings = 0
    
    Args:
        savings: текущая сумма сбережений
        annual_growth: накопленный за год рост (для налогов)
        withdrawal_amount: сумма изъятия
    
    Returns:
        tuple: (new_savings, new_annual_growth, debt_increase)
    """
    if withdrawal_amount <= 0:
        return savings, annual_growth, 0
    
    # ИСПРАВЛЕНИЕ: Если нет сбережений, то не должно быть и накопленного роста
    if savings <= 0:
        # Любое изъятие из пустых сбережений полностью идет в долг
        return 0, 0, withdrawal_amount
    
    if savings >= withdrawal_amount:
        # Частичное изъятие - пропорционально уменьшаем annual_growth
        withdrawal_ratio = withdrawal_amount / savings
        new_annual_growth = annual_growth * (1 - withdrawal_ratio)
        new_savings = savings - withdrawal_amount
        debt_increase = 0
        
        # Дополнительная проверка консистентности
        if new_savings <= 0:
            new_savings = 0
            new_annual_growth = 0
            
    else:
        # Полное изъятие - обнуляем annual_growth, остаток в долг
        debt_increase = withdrawal_amount - savings
        new_savings = 0
        new_annual_growth = 0  # Весь накопленный рост "съеден"
    
    return new_savings, new_annual_growth, debt_increase


def check_plan_changes(month, plan_data):
    """
    НОВАЯ ФУНКЦИЯ: Проверяет и применяет изменения дохода/расходов согласно плану 
    
    Args:
        month: текущий месяц (1-360)
        plan_data: данные плана из PLANS
    
    Returns:
        tuple: (current_income, current_expenses)
    """
    current_income = plan_data['initial_income']
    current_expenses = plan_data['initial_expenses']
    
    # Применяем изменения дохода
    for change in plan_data['income_changes']:
        if month >= change['month']:
            current_income = change['new_income']
    
    # Применяем изменения расходов
    for change in plan_data['expense_changes']:
        if month >= change['month']:
            current_expenses = change['new_expenses']
    
    return current_income, current_expenses


def calculate_mode_with_probabilities(data, n_points=1000):
    """
    Расчет моды и связанных вероятностей с использованием scipy KDE
    
    Args:
        data: массив данных
        n_points: количество точек для оценки плотности
    
    Returns:
        dict: словарь с модой и вероятностями
    """
    data = np.array(data)
    
    # Проверка на вырожденные случаи
    if len(data) == 0:
        return {
            'mode': 0,
            'mode_density': 0,
            'prob_near_mode_5pct': 0,
            'prob_near_mode_10pct': 0,
            'prob_above_mode': 0,
            'prob_below_mode': 0
        }
    
    # Если все значения одинаковые
    if np.std(data) == 0:
        return {
            'mode': data[0],
            'mode_density': 1.0,
            'prob_near_mode_5pct': 100.0,
            'prob_near_mode_10pct': 100.0,
            'prob_above_mode': 0,
            'prob_below_mode': 0
        }
    
    # Удаляем NaN и inf значения
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {
            'mode': 0,
            'mode_density': 0,
            'prob_near_mode_5pct': 0,
            'prob_near_mode_10pct': 0,
            'prob_above_mode': 0,
            'prob_below_mode': 0
        }
    
    try:
        # Создаем KDE используя scipy (профессиональная реализация)
        kde = gaussian_kde(data)
        
        # Создаем сетку точек для оценки плотности
        data_min, data_max = np.min(data), np.max(data)
        data_range = data_max - data_min
        
        if data_range == 0:
            return {
                'mode': data[0],
                'mode_density': 1.0,
                'prob_near_mode_5pct': 100.0,
                'prob_near_mode_10pct': 100.0,
                'prob_above_mode': 0,
                'prob_below_mode': 0
            }
        
        # Расширяем диапазон на 10% с каждой стороны для лучшего покрытия
        range_extension = data_range * 0.1
        x_range = np.linspace(data_min - range_extension, 
                             data_max + range_extension, 
                             n_points)
        
        # Оцениваем плотность используя scipy KDE
        density = kde(x_range)
        
        # Находим моду (максимум плотности)
        mode_idx = np.argmax(density)
        mode_value = x_range[mode_idx]
        mode_density = density[mode_idx]
        
        # Рассчитываем вероятности попадания в окрестность моды
        # Для дискретных данных считаем процент сценариев в диапазоне
        
        # ±5% от модального значения
        if mode_value != 0:
            range_5pct = abs(mode_value) * 0.05
        else:
            range_5pct = data_range * 0.05
        
        near_mode_5pct = np.sum((data >= mode_value - range_5pct) & 
                               (data <= mode_value + range_5pct))
        prob_near_mode_5pct = (near_mode_5pct / len(data)) * 100
        
        # ±10% от модального значения
        if mode_value != 0:
            range_10pct = abs(mode_value) * 0.10
        else:
            range_10pct = data_range * 0.10
            
        near_mode_10pct = np.sum((data >= mode_value - range_10pct) & 
                                (data <= mode_value + range_10pct))
        prob_near_mode_10pct = (near_mode_10pct / len(data)) * 100
        
        # Вероятность выше/ниже моды
        above_mode = np.sum(data > mode_value)
        below_mode = np.sum(data < mode_value)
        prob_above_mode = (above_mode / len(data)) * 100
        prob_below_mode = (below_mode / len(data)) * 100
        
        return {
            'mode': mode_value,
            'mode_density': mode_density,
            'prob_near_mode_5pct': prob_near_mode_5pct,
            'prob_near_mode_10pct': prob_near_mode_10pct,
            'prob_above_mode': prob_above_mode,
            'prob_below_mode': prob_below_mode
        }
        
    except Exception as e:
        print(f"Ошибка при расчете моды через scipy KDE: {e}")
        # Fallback к простой моде через гистограмму
        hist, bin_edges = np.histogram(data, bins=50)
        mode_idx = np.argmax(hist)
        mode_value = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
        
        # Простые вероятности без KDE
        above_mode = np.sum(data > mode_value)
        below_mode = np.sum(data < mode_value)
        prob_above_mode = (above_mode / len(data)) * 100
        prob_below_mode = (below_mode / len(data)) * 100
        
        return {
            'mode': mode_value,
            'mode_density': hist[mode_idx] / len(data),
            'prob_near_mode_5pct': 0,
            'prob_near_mode_10pct': 0,
            'prob_above_mode': prob_above_mode,
            'prob_below_mode': prob_below_mode
        }


def calculate_ideal_scenario(plan_data, months, planned_expenses_enabled=True):
    """
    ИСПРАВЛЕНО: Помесячная симуляция с полной системой долгов и правильной логикой подушки
    """
    # Инициализация стартового капитала с защитой от некорректных значений
    initial_capital = plan_data.get('initial_capital', 0) or 0
    initial_capital = max(0, initial_capital)
    
    cushion = min(CUSHION_AMOUNT, initial_capital)
    savings = max(0, initial_capital - CUSHION_AMOUNT)
    debt = 0
    annual_growth = 0
    
    # События управления долгом
    is_restructured = False
    
    # Запланированные расходы из плана
    plan_expenses = plan_data.get('planned_expenses', [])
    planned_completed = {i: False for i in range(len(plan_expenses))}
    
    for month in range(1, months + 1):
        # Получаем текущий доход и расходы согласно плану
        current_income, current_expenses = check_plan_changes(month, plan_data)
        
        # Начало года - сброс для налога
        if month % 12 == 1:
            annual_growth = 0
        
        # Погашение долга из активов в начале месяца (сначала cushion, потом savings)
        if debt > 0:
            if cushion > 0:
                repayment = min(cushion, debt)
                cushion -= repayment
                debt -= repayment
            elif savings > 0:
                repayment = min(savings, debt)
                # ИСПРАВЛЕНИЕ: Корректируем annual_growth при погашении долга из savings
                savings, annual_growth, _ = handle_savings_withdrawal(savings, annual_growth, repayment)
                debt -= repayment
        
        # Полная система управления долгом и начисление процентов
        if debt > 0:
            annual_income = current_income * 12
            restructuring_threshold = annual_income * RESTRUCTURING_THRESHOLD_RATIO
            bankruptcy_threshold = annual_income * BANKRUPTCY_THRESHOLD_RATIO
            
            # Этап 3: Банкротство (свыше 3 годовых доходов)
            if debt > bankruptcy_threshold:
                debt = 0
                cushion = 0
                savings = 0
                annual_growth = 0  # ИСПРАВЛЕНИЕ: Обнуляем при банкротстве
                is_restructured = False
                
            # Этап 2: Реструктуризация (1-3 годовых дохода)
            elif debt > restructuring_threshold:
                if not is_restructured:
                    is_restructured = True
                
                debt_interest = debt * (DEBT_INTEREST_RATE * 0.5)  # 12% годовых
                debt += debt_interest
                
            # Этап 1: Нормальное кредитование (до 1 годового дохода)
            else:
                if is_restructured:
                    is_restructured = False
                
                debt_interest = debt * DEBT_INTEREST_RATE  # 24% годовых
                debt += debt_interest
        
        # Начисление доходности только на savings (подушка не растет)
        if savings > 0:
            growth = savings * IDEAL_RETURN_RATE
            annual_growth += growth
            savings += growth
        
        available = current_income - current_expenses
        
        # Погашение долга из текущего потока
        if debt > 0 and available > 0:
            repayment = min(available, debt)
            debt -= repayment
            available -= repayment
        
        # Формирование активов - сначала подушка, потом savings
        if available > 0:
            if cushion < CUSHION_AMOUNT:
                cushion_need = min(available, CUSHION_AMOUNT - cushion)
                cushion += cushion_need
                available -= cushion_need
            
            if available > 0:
                savings += available
        elif available < 0:
            # ИСПРАВЛЕНИЕ: Покрытие дефицита с корректировкой annual_growth
            deficit = -available
            if cushion >= deficit:
                cushion -= deficit
            elif cushion > 0:
                deficit -= cushion
                cushion = 0
                # Изъятие из savings с корректировкой annual_growth
                savings, annual_growth, debt_increase = handle_savings_withdrawal(savings, annual_growth, deficit)
                debt += debt_increase
            else:
                # Изъятие только из savings
                savings, annual_growth, debt_increase = handle_savings_withdrawal(savings, annual_growth, deficit)
                debt += debt_increase
        
        # Обработка запланированных расходов из плана (только из savings)
        if planned_expenses_enabled and plan_expenses:
            current_year = (month - 1) // 12 + 1
            
            for i, expense in enumerate(plan_expenses):
                if planned_completed[i]:
                    continue
                    
                should_spend = False
                
                if expense['type'] == 'time' and current_year >= expense['condition'] and savings >= expense['amount']:
                    should_spend = True
                elif expense['type'] == 'savings_target' and savings >= expense['condition']:
                    should_spend = True
                
                if should_spend:
                    # ИСПРАВЛЕНИЕ: Запланированные расходы тоже корректируют annual_growth
                    savings, annual_growth, debt_increase = handle_savings_withdrawal(savings, annual_growth, expense['amount'])
                    debt += debt_increase
                    planned_completed[i] = True
        
        # Погашение долга из активов в конце месяца
        if debt > 0:
            if cushion > 0:
                repayment = min(cushion, debt)
                cushion -= repayment
                debt -= repayment
            elif savings > 0:
                repayment = min(savings, debt)
                # ИСПРАВЛЕНИЕ: Корректируем annual_growth при погашении долга
                savings, annual_growth, _ = handle_savings_withdrawal(savings, annual_growth, repayment)
                debt -= repayment
        
        # Уплата налога (в конце года)
        if month % 12 == 0 and annual_growth > 0:
            tax_payment = annual_growth * TAX_RATE
            # ИСПРАВЛЕНИЕ: Выплата налога тоже корректирует annual_growth
            if savings >= tax_payment:
                savings, annual_growth, _ = handle_savings_withdrawal(savings, annual_growth, tax_payment)
            else:
                savings, annual_growth, debt_increase = handle_savings_withdrawal(savings, annual_growth, tax_payment)
                debt += debt_increase
            
            # Погашение долга из активов после налога
            if debt > 0:
                if cushion > 0:
                    repayment = min(cushion, debt)
                    cushion -= repayment
                    debt -= repayment
                elif savings > 0:
                    repayment = min(savings, debt)
                    savings, annual_growth, _ = handle_savings_withdrawal(savings, annual_growth, repayment)
                    debt -= repayment
    
    # Финальное погашение долга из активов
    if debt > 0:
        if cushion > 0:
            repayment = min(cushion, debt)
            cushion -= repayment
            debt -= repayment
        if debt > 0 and savings > 0:
            repayment = min(savings, debt)
            savings, annual_growth, _ = handle_savings_withdrawal(savings, annual_growth, repayment)
            debt -= repayment
    
    return cushion + savings - debt


def calculate_linear_scenario(plan_data, months, planned_expenses_enabled=True):
    """
    ИСПРАВЛЕНО: Помесячная симуляция с полной системой долгов, но без роста savings
    """
    # Инициализация стартового капитала с защитой от некорректных значений
    initial_capital = plan_data.get('initial_capital', 0) or 0
    initial_capital = max(0, initial_capital)
    
    cushion = min(CUSHION_AMOUNT, initial_capital)
    savings = max(0, initial_capital - CUSHION_AMOUNT)
    debt = 0
    annual_growth = 0  # В линейном сценарии роста нет, но переменная нужна для единообразия
    
    # События управления долгом
    is_restructured = False
    
    # Запланированные расходы из плана
    plan_expenses = plan_data.get('planned_expenses', [])
    planned_completed = {i: False for i in range(len(plan_expenses))}
    
    for month in range(1, months + 1):
        # Получаем текущий доход и расходы согласно плану
        current_income, current_expenses = check_plan_changes(month, plan_data)
        
        # Погашение долга из активов в начале месяца (сначала cushion, потом savings)
        if debt > 0:
            if cushion > 0:
                repayment = min(cushion, debt)
                cushion -= repayment
                debt -= repayment
            elif savings > 0:
                repayment = min(savings, debt)
                savings -= repayment
                debt -= repayment
        
        # Полная система управления долгом и начисление процентов
        if debt > 0:
            annual_income = current_income * 12
            restructuring_threshold = annual_income * RESTRUCTURING_THRESHOLD_RATIO
            bankruptcy_threshold = annual_income * BANKRUPTCY_THRESHOLD_RATIO
            
            # Этап 3: Банкротство (свыше 3 годовых доходов)
            if debt > bankruptcy_threshold:
                debt = 0
                cushion = 0
                savings = 0
                is_restructured = False
                
            # Этап 2: Реструктуризация (1-3 годовых дохода)
            elif debt > restructuring_threshold:
                if not is_restructured:
                    is_restructured = True
                
                debt_interest = debt * (DEBT_INTEREST_RATE * 0.5)  # 12% годовых
                debt += debt_interest
                
            # Этап 1: Нормальное кредитование (до 1 годового дохода)
            else:
                if is_restructured:
                    is_restructured = False
                
                debt_interest = debt * DEBT_INTEREST_RATE  # 24% годовых
                debt += debt_interest
        
        # НЕТ роста savings (0% доходности) - это отличие от идеального сценария
        
        available = current_income - current_expenses
        
        # Погашение долга из текущего потока
        if debt > 0 and available > 0:
            repayment = min(available, debt)
            debt -= repayment
            available -= repayment
        
        # Формирование активов - сначала подушка, потом savings
        if available > 0:
            if cushion < CUSHION_AMOUNT:
                cushion_need = min(available, CUSHION_AMOUNT - cushion)
                cushion += cushion_need
                available -= cushion_need
            
            if available > 0:
                savings += available
        elif available < 0:
            # Покрытие дефицита из активов - сначала cushion, потом savings
            deficit = -available
            if cushion >= deficit:
                cushion -= deficit
            elif cushion > 0:
                deficit -= cushion
                cushion = 0
                if savings >= deficit:
                    savings -= deficit
                else:
                    debt += deficit - savings
                    savings = 0
            else:
                if savings >= deficit:
                    savings -= deficit
                else:
                    debt += deficit - savings
                    savings = 0
        
        # Обработка запланированных расходов из плана (только из savings)
        if planned_expenses_enabled and plan_expenses:
            current_year = (month - 1) // 12 + 1
            
            for i, expense in enumerate(plan_expenses):
                if planned_completed[i]:
                    continue
                    
                should_spend = False
                
                if expense['type'] == 'time' and current_year >= expense['condition'] and savings >= expense['amount']:
                    should_spend = True
                elif expense['type'] == 'savings_target' and savings >= expense['condition']:
                    should_spend = True
                
                if should_spend:
                    if savings >= expense['amount']:
                        savings -= expense['amount']
                        planned_completed[i] = True
                    else:
                        # Если не хватает сбережений, добавляем в долг
                        debt += expense['amount'] - savings
                        savings = 0
                        planned_completed[i] = True
        
        # Погашение долга из активов в конце месяца
        if debt > 0:
            if cushion > 0:
                repayment = min(cushion, debt)
                cushion -= repayment
                debt -= repayment
            elif savings > 0:
                repayment = min(savings, debt)
                savings -= repayment
                debt -= repayment
    
    # Финальное погашение долга из активов
    if debt > 0:
        if cushion > 0:
            repayment = min(cushion, debt)
            cushion -= repayment
            debt -= repayment
        if debt > 0 and savings > 0:
            repayment = min(savings, debt)
            savings -= repayment
            debt -= repayment
    
    return cushion + savings - debt


def run_simulation(plan_id, plan_data):
    """
    ВЕКТОРИЗОВАНО: добавлен батчевый менеджер случайных чисел для ускорения
    ИСПРАВЛЕНО: Налогообложение фантомного роста
    """
    print(f"\nЗапуск модели для Плана {plan_id} (начальный доход {plan_data['initial_income']}₽/мес, стартовый капитал {plan_data.get('initial_capital', 0):,}₽)...")
    start_time = time.time()
    
    # НОВОЕ: Инициализация батчевого менеджера случайных чисел
    batch_manager = RandomBatchManager(batch_size=500)  # Размер батча оптимизирован для веб-версии
    
    # НОВОЕ: Получаем запланированные расходы из плана
    plan_expenses = plan_data.get('planned_expenses', [])
    
    results_by_horizon = {years: {
        'net_wealth': np.zeros(N_SCENARIOS),
        'final_debt': np.zeros(N_SCENARIOS),
        'total_cash_flow': 0,
        'months_zero': np.zeros(N_SCENARIOS),
        'minor_emergencies': np.zeros(N_SCENARIOS),
        'medium_emergencies': np.zeros(N_SCENARIOS),
        'major_emergencies': np.zeros(N_SCENARIOS),
        'shock_pcts': [],  # List для всех shock_pct по месяцам/сценариям (flatten позже)
        'ideal_wealth': 0,  # Новый показатель
        'linear_wealth': 0,  # Новый показатель
        # ДОБАВЛЕНО: отслеживание потери компаундинга
        'scenarios_direct_losses': [],
        'scenarios_compounding_loss': [],
        # НОВОЕ: отслеживание запланированных расходов
        'scenarios_planned_expenses': [],
        'scenarios_planned_compounding_loss': [],
        'planned_expenses_stats': {exp['name']: {'count': 0, 'total_amount': 0} for exp in plan_expenses},
        # НОВОЕ: детальная статистика по долгу
        'max_debt': np.zeros(N_SCENARIOS),  # Максимальный долг за период
        'months_in_debt': np.zeros(N_SCENARIOS),  # Количество месяцев в долгу
        'total_interest_paid': np.zeros(N_SCENARIOS),  # Общая сумма процентов
        'avg_debt_when_in_debt': np.zeros(N_SCENARIOS),  # Средний размер долга (когда он был)
        # НОВОЕ: события управления долгом
        'restructuring_events': np.zeros(N_SCENARIOS),  # Количество реструктуризаций
        'bankruptcy_events': np.zeros(N_SCENARIOS),     # Количество банкротств
        'months_in_restructuring': np.zeros(N_SCENARIOS),  # Месяцев под реструктуризацией
    } for years in HORIZONS}
    
    # Расчет идеальных и линейных сценариев для всех горизонтов
    for years in HORIZONS:
        months = years * 12
        results_by_horizon[years]['ideal_wealth'] = calculate_ideal_scenario(plan_data, months, True)
        results_by_horizon[years]['linear_wealth'] = calculate_linear_scenario(plan_data, months, True)
    
    for scenario in range(N_SCENARIOS):
        # Инициализация стартового капитала с защитой от некорректных значений
        initial_capital = plan_data.get('initial_capital', 0) or 0
        initial_capital = max(0, initial_capital)
        
        cushion = min(CUSHION_AMOUNT, initial_capital)
        savings = max(0, initial_capital - CUSHION_AMOUNT)
        debt = 0
        minor_cluster_active = False
        major_cluster_remaining = 0
        active_partial_loss = 0
        active_full_loss = 0
        scenario_cash_flow = 0
        annual_growth = 0
        start_of_year_savings = 0
        
        minor_em_count = 0
        medium_em_count = 0
        major_em_count = 0
        
        horizon_counters = {years: {'zero': 0} for years in HORIZONS}
        
        shock_pcts_scenario = []
        
        # Отслеживание шоков с временными метками
        shock_history = []
        
        # Отслеживание запланированных расходов
        planned_expenses_history = []
        planned_completed = {i: False for i in range(len(plan_expenses))}
        
        # Отслеживание статистики по долгу
        debt_history = []
        total_interest_paid = 0
        
        # События управления долгом
        restructuring_count = 0
        bankruptcy_count = 0
        months_restructuring = 0
        is_restructured = False
        
        # НОВОЕ: Виртуальный сценарий для правильного расчета потерь компаундинга
        virtual_cushion = cushion
        virtual_savings = savings
        virtual_debt = 0
        virtual_annual_growth = 0
        virtual_is_restructured = False
        
        for month in range(1, N_MONTHS + 1):
            # Получаем текущий доход и расходы согласно плану
            current_income, current_expenses = check_plan_changes(month, plan_data)
            
            # Динамический буфер для процентных расчетов
            target_savings = current_income - current_expenses
            
            # Начало года - сброс для налога
            if month % 12 == 1:
                start_of_year_savings = savings
                annual_growth = 0
                virtual_annual_growth = 0
            
            # РЕАЛЬНЫЙ СЦЕНАРИЙ
            
            # Погашение долга из активов в начале месяца (сначала cushion, потом savings)
            if debt > 0:
                if cushion > 0:
                    repayment = min(cushion, debt)
                    cushion -= repayment
                    debt -= repayment
                elif savings > 0:
                    repayment = min(savings, debt)
                    # ИСПРАВЛЕНИЕ: Корректируем annual_growth при погашении долга
                    savings, annual_growth, _ = handle_savings_withdrawal(savings, annual_growth, repayment)
                    debt -= repayment
            
            # Поэтапное управление долгом и начисление процентов
            if debt > 0:
                annual_income = current_income * 12
                restructuring_threshold = annual_income * RESTRUCTURING_THRESHOLD_RATIO
                bankruptcy_threshold = annual_income * BANKRUPTCY_THRESHOLD_RATIO
                
                # Этап 3: Банкротство (свыше 3 годовых доходов)
                if debt > bankruptcy_threshold:
                    bankruptcy_count += 1
                    debt = 0
                    cushion = 0
                    savings = 0
                    annual_growth = 0  # ИСПРАВЛЕНИЕ: Обнуляем при банкротстве
                    is_restructured = False
                    
                # Этап 2: Реструктуризация (1-3 годовых дохода)
                elif debt > restructuring_threshold:
                    if not is_restructured:
                        restructuring_count += 1
                        is_restructured = True
                    
                    months_restructuring += 1
                    debt_interest = debt * (DEBT_INTEREST_RATE * 0.5)  # 12% годовых (1% в месяц)
                    debt += debt_interest
                    total_interest_paid += debt_interest
                    
                # Этап 1: Нормальное кредитование (до 1 годового дохода)
                else:
                    if is_restructured:
                        is_restructured = False  # Выход из реструктуризации
                    
                    debt_interest = debt * DEBT_INTEREST_RATE  # 24% годовых (2% в месяц)
                    debt += debt_interest
                    total_interest_paid += debt_interest
            
            # Начисление доходности только на savings (подушка не растет)
            if savings > 0:
                growth = savings * SAVINGS_RETURN_RATE
                annual_growth += growth
                savings += growth
                # ВАЛИДАЦИЯ: Проверяем состояние после начисления роста
                if DEBUG_VALIDATION:
                    validate_financial_state(savings, annual_growth, f"Plan {plan_id}, scenario {scenario}, month {month} - after growth")
            
            available = current_income - current_expenses
            emergency_cost = 0
            minor_em_occurred = False
            medium_em_occurred = False
            major_em_occurred = False
            
            # ВЕКТОРИЗОВАННАЯ ГЕНЕРАЦИЯ ЧП: Получаем батч случайных чисел
            randoms = batch_manager.get_randoms()
            r_uniform = randoms['uniform']
            
            # Обработка крупных ЧП с кластеризацией Пуассона
            if major_cluster_remaining > 0:
                emergency_cost += MAJOR_EMERGENCY_COST
                shock_history.append((month, MAJOR_EMERGENCY_COST, 'major_emergency'))
                major_em_count += 1
                major_em_occurred = True
                major_cluster_remaining -= 1
            
            # ВЕКТОРИЗОВАННАЯ ГЕНЕРАЦИЯ: Генерация новых ЧП (используем предгенерированные числа)
            # Мелкие ЧП
            if not minor_cluster_active and r_uniform[batch_manager.MINOR_EM_IDX] < MINOR_EMERGENCY_PROB:
                emergency_cost += MINOR_EMERGENCY_COST
                shock_history.append((month, MINOR_EMERGENCY_COST, 'minor_emergency'))
                minor_em_count += 1
                minor_em_occurred = True
                minor_cluster_active = True
            
            # Средние ЧП
            if not minor_cluster_active and r_uniform[batch_manager.MEDIUM_EM_IDX] < MEDIUM_EMERGENCY_PROB:
                emergency_cost += MEDIUM_EMERGENCY_COST
                shock_history.append((month, MEDIUM_EMERGENCY_COST, 'medium_emergency'))
                medium_em_count += 1
                medium_em_occurred = True
                minor_cluster_active = True
            
            # Крупные ЧП (только если нет активного кластера)
            if major_cluster_remaining == 0 and r_uniform[batch_manager.MAJOR_EM_IDX] < MAJOR_EMERGENCY_PROB:
                emergency_cost += MAJOR_EMERGENCY_COST
                shock_history.append((month, MAJOR_EMERGENCY_COST, 'major_emergency'))
                major_em_count += 1
                major_em_occurred = True
                # ВЕКТОРИЗОВАННАЯ ГЕНЕРАЦИЯ: Запуск кластера Пуассона для крупных ЧП
                major_cluster_remaining = randoms['poisson']
            
            # Обработка кластера для мелких/средних ЧП
            if minor_cluster_active:
                # ВЕКТОРИЗОВАННАЯ ГЕНЕРАЦИЯ: С вероятностью 38% генерируем дополнительное ЧП в кластере
                if r_uniform[batch_manager.CLUSTER_CONTINUE_IDX] < MINOR_CLUSTER_PROB:
                    cluster_type = r_uniform[batch_manager.CLUSTER_TYPE_IDX]
                    if cluster_type < 0.651:  # Мелкие (65.1%)
                        emergency_cost += MINOR_EMERGENCY_COST
                        shock_history.append((month, MINOR_EMERGENCY_COST, 'minor_cluster'))
                        minor_em_count += 1
                        minor_em_occurred = True
                    else:  # Средние (34.9%)
                        emergency_cost += MEDIUM_EMERGENCY_COST
                        shock_history.append((month, MEDIUM_EMERGENCY_COST, 'medium_cluster'))
                        medium_em_count += 1
                        medium_em_occurred = True
                else:
                    minor_cluster_active = False
            
            available -= emergency_cost
            
            # ВЕКТОРИЗОВАННАЯ ГЕНЕРАЦИЯ: Генерация потерь дохода
            loss = 0
            if active_partial_loss == 0 and r_uniform[batch_manager.PARTIAL_LOSS_IDX] < PARTIAL_LOSS_PROB:
                active_partial_loss = max(1, int(randoms['exponential']))
            
            if active_full_loss == 0 and r_uniform[batch_manager.FULL_LOSS_IDX] < FULL_LOSS_PROB:
                duration = randoms['normal']
                active_full_loss = max(1, int(np.round(duration)))
            
            if active_partial_loss > 0:
                loss += current_income * PARTIAL_LOSS_RATE
                shock_history.append((month, current_income * PARTIAL_LOSS_RATE, 'partial_income_loss'))
                active_partial_loss -= 1
            
            if active_full_loss > 0:
                loss += current_income
                shock_history.append((month, current_income, 'full_income_loss'))
                active_full_loss -= 1
            
            available -= loss
            
            # Обработка запланированных расходов из плана (только из savings, не из подушки)
            if plan_expenses:
                current_year = (month - 1) // 12 + 1
                
                for i, expense in enumerate(plan_expenses):
                    if planned_completed[i]:
                        continue
                        
                    should_spend = False
                    
                    if expense['type'] == 'time' and current_year >= expense['condition'] and savings >= expense['amount']:
                        should_spend = True
                    elif expense['type'] == 'savings_target' and savings >= expense['condition']:
                        should_spend = True
                    
                    if should_spend:
                        # ИСПРАВЛЕНИЕ: Запланированные расходы корректируют annual_growth
                        savings, annual_growth, debt_increase = handle_savings_withdrawal(savings, annual_growth, expense['amount'])
                        debt += debt_increase
                        planned_expenses_history.append((month, expense['amount'], expense['name']))
                        planned_completed[i] = True
            
            # Новая метрика: Если шок >0, рассчитываем %
            shock_total = emergency_cost + loss
            if shock_total > 0 and target_savings > 0:
                shock_pct = (shock_total / target_savings) * 100
                shock_pcts_scenario.append(shock_pct)
            
            # Погашение долга из текущего потока
            if debt > 0 and available > 0:
                repayment = min(available, debt)
                debt -= repayment
                available -= repayment
            
            # Формирование активов - сначала подушка, потом savings
            if available > 0:
                if cushion < CUSHION_AMOUNT:
                    cushion_need = min(available, CUSHION_AMOUNT - cushion)
                    cushion += cushion_need
                    available -= cushion_need
                    contribution_type = 'positive' if available > 0 else ('positive' if cushion_need > 0 else 'zero')
                else:
                    contribution_type = 'positive'
                
                if available > 0:
                    savings += available
            elif available < 0:
                # ИСПРАВЛЕНИЕ: Покрытие дефицита с корректировкой annual_growth
                deficit = -available
                if cushion >= deficit:
                    cushion -= deficit
                elif cushion > 0:
                    deficit -= cushion
                    cushion = 0
                    # Изъятие из savings с корректировкой annual_growth
                    savings, annual_growth, debt_increase = handle_savings_withdrawal(savings, annual_growth, deficit)
                    debt += debt_increase
                else:
                    # Изъятие только из savings
                    savings, annual_growth, debt_increase = handle_savings_withdrawal(savings, annual_growth, deficit)
                    debt += debt_increase
                contribution_type = 'zero'
            else:
                contribution_type = 'zero'
            
            # Погашение долга из активов в конце месяца (сначала cushion, потом savings)
            if debt > 0:
                if cushion > 0:
                    repayment = min(cushion, debt)
                    cushion -= repayment
                    debt -= repayment
                elif savings > 0:
                    repayment = min(savings, debt)
                    # ИСПРАВЛЕНИЕ: Корректируем annual_growth при погашении долга
                    savings, annual_growth, _ = handle_savings_withdrawal(savings, annual_growth, repayment)
                    debt -= repayment
            
            # Денежный поток
            cash_flow = current_income - current_expenses - loss - emergency_cost
            scenario_cash_flow += cash_flow
            
            # Уплата налога (в конце года)
            if month % 12 == 0 and annual_growth > 0:
                tax_payment = annual_growth * TAX_RATE
                # ВАЛИДАЦИЯ: Проверяем состояние перед выплатой налога
                if DEBUG_VALIDATION:
                    validate_financial_state(savings, annual_growth, f"Plan {plan_id}, scenario {scenario}, month {month} - before tax")
                
                # ИСПРАВЛЕНИЕ: Выплата налога корректирует annual_growth
                if savings >= tax_payment:
                    savings, annual_growth, _ = handle_savings_withdrawal(savings, annual_growth, tax_payment)
                else:
                    savings, annual_growth, debt_increase = handle_savings_withdrawal(savings, annual_growth, tax_payment)
                    debt += debt_increase
                
                # ВАЛИДАЦИЯ: Проверяем состояние после выплаты налога
                if DEBUG_VALIDATION:
                    validate_financial_state(savings, annual_growth, f"Plan {plan_id}, scenario {scenario}, month {month} - after tax")
                
                # Погашение долга из активов после налога (сначала cushion, потом savings)
                if debt > 0:
                    if cushion > 0:
                        repayment = min(cushion, debt)
                        cushion -= repayment
                        debt -= repayment
                    elif savings > 0:
                        repayment = min(savings, debt)
                        # ИСПРАВЛЕНИЕ: Корректируем annual_growth при погашении долга
                        savings, annual_growth, _ = handle_savings_withdrawal(savings, annual_growth, repayment)
                        debt -= repayment
            
            # ВИРТУАЛЬНЫЙ СЦЕНАРИЙ (параллельно)
            
            # Погашение долга из активов в начале месяца
            if virtual_debt > 0:
                if virtual_cushion > 0:
                    virtual_repayment = min(virtual_cushion, virtual_debt)
                    virtual_cushion -= virtual_repayment
                    virtual_debt -= virtual_repayment
                elif virtual_savings > 0:
                    virtual_repayment = min(virtual_savings, virtual_debt)
                    # ИСПРАВЛЕНИЕ: Виртуальный сценарий тоже использует корректировку
                    virtual_savings, virtual_annual_growth, _ = handle_savings_withdrawal(virtual_savings, virtual_annual_growth, virtual_repayment)
                    virtual_debt -= virtual_repayment
            
            # Полная система управления долгом (как в реальном)
            if virtual_debt > 0:
                annual_income = current_income * 12
                restructuring_threshold = annual_income * RESTRUCTURING_THRESHOLD_RATIO
                bankruptcy_threshold = annual_income * BANKRUPTCY_THRESHOLD_RATIO
                
                if virtual_debt > bankruptcy_threshold:
                    virtual_debt = 0
                    virtual_cushion = 0
                    virtual_savings = 0
                    virtual_annual_growth = 0  # ИСПРАВЛЕНИЕ
                    virtual_is_restructured = False
                elif virtual_debt > restructuring_threshold:
                    if not virtual_is_restructured:
                        virtual_is_restructured = True
                    virtual_debt_interest = virtual_debt * (DEBT_INTEREST_RATE * 0.5)
                    virtual_debt += virtual_debt_interest
                else:
                    if virtual_is_restructured:
                        virtual_is_restructured = False
                    virtual_debt_interest = virtual_debt * DEBT_INTEREST_RATE
                    virtual_debt += virtual_debt_interest
            
            # Рост по идеальной доходности (только savings)
            if virtual_savings > 0:
                virtual_growth = virtual_savings * IDEAL_RETURN_RATE
                virtual_annual_growth += virtual_growth
                virtual_savings += virtual_growth
            
            # Те же денежные потоки (без шоков)
            virtual_available = current_income - current_expenses
            
            # Погашение долга из текущего потока
            if virtual_debt > 0 and virtual_available > 0:
                virtual_repayment = min(virtual_available, virtual_debt)
                virtual_debt -= virtual_repayment
                virtual_available -= virtual_repayment
            
            # Формирование активов
            if virtual_available > 0:
                if virtual_cushion < CUSHION_AMOUNT:
                    cushion_need = min(virtual_available, CUSHION_AMOUNT - virtual_cushion)
                    virtual_cushion += cushion_need
                    virtual_available -= cushion_need
                
                if virtual_available > 0:
                    virtual_savings += virtual_available
            elif virtual_available < 0:
                # ИСПРАВЛЕНИЕ: Виртуальный сценарий тоже корректирует annual_growth
                deficit = -virtual_available
                if virtual_cushion >= deficit:
                    virtual_cushion -= deficit
                elif virtual_cushion > 0:
                    deficit -= virtual_cushion
                    virtual_cushion = 0
                    virtual_savings, virtual_annual_growth, virtual_debt_increase = handle_savings_withdrawal(virtual_savings, virtual_annual_growth, deficit)
                    virtual_debt += virtual_debt_increase
                else:
                    virtual_savings, virtual_annual_growth, virtual_debt_increase = handle_savings_withdrawal(virtual_savings, virtual_annual_growth, deficit)
                    virtual_debt += virtual_debt_increase
            
            # КЛЮЧЕВОЕ: Синхронизированные траты из реального сценария
            for planned_month, planned_amount, planned_name in planned_expenses_history:
                if planned_month == month:
                    # ИСПРАВЛЕНИЕ: Виртуальный сценарий тоже корректирует annual_growth
                    virtual_savings, virtual_annual_growth, virtual_debt_increase = handle_savings_withdrawal(virtual_savings, virtual_annual_growth, planned_amount)
                    virtual_debt += virtual_debt_increase
            
            # Погашение долга в конце месяца
            if virtual_debt > 0:
                if virtual_cushion > 0:
                    virtual_repayment = min(virtual_cushion, virtual_debt)
                    virtual_cushion -= virtual_repayment
                    virtual_debt -= virtual_repayment
                elif virtual_savings > 0:
                    virtual_repayment = min(virtual_savings, virtual_debt)
                    # ИСПРАВЛЕНИЕ: Корректируем annual_growth при погашении долга
                    virtual_savings, virtual_annual_growth, _ = handle_savings_withdrawal(virtual_savings, virtual_annual_growth, virtual_repayment)
                    virtual_debt -= virtual_repayment
            
            # Уплата налога (в конце года)
            if month % 12 == 0 and virtual_annual_growth > 0:
                virtual_tax_payment = virtual_annual_growth * TAX_RATE
                # ВАЛИДАЦИЯ: Проверяем виртуальное состояние перед выплатой налога
                if DEBUG_VALIDATION:
                    validate_financial_state(virtual_savings, virtual_annual_growth, f"Plan {plan_id}, scenario {scenario}, month {month} - virtual before tax")
                
                # ИСПРАВЛЕНИЕ: Виртуальный сценарий тоже корректирует annual_growth при выплате налога
                if virtual_savings >= virtual_tax_payment:
                    virtual_savings, virtual_annual_growth, _ = handle_savings_withdrawal(virtual_savings, virtual_annual_growth, virtual_tax_payment)
                else:
                    virtual_savings, virtual_annual_growth, virtual_debt_increase = handle_savings_withdrawal(virtual_savings, virtual_annual_growth, virtual_tax_payment)
                    virtual_debt += virtual_debt_increase
                
                # ВАЛИДАЦИЯ: Проверяем виртуальное состояние после выплаты налога
                if DEBUG_VALIDATION:
                    validate_financial_state(virtual_savings, virtual_annual_growth, f"Plan {plan_id}, scenario {scenario}, month {month} - virtual after tax")
                
                # Погашение долга после налога
                if virtual_debt > 0:
                    if virtual_cushion > 0:
                        virtual_repayment = min(virtual_cushion, virtual_debt)
                        virtual_cushion -= virtual_repayment
                        virtual_debt -= virtual_repayment
                    elif virtual_savings > 0:
                        virtual_repayment = min(virtual_savings, virtual_debt)
                        virtual_savings, virtual_annual_growth, _ = handle_savings_withdrawal(virtual_savings, virtual_annual_growth, virtual_repayment)
                        virtual_debt -= virtual_repayment
            
            # Обновление счетчиков
            for years in HORIZONS:
                if month <= years * 12:
                    if contribution_type == 'zero':
                        horizon_counters[years]['zero'] += 1
            
            # Сохранение истории долга для каждого горизонта
            debt_history.append(debt)
            
            # Фиксация результатов
            for years in HORIZONS:
                if month == years * 12:
                    # Финальное погашение долга из активов в конце периода (сначала cushion, потом savings)
                    if debt > 0:
                        if cushion > 0:
                            repayment = min(cushion, debt)
                            cushion -= repayment
                            debt -= repayment
                        if debt > 0 and savings > 0:
                            repayment = min(savings, debt)
                            # ИСПРАВЛЕНИЕ: Финальное погашение тоже корректирует annual_growth
                            savings, annual_growth, _ = handle_savings_withdrawal(savings, annual_growth, repayment)
                            debt -= repayment
                    
                    # Финальное погашение долга для виртуального сценария
                    if virtual_debt > 0:
                        if virtual_cushion > 0:
                            virtual_repayment = min(virtual_cushion, virtual_debt)
                            virtual_cushion -= virtual_repayment
                            virtual_debt -= virtual_repayment
                        if virtual_debt > 0 and virtual_savings > 0:
                            virtual_repayment = min(virtual_savings, virtual_debt)
                            virtual_savings, virtual_annual_growth, _ = handle_savings_withdrawal(virtual_savings, virtual_annual_growth, virtual_repayment)
                            virtual_debt -= virtual_repayment
                    
                    horizon_data = results_by_horizon[years]
                    # Общие активы = подушка + сбережения
                    total_wealth = cushion + savings
                    horizon_data['net_wealth'][scenario] = total_wealth - debt
                    horizon_data['final_debt'][scenario] = debt
                    horizon_data['total_cash_flow'] += scenario_cash_flow
                    horizon_data['minor_emergencies'][scenario] = minor_em_count
                    horizon_data['medium_emergencies'][scenario] = medium_em_count
                    horizon_data['major_emergencies'][scenario] = major_em_count
                    horizon_data['months_zero'][scenario] = horizon_counters[years]['zero']
                    # Добавляем shock_pcts_scenario в общий list (копируем, чтобы не мутировать)
                    horizon_data['shock_pcts'].extend(shock_pcts_scenario[:years*12])  # Только до horizon месяцев
                    
                    # Расчет прямых потерь для каждого горизонта
                    horizon_months = years * 12
                    horizon_direct_losses = sum(amount for month_shock, amount, shock_type in shock_history 
                                               if month_shock <= horizon_months) / 1000000  # в млн
                    horizon_data['scenarios_direct_losses'].append(horizon_direct_losses)
                    
                    # Расчет запланированных расходов для каждого горизонта
                    horizon_planned_expenses = sum(amount for month_exp, amount, name in planned_expenses_history
                                                  if month_exp <= horizon_months) / 1000000  # в млн
                    horizon_data['scenarios_planned_expenses'].append(horizon_planned_expenses)
                    
                    # НОВОЕ: Правильный расчет потерь компаундинга через виртуальный сценарий
                    virtual_total_wealth = virtual_cushion + virtual_savings
                    virtual_net_wealth = virtual_total_wealth - virtual_debt
                    real_net_wealth = total_wealth - debt
                    
                    # Потеря компаундинга = (виртуальные активы - реальные активы) - прямые потери
                    compounding_loss = (virtual_net_wealth - real_net_wealth) / 1000000 - horizon_direct_losses
                    compounding_loss = max(0, compounding_loss)  # не может быть отрицательной
                    horizon_data['scenarios_compounding_loss'].append(compounding_loss)
                    
                    # ИСПРАВЛЕНО: правильный расчет потери компаундинга от запланированных расходов
                    planned_compounding_loss = 0
                    for month_exp, amount, name in planned_expenses_history:
                        if month_exp <= horizon_months:
                            # Сколько месяцев осталось от момента покупки до конца горизонта
                            remaining_months = horizon_months - month_exp
                            if remaining_months > 0:
                                # Потеря компаундинга = сколько бы выросли эти деньги за оставшееся время
                                compounding_growth = amount * ((1 + SAVINGS_RETURN_RATE) ** remaining_months - 1)
                                planned_compounding_loss += compounding_growth
                    
                    horizon_data['scenarios_planned_compounding_loss'].append(planned_compounding_loss / 1000000)  # в млн
                    
                    # Статистика по типам запланированных расходов
                    for month_exp, amount, name in planned_expenses_history:
                        if month_exp <= horizon_months:
                            if name in horizon_data['planned_expenses_stats']:
                                horizon_data['planned_expenses_stats'][name]['count'] += 1
                                horizon_data['planned_expenses_stats'][name]['total_amount'] += amount
                    
                    # Расчет статистики по долгу для данного горизонта
                    horizon_debt_history = debt_history[:horizon_months]
                    
                    # Максимальный долг за период
                    horizon_data['max_debt'][scenario] = max(horizon_debt_history) if horizon_debt_history else 0
                    
                    # Количество месяцев в долгу
                    months_with_debt = sum(1 for debt_amount in horizon_debt_history if debt_amount > 0)
                    horizon_data['months_in_debt'][scenario] = months_with_debt
                    
                    # Общая сумма процентов за период (пропорционально)
                    if N_MONTHS > 0:
                        horizon_interest_paid = total_interest_paid * (horizon_months / N_MONTHS)
                    else:
                        horizon_interest_paid = 0
                    horizon_data['total_interest_paid'][scenario] = horizon_interest_paid
                    
                    # Средний размер долга (когда он был)
                    if months_with_debt > 0:
                        debt_sum = sum(debt_amount for debt_amount in horizon_debt_history if debt_amount > 0)
                        horizon_data['avg_debt_when_in_debt'][scenario] = debt_sum / months_with_debt
                    else:
                        horizon_data['avg_debt_when_in_debt'][scenario] = 0
                    
                    # События управления долгом
                    horizon_data['restructuring_events'][scenario] = restructuring_count
                    horizon_data['bankruptcy_events'][scenario] = bankruptcy_count
                    
                    # Месяцев в реструктуризации (пропорционально для горизонта)
                    if N_MONTHS > 0:
                        horizon_months_restructuring = months_restructuring * (horizon_months / N_MONTHS)
                    else:
                        horizon_months_restructuring = 0
                    horizon_data['months_in_restructuring'][scenario] = horizon_months_restructuring
        
        if (scenario + 1) % 200 == 0:  # Прогресс каждые 200 сценариев для 1000 всего
            elapsed = time.time() - start_time
            progress = (scenario + 1) / N_SCENARIOS * 100
            print(f"  {scenario+1}/{N_SCENARIOS} ({progress:.0f}%) - {elapsed:.1f} сек")
    
    # Расчет итоговых показателей
    print(f"  Расчет статистик и моды через scipy KDE...")
    for years, horizon_data in results_by_horizon.items():
        months = years * 12
        net_wealth = horizon_data['net_wealth']
        final_debt = horizon_data['final_debt']
        
        horizon_data['avg_wealth'] = np.mean(net_wealth)
        horizon_data['median_wealth'] = np.median(net_wealth)
        
        # Расчет модальных значений и вероятностей
        modal_data = calculate_mode_with_probabilities(net_wealth)
        horizon_data['modal_wealth'] = modal_data['mode']
        horizon_data['modal_density'] = modal_data['mode_density']
        horizon_data['prob_near_mode_5pct'] = modal_data['prob_near_mode_5pct']
        horizon_data['prob_near_mode_10pct'] = modal_data['prob_near_mode_10pct']
        horizon_data['prob_above_mode'] = modal_data['prob_above_mode']
        horizon_data['prob_below_mode'] = modal_data['prob_below_mode']
        horizon_data['p10_wealth'] = np.percentile(net_wealth, 10)
        horizon_data['p1_wealth'] = np.percentile(net_wealth, 1)  # ДОБАВЛЕН 1-й ПЕРЦЕНТИЛЬ
        horizon_data['min_wealth'] = np.min(net_wealth)
        horizon_data['max_wealth'] = np.max(net_wealth)
        
        # Процент месяцев
        horizon_data['pct_zero'] = np.mean(horizon_data['months_zero']) / months * 100
        
        # Долги
        debt_mask = final_debt > 0
        horizon_data['pct_in_debt'] = np.sum(debt_mask) / N_SCENARIOS * 100
        horizon_data['avg_debt'] = np.mean(final_debt[debt_mask]) if np.any(debt_mask) else 0
        
        # Денежный поток
        horizon_data['real_avg_cash_flow'] = horizon_data['total_cash_flow'] / (N_SCENARIOS * months)
        
        # Среднее количество ЧП
        horizon_data['avg_minor_em'] = np.mean(horizon_data['minor_emergencies'])
        horizon_data['avg_medium_em'] = np.mean(horizon_data['medium_emergencies'])
        horizon_data['avg_major_em'] = np.mean(horizon_data['major_emergencies'])
        
        # Шоки %: Если есть данные
        if horizon_data['shock_pcts']:
            shock_array = np.array(horizon_data['shock_pcts'])
            horizon_data['median_shock_pct'] = np.median(shock_array)
            horizon_data['p90_shock_pct'] = np.percentile(shock_array, 90)
            horizon_data['p95_shock_pct'] = np.percentile(shock_array, 95)
        else:
            horizon_data['median_shock_pct'] = 0
            horizon_data['p90_shock_pct'] = 0
            horizon_data['p95_shock_pct'] = 0
        
        # Расчет вклада стартового капитала
        horizon_data['avg_direct_losses'] = np.mean(horizon_data['scenarios_direct_losses'])
        horizon_data['avg_compounding_loss'] = np.mean(horizon_data['scenarios_compounding_loss'])
        horizon_data['compounding_vs_direct_ratio'] = (
            horizon_data['avg_compounding_loss'] / horizon_data['avg_direct_losses'] 
            if horizon_data['avg_direct_losses'] > 0 else 0
        )
        
        # Расчет вклада стартового капитала
        initial_capital = plan_data.get('initial_capital', 0) or 0
        if initial_capital > 0:
            # Потенциальная стоимость стартового капитала при идеальной доходности
            ideal_growth_factor = (1 + IDEAL_RETURN_RATE) ** months
            horizon_data['initial_capital_potential'] = initial_capital * ideal_growth_factor
            horizon_data['initial_capital_profit'] = horizon_data['initial_capital_potential'] - initial_capital
        else:
            horizon_data['initial_capital_potential'] = 0
            horizon_data['initial_capital_profit'] = 0
        
        # Статистика запланированных расходов
        horizon_data['avg_planned_expenses'] = np.mean(horizon_data['scenarios_planned_expenses'])
        horizon_data['avg_planned_compounding_loss'] = np.mean(horizon_data['scenarios_planned_compounding_loss'])
        
        # Финализация статистики по типам запланированных расходов
        for name, stats in horizon_data['planned_expenses_stats'].items():
            if stats['count'] > 0:
                stats['avg_amount'] = stats['total_amount'] / stats['count']
                stats['frequency'] = stats['count'] / N_SCENARIOS * 100
            else:
                stats['avg_amount'] = 0
                stats['frequency'] = 0
    
    print(f"  Завершено за {time.time() - start_time:.1f} сек")
    return results_by_horizon