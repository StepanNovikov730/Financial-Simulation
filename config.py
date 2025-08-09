import random
import numpy as np

# ===== ВОСПРОИЗВОДИМОСТЬ =====
RANDOM_SEED = 42  # Можно изменить на любое число

def set_random_seeds():
    """Устанавливает seeds для воспроизводимости результатов"""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

# ===== ПАРАМЕТРЫ СИМУЛЯЦИИ =====
N_SCENARIOS = 1000  # ВЕКТОРИЗОВАНО: Уменьшено до 1000 для веб-версии (было 10000)
N_MONTHS = 360

# ===== ФИНАНСОВЫЕ ПАРАМЕТРЫ =====
SAVINGS_RETURN_RATE = 0.005
TAX_RATE = 0.13

# НОВОЕ: Подушка безопасности
CUSHION_AMOUNT = 200000  # Размер подушки безопасности

# НОВОЕ: Процентная ставка на долг и пороги
DEBT_INTEREST_RATE = 0.24 / 12  # 24% годовых = 2% в месяц
RESTRUCTURING_THRESHOLD_RATIO = 1.0  # 1 годовой доход
BANKRUPTCY_THRESHOLD_RATIO = 3.0     # 3 годовых дохода

# Параметры для идеального сценария
IDEAL_RETURN_RATE = 0.06 / 12  # 6% годовых = 0.5% в месяц

# ===== ПАРАМЕТРЫ ЧП =====
# Параметры ЧП
MINOR_EMERGENCY_PROB = 0.1
MINOR_EMERGENCY_COST = 9000
MEDIUM_EMERGENCY_PROB = 0.024
MEDIUM_EMERGENCY_COST = 65000
MAJOR_EMERGENCY_PROB = 0.006
MAJOR_EMERGENCY_COST = 300000

# Кластеризация
MINOR_CLUSTER_PROB = 0.38
MAJOR_CLUSTER_LAMBDA = 0.3

# Потеря дохода
PARTIAL_LOSS_PROB = 0.008
PARTIAL_LOSS_RATE = 0.4
PARTIAL_LOSS_DURATION = 2
FULL_LOSS_PROB = 0.0045
FULL_LOSS_DURATION_MEAN = 5
FULL_LOSS_DURATION_SD = 1.5

# ===== ГОРИЗОНТЫ АНАЛИЗА =====
HORIZONS = [5, 10, 15, 20, 25, 30]

# ===== ПЛАНЫ (ТРАЕКТОРИИ) =====
"""
Структура плана:
- initial_income: начальный доход в месяц
- initial_expenses: начальные расходы в месяц  
- initial_capital: стартовый капитал (сначала заполняет подушку, остаток в сбережения)
- income_changes: список изменений дохода [{month, new_income}, ...]
- expense_changes: список изменений расходов [{month, new_expenses}, ...]
- planned_expenses: список запланированных расходов
  - name: название расхода
  - amount: сумма в рублях
  - type: 'time' или 'savings_target'
  - condition: 
    * для 'time': год (начиная с которого возможен расход, если достаточно денег)
    * для 'savings_target': целевая сумма накоплений для покупки
  - repeat: повторяется ли расход (пока не используется)
"""
PLANS = {
    'A': {
        'initial_income': 110000,
        'initial_expenses': 80000,
        'initial_capital': 0,  # Стартовый капитал
        'income_changes': [
            # Пример: {'month': 24, 'new_income': 130000},
        ],
        'expense_changes': [
            # Пример: {'month': 36, 'new_expenses': 85000},
        ],
        'planned_expenses': [
            {'name': 'Электроскутер', 'amount': 400000, 'type': 'savings_target', 'condition': 400000, 'repeat': False},
            # {'name': 'Первый ремонт', 'amount': 500000, 'type': 'time', 'condition': 2, 'repeat': False},
        ]
    },
    'B': {
        'initial_income': 150000,
        'initial_expenses': 80000,
        'initial_capital': 0,  # Стартовый капитал
        'income_changes': [],
        'expense_changes': [],
        'planned_expenses': [
            {'name': 'Электроскутер', 'amount': 400000, 'type': 'savings_target', 'condition': 400000, 'repeat': False},
            # {'name': 'Покупка автомобиля', 'amount': 1200000, 'type': 'time', 'condition': 3, 'repeat': False},
            # {'name': 'Отпуск мечты', 'amount': 400000, 'type': 'savings_target', 'condition': 600000, 'repeat': False},
        ]
    },
    'C': {
        'initial_income': 200000,
        'initial_expenses': 80000,
        'initial_capital': 0,  # Стартовый капитал
        'income_changes': [],
        'expense_changes': [],
        'planned_expenses': [
            {'name': 'Электроскутер', 'amount': 400000, 'type': 'savings_target', 'condition': 400000, 'repeat': False},
            # {'name': 'Крупный ремонт', 'amount': 800000, 'type': 'time', 'condition': 1, 'repeat': False},
            # {'name': 'Покупка недвижимости', 'amount': 2000000, 'type': 'time', 'condition': 5, 'repeat': False},
            # {'name': 'Премиум авто', 'amount': 1500000, 'type': 'savings_target', 'condition': 2000000, 'repeat': False},
        ]
    },
    'D': {
        'initial_income': 250000,
        'initial_expenses': 80000,
        'initial_capital': 0,  # Стартовый капитал
        'income_changes': [],
        'expense_changes': [],
        'planned_expenses': [
            {'name': 'Электроскутер', 'amount': 400000, 'type': 'savings_target', 'condition': 400000, 'repeat': False},
            # {'name': 'Элитный ремонт', 'amount': 1500000, 'type': 'time', 'condition': 1, 'repeat': False},
            # {'name': 'Бизнес-инвестиции', 'amount': 3000000, 'type': 'time', 'condition': 7, 'repeat': False},
            # {'name': 'Образование детей', 'amount': 2500000, 'type': 'time', 'condition': 15, 'repeat': False},
            # {'name': 'Люксовое авто', 'amount': 2000000, 'type': 'savings_target', 'condition': 3000000, 'repeat': False},
        ]
    }
}