import numpy as np
import datetime

# Импорты из config.py (будут доступны после создания config.py)
from config import (
    PLANS, N_SCENARIOS, HORIZONS,
    DEBT_INTEREST_RATE, RESTRUCTURING_THRESHOLD_RATIO, BANKRUPTCY_THRESHOLD_RATIO,
    SAVINGS_RETURN_RATE, IDEAL_RETURN_RATE, TAX_RATE, CUSHION_AMOUNT,
    MINOR_EMERGENCY_PROB, MINOR_EMERGENCY_COST,
    MEDIUM_EMERGENCY_PROB, MEDIUM_EMERGENCY_COST,
    MAJOR_EMERGENCY_PROB, MAJOR_EMERGENCY_COST,
    MINOR_CLUSTER_PROB, MAJOR_CLUSTER_LAMBDA,
    PARTIAL_LOSS_PROB, PARTIAL_LOSS_RATE, PARTIAL_LOSS_DURATION,
    FULL_LOSS_PROB, FULL_LOSS_DURATION_MEAN, FULL_LOSS_DURATION_SD,
    RANDOM_SEED, N_MONTHS
)


def print_comparative_results(all_results):
    """Вывод сравнительных результатов симуляции в консоль"""
    avg_emergency_cost = (MINOR_EMERGENCY_PROB * MINOR_EMERGENCY_COST +
                          MEDIUM_EMERGENCY_PROB * MEDIUM_EMERGENCY_COST +
                          MAJOR_EMERGENCY_PROB * MAJOR_EMERGENCY_COST)
    
    # ИЗМЕНЕНО: теперь используем названия планов
    headers = [f"План {plan_id}" for plan_id in PLANS.keys()]
    col_width = 12
    
    for years in HORIZONS:
        months = years * 12
        print("\n" + "="*70)
        print(f" СРАВНЕНИЕ РЕЗУЛЬТАТОВ ЗА {years} ЛЕТ ({months} месяцев) ")
        print("="*70)
        
        header_str = f"{'Параметр':<30} | " + " | ".join(f"{h:>{col_width}}" for h in headers)
        print(header_str)
        print("-" * (30 + 3 + (col_width + 3) * len(PLANS) - 3))
        
        def get_values(key):
            return [all_results[plan_id][years][key] for plan_id in PLANS.keys()]
        
        # Основные показатели
        avg_wealth = get_values('avg_wealth')
        print(f"{'Средние активы (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.2f}" for v in avg_wealth))
        
        # НОВЫЕ СТРОКИ: Вероятности для моды
        modal_density = get_values('modal_density')
        print(f"{'Плотность в моде':<30} | " + " | ".join(f"{v:>{col_width}.4f}" for v in modal_density))
        
        prob_near_5 = get_values('prob_near_mode_5pct')
        print(f"{'Вероятность ±5% от моды (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in prob_near_5))
        
        prob_near_10 = get_values('prob_near_mode_10pct')
        print(f"{'Вероятность ±10% от моды (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in prob_near_10))
        
        prob_above = get_values('prob_above_mode')
        print(f"{'Вероятность выше моды (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in prob_above))
        
        prob_below = get_values('prob_below_mode')
        print(f"{'Вероятность ниже моды (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in prob_below))
        
        min_wealth = get_values('min_wealth')
        print(f"{'Минимальные активы (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.2f}" for v in min_wealth))
        
        max_wealth = get_values('max_wealth')
        print(f"{'Максимальные активы (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.2f}" for v in max_wealth))
        
        pct_zero = get_values('pct_zero')
        print(f"{'Месяцев в дефиците (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in pct_zero))
        
        pct_in_debt = get_values('pct_in_debt')
        print(f"{'Сценариев с долгом (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in pct_in_debt))
        
        avg_debt = get_values('avg_debt')
        print(f"{'Средний долг у должников (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.3f}" for v in avg_debt))
        
        # НОВОЕ: Добавляем общий средний долг для ясности
        def get_total_avg_debt(plan_id, years):
            return np.mean(all_results[plan_id][years]['final_debt'])
        
        total_avg_debt = [get_total_avg_debt(plan_id, years) for plan_id in PLANS.keys()]
        print(f"{'Общий средний долг (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.3f}" for v in total_avg_debt))
        
        # ИЗМЕНЕНО: теперь рассчитываем теоретический поток для каждого плана отдельно
        theoretical = {}
        for plan_id in PLANS.keys():
            plan_data = PLANS[plan_id]
            # Упрощенный расчет - берем начальные значения
            base_income = plan_data['initial_income']
            base_expenses = plan_data['initial_expenses']
            theoretical[plan_id] = (base_income - base_expenses 
                                   - avg_emergency_cost
                                   - PARTIAL_LOSS_PROB * base_income * PARTIAL_LOSS_RATE * PARTIAL_LOSS_DURATION
                                   - FULL_LOSS_PROB * base_income * FULL_LOSS_DURATION_MEAN)
        
        theor_values = [theoretical[plan_id] for plan_id in PLANS.keys()]
        print(f"{'Теор. денежный поток (₽)':<30} | " + " | ".join(f"{v:>{col_width},.0f}" for v in theor_values))
        
        real_cash = get_values('real_avg_cash_flow')
        print(f"{'Реал. денежный поток (₽)':<30} | " + " | ".join(f"{v:>{col_width},.0f}" for v in real_cash))
        
        efficiency = []
        for i, plan_id in enumerate(PLANS.keys()):
            eff = all_results[plan_id][years]['avg_wealth'] / (theoretical[plan_id] * months) if theoretical[plan_id] != 0 else 0
            efficiency.append(eff)
        print(f"{'Эффективность накопления':<30} | " + " | ".join(f"{v:>{col_width}.2f}" for v in efficiency))
        
        avg_minor = get_values('avg_minor_em')
        print(f"{'Мелкие ЧП (за период)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in avg_minor))
        
        avg_medium = get_values('avg_medium_em')
        print(f"{'Средние ЧП (за период)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in avg_medium))
        
        avg_major = get_values('avg_major_em')
        print(f"{'Крупные ЧП (за период)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in avg_major))
        
        # Добавляем новые метрики как отдельные строки
        median_shock = get_values('median_shock_pct')
        print(f"{'Медиана шоков (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in median_shock))
        
        p90_shock = get_values('p90_shock_pct')
        print(f"{'90-й перцентиль шоков (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in p90_shock))
        
        p95_shock = get_values('p95_shock_pct')
        print(f"{'95-й перцентиль шоков (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in p95_shock))
        
        print("-" * (30 + 3 + (col_width + 3) * len(PLANS) - 3))
        
        # НОВОЕ: Показатели стартового капитала
        print(f"{'СТАРТОВЫЙ КАПИТАЛ:':<30} | " + " | ".join(f"{'':>{col_width}}" for _ in PLANS))
        
        initial_capitals = [PLANS[plan_id].get('initial_capital', 0) for plan_id in PLANS.keys()]
        print(f"{'Изначальная сумма (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.3f}" for v in initial_capitals))
        
        potential_values = get_values('initial_capital_potential')
        print(f"{'Потенциал при 6% (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.2f}" for v in potential_values))
        
        profit_values = get_values('initial_capital_profit')
        print(f"{'Потенциальная прибыль (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.2f}" for v in profit_values))
        
        print("-" * (30 + 3 + (col_width + 3) * len(PLANS) - 3))


def save_key_scenarios_analysis(all_results, filepath):
    """
    НОВАЯ ФУНКЦИЯ: Сохранение анализа ключевых сценариев в отдельный файл
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" КЛЮЧЕВЫЕ СЦЕНАРИИ С ДЕТАЛИЗАЦИЕЙ ШОКОВ \n")
        f.write("="*70 + "\n")
        f.write(f"Сценариев: {N_SCENARIOS}\n")
        f.write("Анализ типичных сценариев с разбивкой потерь от шоков на прямые и компаундинговые\n\n")
        
        for years in HORIZONS:
            f.write(f"\n{'='*50}\n")
            f.write(f" {years} ЛЕТ - КЛЮЧЕВЫЕ СЦЕНАРИИ \n")
            f.write(f"{'='*50}\n")
            
            for plan_id in PLANS.keys():
                data = all_results[plan_id][years]
                net_wealth = data['net_wealth']
                direct_losses = np.array(data['scenarios_direct_losses'])
                compounding_losses = np.array(data['scenarios_compounding_loss'])
                
                f.write(f"\n--- План {plan_id} (начальный доход {PLANS[plan_id]['initial_income']:,}₽, стартовый капитал {PLANS[plan_id].get('initial_capital', 0):,}₽) ---\n")
                
                # Показатели стартового капитала
                initial_capital = PLANS[plan_id].get('initial_capital', 0) or 0
                if initial_capital > 0:
                    f.write(f"Стартовый капитал: {initial_capital/1e6:.3f} млн\n")
                    f.write(f"Потенциал стартового капитала: {data['initial_capital_potential']/1e6:.2f} млн (при 6% годовых)\n")
                    f.write(f"Потенциальная прибыль: {data['initial_capital_profit']/1e6:.2f} млн\n")
                else:
                    f.write(f"Стартовый капитал: 0.000 млн (начинаем с нуля)\n")
                    f.write(f"Потенциал стартового капитала: 0.00 млн\n")
                    f.write(f"Потенциальная прибыль: 0.00 млн\n")
                
                # 1. ИДЕАЛЬНЫЙ СЦЕНАРИЙ
                ideal_wealth = data['ideal_wealth'] / 1e6
                f.write(f"Идеальные активы: {ideal_wealth:.2f} млн (без шоков, 6% годовых)\n")
                
                # 2. ЛИНЕЙНЫЙ СЦЕНАРИЙ
                linear_wealth = data['linear_wealth'] / 1e6
                f.write(f"Линейные активы: {linear_wealth:.2f} млн (0% доходности, без шоков)\n")
                
                # 3. МЕДИАННЫЙ СЦЕНАРИЙ
                median_wealth = np.median(net_wealth) / 1e6
                median_idx = np.argmin(np.abs(net_wealth - np.median(net_wealth)))
                median_direct = direct_losses[median_idx]
                median_compounding = compounding_losses[median_idx]
                total_median_shocks = median_direct + median_compounding
                f.write(f"Медианные активы: {median_wealth:.2f} млн\n")
                if total_median_shocks > 0:
                    f.write(f"  ├── Общие потери от шоков: {total_median_shocks:.2f} млн\n")
                    f.write(f"  ├── Прямые потери: {median_direct:.2f} млн ({median_direct/total_median_shocks*100:.0f}%)\n")
                    f.write(f"  └── Потеря компаундинга: {median_compounding:.2f} млн ({median_compounding/total_median_shocks*100:.0f}%)\n")
                else:
                    f.write(f"  └── Потери от шоков: отсутствуют\n")
                
                # 4. МОДАЛЬНЫЙ СЦЕНАРИЙ (ближайший к моде)
                modal_value = data['modal_wealth']
                modal_idx = np.argmin(np.abs(net_wealth - modal_value))
                modal_wealth = net_wealth[modal_idx] / 1e6
                modal_direct = direct_losses[modal_idx]
                modal_compounding = compounding_losses[modal_idx]
                total_modal_shocks = modal_direct + modal_compounding
                f.write(f"Модальные активы: {modal_wealth:.2f} млн (наиболее вероятный)\n")
                if total_modal_shocks > 0:
                    f.write(f"  ├── Общие потери от шоков: {total_modal_shocks:.2f} млн\n")
                    f.write(f"  ├── Прямые потери: {modal_direct:.2f} млн ({modal_direct/total_modal_shocks*100:.0f}%)\n")
                    f.write(f"  └── Потеря компаундинга: {modal_compounding:.2f} млн ({modal_compounding/total_modal_shocks*100:.0f}%)\n")
                else:
                    f.write(f"  └── Потери от шоков: отсутствуют\n")
                
                # 5. 30-Й ПЕРЦЕНТИЛЬ (умеренно плохой сценарий)
                p30_value = np.percentile(net_wealth, 30)
                p30_idx = np.argmin(np.abs(net_wealth - p30_value))
                p30_wealth = net_wealth[p30_idx] / 1e6
                p30_direct = direct_losses[p30_idx]
                p30_compounding = compounding_losses[p30_idx]
                total_p30_shocks = p30_direct + p30_compounding
                f.write(f"30-й перцентиль умеренно плохой: {p30_wealth:.2f} млн (30% худших)\n")
                if total_p30_shocks > 0:
                    f.write(f"  ├── Общие потери от шоков: {total_p30_shocks:.2f} млн\n")
                    f.write(f"  ├── Прямые потери: {p30_direct:.2f} млн ({p30_direct/total_p30_shocks*100:.0f}%)\n")
                    f.write(f"  └── Потеря компаундинга: {p30_compounding:.2f} млн ({p30_compounding/total_p30_shocks*100:.0f}%)\n")
                else:
                    f.write(f"  └── Потери от шоков: отсутствуют\n")
                
                # 6. 20-Й ПЕРЦЕНТИЛЬ (плохой сценарий)
                p20_value = np.percentile(net_wealth, 20)
                p20_idx = np.argmin(np.abs(net_wealth - p20_value))
                p20_wealth = net_wealth[p20_idx] / 1e6
                p20_direct = direct_losses[p20_idx]
                p20_compounding = compounding_losses[p20_idx]
                total_p20_shocks = p20_direct + p20_compounding
                f.write(f"20-й перцентиль плохой: {p20_wealth:.2f} млн (20% худших)\n")
                if total_p20_shocks > 0:
                    f.write(f"  ├── Общие потери от шоков: {total_p20_shocks:.2f} млн\n")
                    f.write(f"  ├── Прямые потери: {p20_direct:.2f} млн ({p20_direct/total_p20_shocks*100:.0f}%)\n")
                    f.write(f"  └── Потеря компаундинга: {p20_compounding:.2f} млн ({p20_compounding/total_p20_shocks*100:.0f}%)\n")
                else:
                    f.write(f"  └── Потери от шоков: отсутствуют\n")
                
                # 7. 10-Й ПЕРЦЕНТИЛЬ (плохой сценарий)
                p10_value = np.percentile(net_wealth, 10)
                p10_idx = np.argmin(np.abs(net_wealth - p10_value))
                p10_wealth = net_wealth[p10_idx] / 1e6
                p10_direct = direct_losses[p10_idx]
                p10_compounding = compounding_losses[p10_idx]
                total_p10_shocks = p10_direct + p10_compounding
                f.write(f"10-й перцентиль плохой: {p10_wealth:.2f} млн (10% худших)\n")
                if total_p10_shocks > 0:
                    f.write(f"  ├── Общие потери от шоков: {total_p10_shocks:.2f} млн\n")
                    f.write(f"  ├── Прямые потери: {p10_direct:.2f} млн ({p10_direct/total_p10_shocks*100:.0f}%)\n")
                    f.write(f"  └── Потеря компаундинга: {p10_compounding:.2f} млн ({p10_compounding/total_p10_shocks*100:.0f}%)\n")
                else:
                    f.write(f"  └── Потери от шоков: отсутствуют\n")
                
                # 8. 1-Й ПЕРЦЕНТИЛЬ (критический сценарий)
                p1_value = np.percentile(net_wealth, 1)
                p1_idx = np.argmin(np.abs(net_wealth - p1_value))
                p1_wealth = net_wealth[p1_idx] / 1e6
                p1_direct = direct_losses[p1_idx]
                p1_compounding = compounding_losses[p1_idx]
                total_p1_shocks = p1_direct + p1_compounding
                f.write(f"1-й перцентиль критический: {p1_wealth:.2f} млн (1% худших)\n")
                if total_p1_shocks > 0:
                    f.write(f"  ├── Общие потери от шоков: {total_p1_shocks:.2f} млн\n")
                    f.write(f"  ├── Прямые потери: {p1_direct:.2f} млн ({p1_direct/total_p1_shocks*100:.0f}%)\n")
                    f.write(f"  └── Потеря компаундинга: {p1_compounding:.2f} млн ({p1_compounding/total_p1_shocks*100:.0f}%)\n")
                else:
                    f.write(f"  └── Потери от шоков: отсутствуют\n")
                
                f.write("\n")


def save_wealth_distribution_analysis(all_results, filepath):
    """
    НОВАЯ ФУНКЦИЯ: Сохранение распределения активов по бинам в отдельный файл
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" РАСПРЕДЕЛЕНИЕ АКТИВОВ ПО БИНАМ \n")
        f.write("="*70 + "\n")
        f.write(f"Сценариев: {N_SCENARIOS}\n")
        f.write("Анализ распределения итоговых активов с группировкой по диапазонам\n\n")
        
        for years in HORIZONS:
            f.write(f"\n{'='*50}\n")
            f.write(f" {years} ЛЕТ - РАСПРЕДЕЛЕНИЕ АКТИВОВ \n")
            f.write(f"{'='*50}\n")
            
            for i, plan_id in enumerate(PLANS.keys()):
                net_wealth = all_results[plan_id][years]['net_wealth'] / 1e6  # в млн
                min_w = np.min(net_wealth)
                max_w = np.max(net_wealth)
                med_w = np.median(net_wealth)
                mod_w = all_results[plan_id][years]['modal_wealth'] / 1e6  # модальное значение в млн
                
                f.write(f"\n--- План {plan_id} (начальный доход {PLANS[plan_id]['initial_income']:,}₽, стартовый капитал {PLANS[plan_id].get('initial_capital', 0):,}₽) ---\n")
                
                if max_w - min_w == 0:
                    f.write(f"Все сценарии дали одинаковый результат: {med_w:.2f} млн (100.0%, {N_SCENARIOS} сценариев)\n")
                    continue
                
                # Базовые 6 бинов
                range_w = max_w - min_w
                num_bins = 6
                bin_step = range_w / num_bins
                bin_edges = np.arange(min_w, max_w + bin_step / 2, bin_step)
                
                hist, edges = np.histogram(net_wealth, bins=bin_edges)
                pcts = hist / N_SCENARIOS * 100
                counts = hist
                
                # Объединение только мелких бинов
                merged_pcts = []      
                merged_counts = []    
                merged_labels = []    
                
                # Отрицательный хвост (активы < 0)
                tail_neg_pct = 0      
                tail_neg_count = 0    
                tail_neg_min = float('inf')   
                tail_neg_max = float('-inf')  
                
                # Положительный хвост (активы > 0)
                tail_pos_pct = 0      
                tail_pos_count = 0    
                tail_pos_min = float('inf')   
                tail_pos_max = float('-inf')  

                for j in range(len(hist)):
                    low = edges[j]    
                    high = edges[j+1] 
                    pct = pcts[j]     
                    count = counts[j] 
                    
                    # ПРАВИЛО: Если bin мелкий (<5%), он идет в хвост
                    if pct < 5:
                        # СЛУЧАЙ 1: Bin полностью отрицательный (все значения ≤ 0)
                        if high <= 0:
                            tail_neg_pct += pct
                            tail_neg_count += count
                            tail_neg_min = min(tail_neg_min, low)
                            tail_neg_max = max(tail_neg_max, high)
                        
                        # СЛУЧАЙ 2: Bin полностью положительный (все значения ≥ 0)
                        elif low >= 0:
                            tail_pos_pct += pct
                            tail_pos_count += count
                            tail_pos_min = min(tail_pos_min, low)
                            tail_pos_max = max(tail_pos_max, high)
                        
                        # СЛУЧАЙ 3: Bin пересекает ноль
                        else:
                            total_width = high - low
                            if total_width > 0:
                                # Отрицательная часть: от low до 0
                                neg_width = -low
                                neg_frac = neg_width / total_width
                                tail_neg_pct += pct * neg_frac
                                tail_neg_count += int(count * neg_frac)
                                tail_neg_min = min(tail_neg_min, low)
                                tail_neg_max = max(tail_neg_max, 0)
                                
                                # Положительная часть: от 0 до high
                                pos_width = high
                                pos_frac = pos_width / total_width
                                tail_pos_pct += pct * pos_frac
                                tail_pos_count += int(count * pos_frac)
                                tail_pos_min = min(tail_pos_min, 0)
                                tail_pos_max = max(tail_pos_max, high)
                    else:
                        # СЛУЧАЙ 4: Bin крупный (≥5%) - остается как отдельный бин
                        bin_label = f"{low:.1f}-{high:.1f} млн"
                        merged_pcts.append(pct)
                        merged_counts.append(count)
                        merged_labels.append(bin_label)

                # Добавление хвостов в итоговый список
                if tail_neg_pct > 0:
                    tail_label = f"{tail_neg_min:.1f}-{tail_neg_max:.1f} млн (Tail <{tail_neg_pct:.0f}%)"
                    merged_pcts.insert(0, tail_neg_pct)   
                    merged_counts.insert(0, tail_neg_count)
                    merged_labels.insert(0, tail_label)
                    
                if tail_pos_pct > 0:
                    tail_label = f"{tail_pos_min:.1f}-{tail_pos_max:.1f} млн (Tail <{tail_pos_pct:.0f}%)"
                    insert_idx = 1 if tail_neg_pct > 0 else 0
                    merged_pcts.insert(insert_idx, tail_pos_pct)
                    merged_counts.insert(insert_idx, tail_pos_count)
                    merged_labels.insert(insert_idx, tail_label)

                f.write(f"Диапазон: от {min_w:.2f} до {max_w:.2f} млн\n")
                f.write(f"Медиана: {med_w:.2f} млн\n")
                f.write(f"Мода: {mod_w:.2f} млн\n")
                f.write(f"Плотность в моде: {all_results[plan_id][years]['modal_density']:.4f}\n")
                f.write(f"Вероятность ±5% от моды: {all_results[plan_id][years]['prob_near_mode_5pct']:.1f}%\n")
                f.write(f"Вероятность ±10% от моды: {all_results[plan_id][years]['prob_near_mode_10pct']:.1f}%\n\n")
                
                f.write("Распределение по бинам:\n")
                for k in range(len(merged_pcts)):
                    pct = merged_pcts[k]
                    count = merged_counts[k]
                    bin_label = merged_labels[k]
                    if pct == 0:
                        continue

                    # Визуальный бар
                    bar_length = min(int(pct / 5), 20)
                    bar = '#' * bar_length

                    f.write(f"  {bin_label}: {pct:.1f}% ({count}) {bar}\n")
                f.write("\n")


def save_debt_analysis(all_results, filepath):
    """
    НОВАЯ ФУНКЦИЯ: Сохранение детального анализа долговой нагрузки в отдельный файл
    ОБНОВЛЕНО: теперь работает с планами
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" ДЕТАЛЬНЫЙ АНАЛИЗ ДОЛГОВОЙ НАГРУЗКИ \n")
        f.write("="*70 + "\n")
        f.write(f"Сценариев: {N_SCENARIOS}\n")
        f.write(f"Ставка по долгу (нормальная): {DEBT_INTEREST_RATE*100:.1f}%/мес (~{DEBT_INTEREST_RATE*12*100:.0f}% годовых)\n")
        f.write(f"Ставка по долгу (реструктуризация): {DEBT_INTEREST_RATE*0.5*100:.1f}%/мес (~{DEBT_INTEREST_RATE*0.5*12*100:.0f}% годовых)\n")
        f.write(f"Порог реструктуризации: {RESTRUCTURING_THRESHOLD_RATIO} годовых дохода\n")
        f.write(f"Порог банкротства: {BANKRUPTCY_THRESHOLD_RATIO} годовых дохода\n")
        f.write(f"Доходность сбережений: {SAVINGS_RETURN_RATE*100:.1f}%/мес (~{SAVINGS_RETURN_RATE*12*100:.0f}% годовых)\n")
        f.write("Примечание: ставки нельзя напрямую сравнивать (номинальная vs реальная)\n\n")
        
        for years in HORIZONS:
            f.write(f"\n{'='*50}\n")
            f.write(f" {years} ЛЕТ - АНАЛИЗ ДОЛГОВОЙ НАГРУЗКИ \n")
            f.write(f"{'='*50}\n")
            
            for plan_id in PLANS.keys():
                data = all_results[plan_id][years]
                
                f.write(f"\n--- План {plan_id} (начальный доход {PLANS[plan_id]['initial_income']:,}₽, стартовый капитал {PLANS[plan_id].get('initial_capital', 0):,}₽) ---\n")
                
                # Получаем массивы для анализа
                net_wealth = data['net_wealth']
                final_debt = data['final_debt']
                max_debt = data['max_debt']
                months_in_debt = data['months_in_debt']
                total_interest_paid = data['total_interest_paid']
                avg_debt_when_in_debt = data['avg_debt_when_in_debt']
                
                # НОВОЕ: События управления долгом
                restructuring_events = data['restructuring_events']
                bankruptcy_events = data['bankruptcy_events']
                months_in_restructuring = data['months_in_restructuring']
                
                # 1. ОБЩАЯ СТАТИСТИКА ПО ДОЛГАМ
                debt_scenarios = np.sum(final_debt > 0)
                debt_percentage = (debt_scenarios / N_SCENARIOS) * 100
                
                f.write(f"Общая статистика:\n")
                f.write(f"  ├── Сценариев с финальным долгом: {debt_scenarios} из {N_SCENARIOS} ({debt_percentage:.1f}%)\n")
                f.write(f"  ├── Средний финальный долг: {np.mean(final_debt)/1e6:.3f} млн ₽\n")
                f.write(f"  ├── Средний максимальный долг: {np.mean(max_debt)/1e6:.3f} млн ₽\n")
                f.write(f"  ├── Среднее время в долгу: {np.mean(months_in_debt):.1f} месяцев\n")
                f.write(f"  ├── Средняя сумма процентов: {np.mean(total_interest_paid)/1e6:.3f} млн ₽\n")
                f.write(f"  ├── Сценариев с реструктуризацией: {np.sum(restructuring_events > 0)} ({np.sum(restructuring_events > 0)/N_SCENARIOS*100:.1f}%)\n")
                f.write(f"  ├── Сценариев с банкротством: {np.sum(bankruptcy_events > 0)} ({np.sum(bankruptcy_events > 0)/N_SCENARIOS*100:.1f}%)\n")
                f.write(f"  └── Среднее время в реструктуризации: {np.mean(months_in_restructuring):.1f} месяцев\n\n")
                
                # 2. АНАЛИЗ ТИПИЧНЫХ СЦЕНАРИЕВ
                f.write(f"Анализ по типичным сценариям:\n\n")
                
                # Определяем индексы для каждого типичного сценария
                scenarios_data = [
                    ("Модальный сценарий (наиболее вероятный)", np.argmin(np.abs(net_wealth - data['modal_wealth']))),
                    ("30-й перцентиль (умеренно плохой)", np.argmin(np.abs(net_wealth - np.percentile(net_wealth, 30)))),
                    ("20-й перцентиль (плохой)", np.argmin(np.abs(net_wealth - np.percentile(net_wealth, 20)))),
                    ("10-й перцентиль (плохой)", np.argmin(np.abs(net_wealth - np.percentile(net_wealth, 10)))),
                    ("1-й перцентиль (критический)", np.argmin(np.abs(net_wealth - np.percentile(net_wealth, 1))))
                ]
                
                for scenario_name, scenario_idx in scenarios_data:
                    f.write(f"{scenario_name}:\n")
                    f.write(f"  ├── Итоговые активы: {net_wealth[scenario_idx]/1e6:.2f} млн\n")
                    f.write(f"  ├── Финальный долг: {final_debt[scenario_idx]/1e6:.3f} млн ₽\n")
                    f.write(f"  ├── Максимальный долг: {max_debt[scenario_idx]/1e6:.3f} млн ₽\n")
                    f.write(f"  ├── Месяцев в долгу: {months_in_debt[scenario_idx]:.0f} из {years*12}\n")
                    f.write(f"  ├── Проценты выплачено: {total_interest_paid[scenario_idx]/1e6:.3f} млн ₽\n")
                    f.write(f"  └── Средний долг (когда был): {avg_debt_when_in_debt[scenario_idx]/1e6:.3f} млн ₽\n")
                    
                    # Дополнительные метрики
                    if months_in_debt[scenario_idx] > 0:
                        debt_burden = (months_in_debt[scenario_idx] / (years * 12)) * 100
                        f.write(f"      └── Долговая нагрузка: {debt_burden:.1f}% времени\n")
                    
                    f.write("\n")
                
                # 3. РАСПРЕДЕЛЕНИЕ ПО УРОВНЯМ ДОЛГОВОЙ НАГРУЗКИ
                f.write(f"Распределение по уровням долговой нагрузки:\n")
                
                # Категории по месяцам в долгу
                no_debt = np.sum(months_in_debt == 0)
                light_debt = np.sum((months_in_debt > 0) & (months_in_debt <= years * 2))  # до 2 месяцев в год
                moderate_debt = np.sum((months_in_debt > years * 2) & (months_in_debt <= years * 6))  # 2-6 месяцев в год
                heavy_debt = np.sum((months_in_debt > years * 6) & (months_in_debt <= years * 10))  # 6-10 месяцев в год
                extreme_debt = np.sum(months_in_debt > years * 10)  # больше 10 месяцев в год
                
                f.write(f"  ├── Без долгов: {no_debt} сценариев ({no_debt/N_SCENARIOS*100:.1f}%)\n")
                f.write(f"  ├── Легкая нагрузка (≤{years*2} мес): {light_debt} сценариев ({light_debt/N_SCENARIOS*100:.1f}%)\n")
                f.write(f"  ├── Умеренная нагрузка ({years*2}-{years*6} мес): {moderate_debt} сценариев ({moderate_debt/N_SCENARIOS*100:.1f}%)\n")
                f.write(f"  ├── Тяжелая нагрузка ({years*6}-{years*10} мес): {heavy_debt} сценариев ({heavy_debt/N_SCENARIOS*100:.1f}%)\n")
                f.write(f"  └── Критическая нагрузка (>{years*10} мес): {extreme_debt} сценариев ({extreme_debt/N_SCENARIOS*100:.1f}%)\n\n")
                
                # 4. ВЛИЯНИЕ ПРОЦЕНТОВ НА ИТОГОВЫЕ РЕЗУЛЬТАТЫ
                f.write(f"Влияние процентов на итоговые результаты:\n")
                scenarios_with_interest = total_interest_paid > 0
                if np.any(scenarios_with_interest):
                    avg_interest_loss = np.mean(total_interest_paid[scenarios_with_interest]) / 1e6
                    max_interest_loss = np.max(total_interest_paid) / 1e6
                    interest_scenarios = np.sum(scenarios_with_interest)
                    
                    f.write(f"  ├── Сценариев с процентами: {interest_scenarios} из {N_SCENARIOS} ({interest_scenarios/N_SCENARIOS*100:.1f}%)\n")
                    f.write(f"  ├── Средняя потеря на процентах: {avg_interest_loss:.3f} млн ₽\n")
                    f.write(f"  ├── Максимальная потеря: {max_interest_loss:.3f} млн ₽\n")
                    
                    # Процент от итоговых активов
                    avg_wealth_with_interest = np.mean(net_wealth[scenarios_with_interest])
                    if avg_wealth_with_interest > 0:
                        interest_impact = (avg_interest_loss * 1e6 / avg_wealth_with_interest) * 100
                        f.write(f"  └── Влияние на итоговые активы: -{interest_impact:.1f}%\n")
                else:
                    f.write(f"  └── Ни в одном сценарии не было процентов по долгу\n")
                
                f.write("\n")


def save_planned_expenses_analysis(all_results, filepath):
    """
    НОВАЯ ФУНКЦИЯ: Сохранение анализа потерь от запланированных расходов в отдельный файл
    ОБНОВЛЕНО: теперь работает с индивидуальными запланированными расходами планов
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" АНАЛИЗ ПОТЕРЬ ОТ ЗАПЛАНИРОВАННЫХ РАСХОДОВ \n")
        f.write("="*70 + "\n")
        f.write(f"Сценариев: {N_SCENARIOS}\n")
        f.write(f"Каждый план имеет индивидуальные запланированные расходы\n\n")
        
        f.write("Запланированные расходы по планам:\n")
        for plan_id, plan_data in PLANS.items():
            plan_expenses = plan_data.get('planned_expenses', [])
            f.write(f"План {plan_id}:\n")
            if plan_expenses:
                for i, expense in enumerate(plan_expenses):
                    f.write(f"  {i+1}. {expense['name']}: {expense['amount']:,} ₽\n")
                    if expense['type'] == 'time':
                        f.write(f"     Тип: Временной (начиная с года {expense['condition']})\n")
                    else:
                        f.write(f"     Тип: По накоплению ({expense['condition']:,} ₽)\n")
            else:
                f.write(f"  Запланированных расходов нет\n")
            f.write("\n")
        
        for years in HORIZONS:
            f.write(f"\n{'='*50}\n")
            f.write(f" {years} ЛЕТ - АНАЛИЗ ЗАПЛАНИРОВАННЫХ РАСХОДОВ \n")
            f.write(f"{'='*50}\n")
            
            for plan_id in PLANS.keys():
                data = all_results[plan_id][years]
                plan_expenses = PLANS[plan_id].get('planned_expenses', [])
                
                f.write(f"\n--- План {plan_id} (начальный доход {PLANS[plan_id]['initial_income']:,}₽, {len(plan_expenses)} запланированных расходов) ---\n")
                
                # 1. ОБЩАЯ СТАТИСТИКА
                planned_expenses_array = np.array(data['scenarios_planned_expenses'])
                
                if np.sum(planned_expenses_array) > 0:
                    # Модальные запланированные расходы (наиболее частое значение)
                    unique_values, counts = np.unique(planned_expenses_array, return_counts=True)
                    modal_idx = np.argmax(counts)
                    modal_planned = unique_values[modal_idx]
                    modal_frequency = counts[modal_idx] / N_SCENARIOS * 100
                    
                    f.write(f"Модальные запланированные расходы: {modal_planned:.2f} млн\n")
                    f.write(f"Частота модального значения: {modal_frequency:.1f}% сценариев\n")
                    
                    # Альтернативное представление для ненулевых трат
                    nonzero_mask = planned_expenses_array > 0
                    if np.any(nonzero_mask):
                        nonzero_expenses = planned_expenses_array[nonzero_mask]
                        nonzero_frequency = len(nonzero_expenses) / N_SCENARIOS * 100
                        avg_nonzero = np.mean(nonzero_expenses)
                        f.write(f"В {nonzero_frequency:.1f}% сценариев потрачено в среднем {avg_nonzero:.2f} млн\n")
                else:
                    f.write(f"Запланированных расходов не было ни в одном сценарии\n")
                
                avg_planned_comp_loss = data['avg_planned_compounding_loss']
                ideal_wealth = data['ideal_wealth'] / 1e6
                
                f.write(f"Средняя потеря компаундинга от запланированных расходов: {avg_planned_comp_loss:.2f} млн\n")
                f.write(f"Идеальное богатство (с учетом запланированных расходов): {ideal_wealth:.2f} млн\n")
                
                if np.mean(planned_expenses_array) > 0:
                    comp_ratio = avg_planned_comp_loss / np.mean(planned_expenses_array)
                    f.write(f"Соотношение потери компаундинга к средним расходам: {comp_ratio:.2f}x\n")
                
                # 2. СТАТИСТИКА ПО ТИПАМ РАСХОДОВ
                f.write(f"\nДетализация по типам запланированных расходов:\n")
                for name, stats in data['planned_expenses_stats'].items():
                    if stats['count'] > 0:
                        avg_amount = stats['total_amount'] / stats['count'] / 1e6
                        frequency = stats['count'] / N_SCENARIOS * 100
                        f.write(f"  {name}:\n")
                        f.write(f"    ├── Частота: {frequency:.1f}% сценариев ({stats['count']} из {N_SCENARIOS})\n")
                        f.write(f"    └── Средняя сумма: {avg_amount:.2f} млн\n")
                    else:
                        f.write(f"  {name}: Не произошло ни в одном сценарии\n")
                
                f.write("\n")


def save_results_to_text(all_results, filepath):
    """Сохранение результатов в текстовый файл с тем же форматированием что и в консоли
    ОБНОВЛЕНО: теперь работает с планами"""
    avg_emergency_cost = (MINOR_EMERGENCY_PROB * MINOR_EMERGENCY_COST +
                          MEDIUM_EMERGENCY_PROB * MEDIUM_EMERGENCY_COST +
                          MAJOR_EMERGENCY_PROB * MAJOR_EMERGENCY_COST)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Заголовок
        f.write("="*70 + "\n")
        f.write(" СРАВНИТЕЛЬНАЯ ФИНАНСОВАЯ СИМУЛЯЦИЯ С АНАЛИЗОМ ПОТЕРЬ ОТ ШОКОВ \n")
        f.write("="*70 + "\n")
        f.write(f"Сценариев: {N_SCENARIOS}, Месяцев: {N_MONTHS} (30 лет)\n")
        f.write("Базовые параметры:\n")
        f.write(f"- Доходность сбережений: {SAVINGS_RETURN_RATE*100:.1f}%/мес (~{SAVINGS_RETURN_RATE*12*100:.0f}% годовых)\n")
        f.write(f"- Налог на инвестиционный доход: {TAX_RATE*100:.0f}%\n")
        f.write(f"- Идеальная доходность: {IDEAL_RETURN_RATE*100:.1f}%/мес (~{IDEAL_RETURN_RATE*12*100:.0f}% годовых)\n")
        f.write(f"- Подушка безопасности: {CUSHION_AMOUNT:,}₽ (0% доходности)\n")
        f.write(f"- Ставка по долгу: {DEBT_INTEREST_RATE*100:.1f}%/мес (~{DEBT_INTEREST_RATE*12*100:.0f}% годовых)\n")
        f.write(f"- Потенциал сбережений (буфер): доход - расходы (индивидуально для каждого плана)\n")
        f.write(f"- Модальные активы: наиболее вероятное значение (пик плотности распределения)\n")
        f.write(f"- Плотность в моде: относительная частота появления модального значения\n")
        f.write(f"- Вероятности ±5%/±10%: процент сценариев в окрестности моды\n")
        f.write("- Все суммы в реальной покупательной способности (после инфляции)\n")
        f.write("- ДОБАВЛЕНО: Подушка безопасности 200,000₽ для непредвиденных расходов\n")
        f.write("- ДОБАВЛЕНО: Процентная ставка на долг 24% годовых (реализм кредитования)\n")
        f.write("- ДОБАВЛЕНО: Умная логика запланированных расходов (ожидание накопления вместо долга)\n")
        f.write("- ДОБАВЛЕНО: Детальный анализ долговой нагрузки по типичным сценариям\n")
        f.write("- ДОБАВЛЕНО: Анализ потерь от шоков с распределением по бинам\n")
        f.write("- ДОБАВЛЕНО: Анализ запланированных расходов и их влияния на компаундинг\n")
        f.write("- НОВОЕ: Динамические планы с изменением дохода/расходов во времени\n")
        f.write("- НОВОЕ: Показатели стартового капитала (потенциал при идеальной доходности 6% годовых)\n")
        f.write("- ИСПРАВЛЕНО: Виртуальный сценарий для правильного расчета потерь компаундинга\n")
        f.write("- ИСПРАВЛЕНО: Единая система долгов во всех сценариях (идеальный, линейный, виртуальный)\n")
        
        f.write("\nРиски ЧП (Москва 2025):\n")
        f.write(f"- Мелкие траты ({MINOR_EMERGENCY_PROB*100:.1f}%/мес): {MINOR_EMERGENCY_COST}₽\n")
        f.write(f"- Средние траты ({MEDIUM_EMERGENCY_PROB*100:.1f}%/мес): {MEDIUM_EMERGENCY_COST}₽\n")
        f.write(f"- Крупные траты ({MAJOR_EMERGENCY_PROB*100:.1f}%/мес): {MAJOR_EMERGENCY_COST}₽\n")
        f.write(f"  Кластеризация мелких/средних: {MINOR_CLUSTER_PROB*100:.0f}% вероятность продолжения\n")
        f.write(f"  Кластеризация крупных: распределение Пуассона (λ={MAJOR_CLUSTER_LAMBDA})\n")
        f.write(f"- Частичная потеря дохода: {PARTIAL_LOSS_PROB*100:.2f}%/мес ({PARTIAL_LOSS_RATE*100:.0f}% от дохода)\n")
        f.write(f"  Средняя длительность: {PARTIAL_LOSS_DURATION:.1f} мес\n")
        f.write(f"- Полная потеря дохода: {FULL_LOSS_PROB*100:.3f}%/мес\n")
        f.write(f"  Средняя длительность: {FULL_LOSS_DURATION_MEAN:.1f} мес\n")
        
        f.write("\nПланы (траектории):\n")
        for plan_id, plan_data in PLANS.items():
            plan_expenses = plan_data.get('planned_expenses', [])
            f.write(f"- План {plan_id}: {plan_data['initial_income']:,}₽ → {plan_data['initial_expenses']:,}₽ (стартовый капитал: {plan_data.get('initial_capital', 0):,}₽, расходов: {len(plan_expenses)})\n")
            if plan_data['income_changes']:
                f.write(f"  Изменения дохода: {len(plan_data['income_changes'])} событий\n")
            if plan_data['expense_changes']:
                f.write(f"  Изменения расходов: {len(plan_data['expense_changes'])} событий\n")
            if plan_expenses:
                f.write(f"  Запланированные расходы: {len(plan_expenses)} событий\n")
                for i, expense in enumerate(plan_expenses):
                    if expense['type'] == 'time':
                        f.write(f"    ├── {expense['name']}: {expense['amount']:,}₽ (начиная с года {expense['condition']})\n")
                    else:
                        f.write(f"    ├── {expense['name']}: {expense['amount']:,}₽ (при накоплении {expense['condition']:,}₽)\n")
        
        f.write("="*70 + "\n")
        
        headers = [f"План {plan_id}" for plan_id in PLANS.keys()]
        col_width = 12
        
        for years in HORIZONS:
            months = years * 12
            f.write("\n" + "="*70 + "\n")
            f.write(f" СРАВНЕНИЕ РЕЗУЛЬТАТОВ ЗА {years} ЛЕТ ({months} месяцев) \n")
            f.write("="*70 + "\n")
            
            header_str = f"{'Параметр':<30} | " + " | ".join(f"{h:>{col_width}}" for h in headers)
            f.write(header_str + "\n")
            f.write("-" * (30 + 3 + (col_width + 3) * len(PLANS) - 3) + "\n")
            
            def get_values(key):
                return [all_results[plan_id][years][key] for plan_id in PLANS.keys()]
            
            # Основные показатели
            avg_wealth = get_values('avg_wealth')
            f.write(f"{'Средние активы (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.2f}" for v in avg_wealth) + "\n")
            
            # НОВЫЕ СТРОКИ: Вероятности для моды
            modal_density = get_values('modal_density')
            f.write(f"{'Плотность в моде':<30} | " + " | ".join(f"{v:>{col_width}.4f}" for v in modal_density) + "\n")
            
            prob_near_5 = get_values('prob_near_mode_5pct')
            f.write(f"{'Вероятность ±5% от моды (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in prob_near_5) + "\n")
            
            prob_near_10 = get_values('prob_near_mode_10pct')
            f.write(f"{'Вероятность ±10% от моды (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in prob_near_10) + "\n")
            
            prob_above = get_values('prob_above_mode')
            f.write(f"{'Вероятность выше моды (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in prob_above) + "\n")
            
            prob_below = get_values('prob_below_mode')
            f.write(f"{'Вероятность ниже моды (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in prob_below) + "\n")
            
            min_wealth = get_values('min_wealth')
            f.write(f"{'Минимальные активы (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.2f}" for v in min_wealth) + "\n")
            
            max_wealth = get_values('max_wealth')
            f.write(f"{'Максимальные активы (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.2f}" for v in max_wealth) + "\n")
            
            pct_zero = get_values('pct_zero')
            f.write(f"{'Месяцев в дефиците (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in pct_zero) + "\n")
            
            pct_in_debt = get_values('pct_in_debt')
            f.write(f"{'Сценариев с долгом (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in pct_in_debt) + "\n")
            
            avg_debt = get_values('avg_debt')
            f.write(f"{'Средний долг у должников (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.3f}" for v in avg_debt) + "\n")
            
            # НОВОЕ: Добавляем общий средний долг для ясности
            def get_total_avg_debt_file(plan_id, years):
                return np.mean(all_results[plan_id][years]['final_debt'])
            
            total_avg_debt = [get_total_avg_debt_file(plan_id, years) for plan_id in PLANS.keys()]
            f.write(f"{'Общий средний долг (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.3f}" for v in total_avg_debt) + "\n")
            
            theoretical = {}
            for plan_id in PLANS.keys():
                plan_data = PLANS[plan_id]
                base_income = plan_data['initial_income']
                base_expenses = plan_data['initial_expenses']
                theoretical[plan_id] = (base_income - base_expenses 
                                       - avg_emergency_cost
                                       - PARTIAL_LOSS_PROB * base_income * PARTIAL_LOSS_RATE * PARTIAL_LOSS_DURATION
                                       - FULL_LOSS_PROB * base_income * FULL_LOSS_DURATION_MEAN)
            
            theor_values = [theoretical[plan_id] for plan_id in PLANS.keys()]
            f.write(f"{'Теор. денежный поток (₽)':<30} | " + " | ".join(f"{v:>{col_width},.0f}" for v in theor_values) + "\n")
            
            real_cash = get_values('real_avg_cash_flow')
            f.write(f"{'Реал. денежный поток (₽)':<30} | " + " | ".join(f"{v:>{col_width},.0f}" for v in real_cash) + "\n")
            
            efficiency = []
            for i, plan_id in enumerate(PLANS.keys()):
                eff = all_results[plan_id][years]['avg_wealth'] / (theoretical[plan_id] * months) if theoretical[plan_id] != 0 else 0
                efficiency.append(eff)
            f.write(f"{'Эффективность накопления':<30} | " + " | ".join(f"{v:>{col_width}.2f}" for v in efficiency) + "\n")
            
            avg_minor = get_values('avg_minor_em')
            f.write(f"{'Мелкие ЧП (за период)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in avg_minor) + "\n")
            
            avg_medium = get_values('avg_medium_em')
            f.write(f"{'Средние ЧП (за период)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in avg_medium) + "\n")
            
            avg_major = get_values('avg_major_em')
            f.write(f"{'Крупные ЧП (за период)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in avg_major) + "\n")
            
            median_shock = get_values('median_shock_pct')
            f.write(f"{'Медиана шоков (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in median_shock) + "\n")
            
            p90_shock = get_values('p90_shock_pct')
            f.write(f"{'90-й перцентиль шоков (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in p90_shock) + "\n")
            
            p95_shock = get_values('p95_shock_pct')
            f.write(f"{'95-й перцентиль шоков (%)':<30} | " + " | ".join(f"{v:>{col_width}.1f}" for v in p95_shock) + "\n")
            
            f.write("-" * (30 + 3 + (col_width + 3) * len(PLANS) - 3) + "\n")
            
            # НОВОЕ: Показатели стартового капитала
            f.write(f"{'СТАРТОВЫЙ КАПИТАЛ:':<30} | " + " | ".join(f"{'':>{col_width}}" for _ in PLANS) + "\n")
            
            initial_capitals = [PLANS[plan_id].get('initial_capital', 0) for plan_id in PLANS.keys()]
            f.write(f"{'Изначальная сумма (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.3f}" for v in initial_capitals) + "\n")
            
            potential_values = get_values('initial_capital_potential')
            f.write(f"{'Потенциал при 6% (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.2f}" for v in potential_values) + "\n")
            
            profit_values = get_values('initial_capital_profit')
            f.write(f"{'Потенциальная прибыль (млн)':<30} | " + " | ".join(f"{v/1e6:>{col_width}.2f}" for v in profit_values) + "\n")
            
            f.write("-" * (30 + 3 + (col_width + 3) * len(PLANS) - 3) + "\n")


def save_shock_analysis_to_text(all_results, filepath):
    """ОБНОВЛЕНО: теперь работает с планами"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n" + "="*70 + "\n")
        f.write(" АНАЛИЗ ПОТЕРЬ ОТ ШОКОВ \n")
        f.write("="*70 + "\n")
        
        for years in HORIZONS:
            f.write(f"\n{'='*50}\n")
            f.write(f" {years} ЛЕТ - РАСПРЕДЕЛЕНИЕ ПОТЕРЬ ОТ ШОКОВ \n")
            f.write(f"{'='*50}\n")
            
            for plan_id in PLANS.keys():
                data = all_results[plan_id][years]
                
                # Получаем данные для анализа
                direct_losses = np.array(data['scenarios_direct_losses'])
                compounding_losses = np.array(data['scenarios_compounding_loss'])
                net_wealth = data['net_wealth'] / 1000000  # в млн
                
                if len(direct_losses) == 0:
                    continue
                
                f.write(f"\n--- План {plan_id} (начальный доход {PLANS[plan_id]['initial_income']:,}₽, стартовый капитал {PLANS[plan_id].get('initial_capital', 0):,}₽) ---\n")
                
                # 1. РАСПРЕДЕЛЕНИЕ ПО БИНАМ ПРЯМЫХ ПОТЕРЬ
                min_loss = np.min(direct_losses)
                max_loss = np.max(direct_losses)
                med_loss = np.median(direct_losses)
                
                if max_loss - min_loss == 0:
                    f.write(f"  План {plan_id} (все сценарии: {med_loss:.2f} млн потерь): 100.0% ({N_SCENARIOS})\n")
                    continue
                
                # Критические сценарии (высокие перцентили = худшие случаи)
                f.write(f"\nКритические сценарии потерь:\n")
                
                # Перцентили для прямых потерь (высокие = худшие)
                percentiles = [99, 95, 90, 50, 10]  # 99-й = 1% худших сценариев
                
                for p in percentiles:
                    loss_threshold = np.percentile(direct_losses, p)
                    
                    # Находим сценарии с потерями >= этого порога
                    mask = direct_losses >= loss_threshold
                    scenarios_count = np.sum(mask)
                    
                    if scenarios_count == 0:
                        continue
                    
                    avg_direct = np.mean(direct_losses[mask])
                    avg_compounding = np.mean(compounding_losses[mask])
                    avg_wealth_final = np.mean(net_wealth[mask])
                    
                    scenario_name = ""
                    if p == 99:
                        scenario_name = "1% худших"
                    elif p == 95:
                        scenario_name = "5% худших" 
                    elif p == 90:
                        scenario_name = "10% худших"
                    elif p == 50:
                        scenario_name = "медианный"
                    elif p == 10:
                        scenario_name = "10% лучших"
                    
                    f.write(f"  {p}-й перцентиль ({scenario_name}): {avg_direct:.2f} млн прямых + {avg_compounding:.2f} млн компаундинга\n")
                    f.write(f"    └── Итоговые активы: {avg_wealth_final:.2f} млн ({scenarios_count} сценариев)\n")
                
                f.write("\n")


def save_simulation_parameters(filepath):
    """Сохранение параметров симуляции для воспроизводимости"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write(" ПАРАМЕТРЫ СИМУЛЯЦИИ \n")
        f.write("="*50 + "\n")
        f.write(f"Дата и время запуска: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Количество сценариев: {N_SCENARIOS:,}\n")
        f.write(f"Количество месяцев: {N_MONTHS} (30 лет)\n")
        f.write(f"Горизонты анализа: {HORIZONS} лет\n\n")
        
        f.write("Финансовые параметры:\n")
        f.write(f"  ├── Доходность сбережений: {SAVINGS_RETURN_RATE*100:.1f}%/мес (~{SAVINGS_RETURN_RATE*12*100:.0f}% годовых)\n")
        f.write(f"  ├── Идеальная доходность: {IDEAL_RETURN_RATE*100:.1f}%/мес (~{IDEAL_RETURN_RATE*12*100:.0f}% годовых)\n")
        f.write(f"  ├── Налог на инвестдоход: {TAX_RATE*100:.0f}%\n")
        f.write(f"  ├── Подушка безопасности: {CUSHION_AMOUNT:,}₽\n")
        f.write(f"  ├── Ставка по долгу: {DEBT_INTEREST_RATE*100:.1f}%/мес (~{DEBT_INTEREST_RATE*12*100:.0f}% годовых)\n")
        f.write(f"  ├── Порог реструктуризации: {RESTRUCTURING_THRESHOLD_RATIO} годовых дохода\n")
        f.write(f"  └── Порог банкротства: {BANKRUPTCY_THRESHOLD_RATIO} годовых дохода\n\n")
        
        f.write("Планы:\n")
        for plan_id, plan_data in PLANS.items():
            f.write(f"  План {plan_id}:\n")
            f.write(f"    ├── Доход: {plan_data['initial_income']:,}₽/мес\n")
            f.write(f"    ├── Расходы: {plan_data['initial_expenses']:,}₽/мес\n")
            f.write(f"    ├── Стартовый капитал: {plan_data.get('initial_capital', 0):,}₽\n")
            f.write(f"    ├── Изменения дохода: {len(plan_data['income_changes'])} событий\n")
            f.write(f"    ├── Изменения расходов: {len(plan_data['expense_changes'])} событий\n")
            f.write(f"    └── Запланированные расходы: {len(plan_data.get('planned_expenses', []))} событий\n")
            
            if plan_data['income_changes']:
                f.write(f"      Изменения дохода:\n")
                for change in plan_data['income_changes']:
                    year = (change['month'] - 1) // 12 + 1
                    f.write(f"        └── Месяц {change['month']} (год {year}): {change['new_income']:,}₽\n")
            
            if plan_data['expense_changes']:
                f.write(f"      Изменения расходов:\n")
                for change in plan_data['expense_changes']:
                    year = (change['month'] - 1) // 12 + 1
                    f.write(f"        └── Месяц {change['month']} (год {year}): {change['new_expenses']:,}₽\n")
            
            # НОВОЕ: Вывод запланированных расходов для каждого плана
            plan_expenses = plan_data.get('planned_expenses', [])
            if plan_expenses:
                f.write(f"      Запланированные расходы:\n")
                for i, expense in enumerate(plan_expenses):
                    f.write(f"        {i+1}. {expense['name']}: {expense['amount']:,}₽\n")
                    if expense['type'] == 'time':
                        f.write(f"           Тип: Временной (начиная с года {expense['condition']})\n")
                    else:
                        f.write(f"           Тип: По накоплению ({expense['condition']:,}₽)\n")
            f.write("\n")
        
        f.write("Риски ЧП:\n")
        f.write(f"  ├── Мелкие ({MINOR_EMERGENCY_PROB*100:.1f}%/мес): {MINOR_EMERGENCY_COST:,}₽\n")
        f.write(f"  ├── Средние ({MEDIUM_EMERGENCY_PROB*100:.1f}%/мес): {MEDIUM_EMERGENCY_COST:,}₽\n")
        f.write(f"  ├── Крупные ({MAJOR_EMERGENCY_PROB*100:.1f}%/мес): {MAJOR_EMERGENCY_COST:,}₽\n")
        f.write(f"  ├── Кластеризация мелких/средних: {MINOR_CLUSTER_PROB*100:.0f}%\n")
        f.write(f"  ├── Кластеризация крупных: Пуассон (λ={MAJOR_CLUSTER_LAMBDA})\n")
        f.write(f"  ├── Частичная потеря дохода: {PARTIAL_LOSS_PROB*100:.2f}%/мес ({PARTIAL_LOSS_RATE*100:.0f}%, {PARTIAL_LOSS_DURATION:.1f} мес)\n")
        f.write(f"  └── Полная потеря дохода: {FULL_LOSS_PROB*100:.3f}%/мес ({FULL_LOSS_DURATION_MEAN:.1f} мес)\n")