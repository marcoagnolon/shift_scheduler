#v2
import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import datetime

# -----------------------------
# Global Constants and Defaults
# -----------------------------
SHOP_NAMES = ["Piazza Milano", "Piazza Aurora"]
SHIFT_NAMES = ["Morning", "Afternoon"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

DEFAULT_SHOP_SHIFT_SCHEDULE = {
    "Piazza Milano": {
         "Morning": (datetime.time(8, 30), datetime.time(12, 45)),
         "Afternoon": (datetime.time(15, 30), datetime.time(21, 0))
    },
    "Piazza Aurora": {
         "Morning": (datetime.time(8, 30), datetime.time(12, 45)),
         "Afternoon": (datetime.time(15, 30), datetime.time(21, 0))
    }
}

DEFAULT_SHOP_CAPACITY = {
    "Piazza Milano": {"min": 4, "max": 6},
    "Piazza Aurora": {"min": 2, "max": 3}
}

DEFAULT_SHOP_SHIFT_FORECAST = {
    "Piazza Milano": {
         "Morning": 50,
         "Afternoon": 50
    },
    "Piazza Aurora": {
         "Morning": 40,
         "Afternoon": 40
    }
}

MIN_REGULAR_SHIFT_DURATION_HOURS = 4
MAX_REGULAR_SHIFT_DURATION_HOURS = 8
MIN_TIME_UNIT = 15
MAX_CONSECUTIVE_SHIFTS = 10  # Maximum consecutive shift slots (within one week)
MAX_CONSECUTIVE_DAYS = 8     # Maximum consecutive days allowed

# -----------------------------
# CSS for Calendar Rendering
# -----------------------------
CALENDAR_CSS = """
<style>
/* Base Styles for Calendar Table */
.calendar-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1rem;
}
.calendar-header {
    background-color: #333;
    color: #fff;
    text-align: center;
    padding: 8px;
    border: 1px solid #ddd;
}
.worker-name-cell {
    font-weight: bold;
    padding: 8px;
    border: 1px solid #ddd;
}
.shift-pill, .shift-box {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 8px;
    margin: 2px;
    color: #fff;
}
.morning-pill {
    background-color: #5F8B4C;
}
.afternoon-pill {
    background-color: #FF9A9A;
}
.calendar-table tr:nth-child(odd) {
    background-color: #505A5B;
}
.calendar-table tr:nth-child(even) {
    background-color: #343F3E;
}
/* Responsive Adjustments */
@media (max-width: 768px) {
    .calendar-table {
        font-size: 12px;
    }
    .calendar-header, .worker-name-cell {
        padding: 4px;
    }
    .shift-pill, .shift-box {
        font-size: 10px;
        padding: 2px 4px;
    }
}
@media (max-width: 480px) {
    .calendar-table {
        font-size: 10px;
    }
    .calendar-header, .worker-name-cell {
        padding: 2px;
    }
    .shift-pill, .shift-box {
        font-size: 8px;
        padding: 1px 2px;
    }
}
</style>
"""

# -----------------------------
# Helper: Copy Parameters from one week to the next
# -----------------------------
def copy_week_parameters(from_week, to_week):
    prefix_from = f"week_{from_week}_"
    prefix_to = f"week_{to_week}_"
    for key in list(st.session_state.keys()):
        if key.startswith(prefix_from):
            new_key = key.replace(prefix_from, prefix_to, 1)
            if new_key not in st.session_state:
                st.session_state[new_key] = st.session_state[key]

# -----------------------------
# Parameter Gathering Functions
# -----------------------------
def get_shop_settings_per_shop(week_key="default"):
    shop_shift_schedule_ui = {}
    shop_capacity_ui = {}
    with st.sidebar.expander("Shop–Specific Global Settings", expanded=True):
        for shop in SHOP_NAMES:
            st.subheader(f"{shop} Global Settings")
            shop_shift_schedule_ui[shop] = {}
            shop_capacity_ui[shop] = {}
            for shift in SHIFT_NAMES:
                st.markdown(f"**{shift} Shift for {shop}**")
                default_open, default_close = DEFAULT_SHOP_SHIFT_SCHEDULE[shop][shift]
                default_min_workers = DEFAULT_SHOP_CAPACITY[shop]["min"]
                open_key = f"{week_key}_{shop}_{shift}_open"
                close_key = f"{week_key}_{shop}_{shift}_close"
                min_key = f"{week_key}_{shop}_{shift}_min"
                open_time = st.time_input(
                    f"{shop} {shift} Opening Time", 
                    value=st.session_state.get(open_key, default_open),
                    key=open_key
                )
                close_time = st.time_input(
                    f"{shop} {shift} Closing Time", 
                    value=st.session_state.get(close_key, default_close),
                    key=close_key
                )
                min_workers = st.number_input(
                    f"{shop} {shift} Minimum Workers", 
                    value=st.session_state.get(min_key, default_min_workers), 
                    min_value=0, step=1, 
                    key=min_key
                )
                shop_shift_schedule_ui[shop][shift] = (open_time, close_time)
                shop_capacity_ui[shop][shift] = min_workers
    return shop_shift_schedule_ui, shop_capacity_ui

def get_day_shift_overrides_per_shop(shop_capacity_ui, week_key="default"):
    day_shift_overrides = {}
    for shop in SHOP_NAMES:
        with st.sidebar.expander(f"{shop} - Day–Shift Overrides", expanded=False):
            st.markdown(f"Overrides for {shop}")
            for day in DAY_NAMES:
                st.markdown(f"**{day}**")
                for shift in SHIFT_NAMES:
                    override_key = f"{week_key}_{shop}_{day}_{shift}_override"
                    override = st.checkbox(
                        f"Override settings for {shop} {day} – {shift}?", 
                        key=override_key
                    )
                    if override:
                        closed_key = f"{week_key}_{shop}_{day}_{shift}_closed"
                        is_closed = st.checkbox(
                            f"Mark {shop} {day} – {shift} as closed?", 
                            key=closed_key
                        )
                        if is_closed:
                            day_shift_overrides[(shop, day, shift)] = {"closed": True}
                        else:
                            open_key = f"{week_key}_{shop}_{day}_{shift}_open"
                            close_key = f"{week_key}_{shop}_{day}_{shift}_close"
                            min_key = f"{week_key}_{shop}_{day}_{shift}_min"
                            override_open = st.time_input(
                                f"{shop} {day} – {shift} Opening Time", 
                                value=st.session_state.get(open_key, datetime.time(8, 30)), 
                                key=open_key
                            )
                            override_close = st.time_input(
                                f"{shop} {day} – {shift} Closing Time", 
                                value=st.session_state.get(close_key, datetime.time(21, 0)), 
                                key=close_key
                            )
                            override_min_workers = st.number_input(
                                f"{shop} {day} – {shift} Minimum Workers",
                                value=st.session_state.get(min_key, shop_capacity_ui[shop][shift]),
                                min_value=0, step=1,
                                key=min_key
                            )
                            day_shift_overrides[(shop, day, shift)] = {
                                "closed": False,
                                "open_time": override_open,
                                "close_time": override_close,
                                "min_workers": override_min_workers
                            }
    return day_shift_overrides

def get_parameters(week_key="default"):
    shop_names = SHOP_NAMES
    day_names = DAY_NAMES
    shop_shift_schedule_ui, shop_capacity_ui = get_shop_settings_per_shop(week_key=week_key)
    day_shift_overrides = get_day_shift_overrides_per_shop(shop_capacity_ui, week_key=week_key)
    
    shift_availability = {}   # True if the shop/day/shift is active; False if closed.
    shift_regular_time = {}   # Regular work time in quarter-hour increments.
    shift_extra_time = {}     # Extra time beyond the regular period.
    min_workers_dict = {}     # Minimum required workers per shop/day/shift.
    shift_forecasts = {}      # Customer forecast per shop/day/shift.
    min_duration_minutes = MIN_REGULAR_SHIFT_DURATION_HOURS * 60
    max_duration_minutes = MAX_REGULAR_SHIFT_DURATION_HOURS * 60
    
    for shop in shop_names:
        for day in day_names:
            for shift in SHIFT_NAMES:
                if (shop, day, shift) in day_shift_overrides:
                    override_values = day_shift_overrides[(shop, day, shift)]
                    if override_values.get("closed", False):
                        shift_availability[(shop, day, shift)] = False
                        min_workers_dict[(shop, day, shift)] = 0
                        shift_forecasts[(shop, day, shift)] = 0
                        shift_regular_time[(shop, day, shift)] = 0
                        shift_extra_time[(shop, day, shift)] = 0
                        continue
                    else:
                        start_time = override_values["open_time"]
                        end_time = override_values["close_time"]
                        min_workers_override = override_values["min_workers"]
                        shift_availability[(shop, day, shift)] = True
                else:
                    start_time, end_time = shop_shift_schedule_ui[shop][shift]
                    min_workers_override = shop_capacity_ui[shop][shift]
                    shift_availability[(shop, day, shift)] = True

                dt_start = datetime.datetime.combine(datetime.date.today(), start_time)
                dt_end = datetime.datetime.combine(datetime.date.today(), end_time)
                if dt_end < dt_start:
                    dt_end += datetime.timedelta(days=1)
                total_minutes = (dt_end - dt_start).seconds // 60
                if total_minutes < min_duration_minutes:
                    st.error(f"Invalid time configuration for {shop} {day} – {shift}: duration is {total_minutes/60:.2f} hours, but must be at least {MIN_REGULAR_SHIFT_DURATION_HOURS} hours.")
                    st.stop()
                if total_minutes > max_duration_minutes:
                    st.error(f"Invalid time configuration for {shop} {day} – {shift}: duration is {total_minutes/60:.2f} hours, but must be at greatest {MAX_REGULAR_SHIFT_DURATION_HOURS} hours.")
                    st.stop()
                total_quarters = total_minutes // MIN_TIME_UNIT
                if total_quarters > MIN_REGULAR_SHIFT_DURATION_HOURS*(60//MIN_TIME_UNIT):
                    shift_regular_time[(shop, day, shift)] = MIN_REGULAR_SHIFT_DURATION_HOURS*(60//MIN_TIME_UNIT)
                    shift_extra_time[(shop, day, shift)] = total_quarters - MIN_REGULAR_SHIFT_DURATION_HOURS*(60//MIN_TIME_UNIT)
                else:
                    shift_regular_time[(shop, day, shift)] = total_quarters
                    shift_extra_time[(shop, day, shift)] = 0
                min_workers_dict[(shop, day, shift)] = min_workers_override
                shift_forecasts[(shop, day, shift)] = DEFAULT_SHOP_SHIFT_FORECAST[shop][shift]
    
    worker_names_key = f"{week_key}_worker_names"
    worker_names_str = st.sidebar.text_input(
        "Worker Names (comma separated)",
        value=st.session_state.get(worker_names_key, "Chiara, Elisabetta, Erika, Claudia, Kety, Luana, Irene, Michela"),
        key=worker_names_key
    )
    worker_names = [w.strip() for w in worker_names_str.split(",") if w.strip()]
    
    st.sidebar.header("Worker Parameters")
    contract_hours = []
    conversion_rate = []
    extra_cost = []
    leave_requests = {}
    is_special_worker = []
    worker_allowed_shops = {}
    worker_roles = []
    SPECIAL_WORKER_COST_SCALE = 1000000
    default_normal_ec = 15
    for worker in worker_names:
        with st.sidebar.expander(f"{worker}", expanded=False):
            allowed_key = f"{week_key}_{worker}_shops"
            allowed = st.multiselect(
                f"Select shops where {worker} can work",
                options=shop_names,
                default=st.session_state.get(allowed_key, shop_names),
                key=allowed_key
            )
            worker_allowed_shops[worker] = allowed
            special_key = f"{week_key}_{worker}_special"
            special = st.checkbox(f"{worker} is a Special Worker", value=st.session_state.get(special_key, False), key=special_key)
            is_special_worker.append(special)
            default_contract = 40
            contract_key = f"{week_key}_{worker}_contract"
            ch = st.number_input(
                f"{worker} Contract Hours", value=st.session_state.get(contract_key, default_contract), min_value=0, step=1, key=contract_key
            )
            if not special:
                default_cr = 100
                default_ec = default_normal_ec
                conv_key = f"{week_key}_{worker}_conv"
                cost_key = f"{week_key}_{worker}_cost"
                cr = st.number_input(
                    f"{worker} Conversion Rate ($ per customer)", value=st.session_state.get(conv_key, default_cr), min_value=0, step=1, key=conv_key
                )
                ec = st.number_input(
                    f"{worker} Extra Cost ($ per extra hour)", value=st.session_state.get(cost_key, default_ec), min_value=0, step=1, key=cost_key
                )
            else:
                st.markdown("**Special Worker settings:** Conversion Rate is 0 and Extra Cost is scaled.")
                cr = 0
                ec = default_normal_ec * SPECIAL_WORKER_COST_SCALE
            leave = {}
            st.markdown("Leave Requests (independent of shop):")
            for d in DAY_NAMES:
                for s in SHIFT_NAMES:
                    leave_key = f"{week_key}_{worker}_leave_{d}_{s}"
                    leave[(d, s)] = st.toggle(
                        f"Leave on {d} {s}",
                        value=st.session_state.get(leave_key, False),
                        key=leave_key
                    )
            role_key = f"{week_key}_{worker}_role"
            role = st.radio(
                f"{worker} Role", 
                options=["Pharmacist", "Salesperson"],
                index=0 if st.session_state.get(role_key, "Pharmacist")=="Pharmacist" else 1,
                key=role_key
            )
            worker_roles.append(role)
            contract_hours.append(ch)
            conversion_rate.append(cr)
            extra_cost.append(ec)
            leave_requests[worker] = leave

    return (SHOP_NAMES, worker_names, SHIFT_NAMES, DAY_NAMES, shift_availability, 
            shift_regular_time, shift_extra_time, shift_forecasts, min_workers_dict, 
            contract_hours, conversion_rate, extra_cost, leave_requests, is_special_worker,
            worker_allowed_shops, worker_roles)

# -----------------------------
# Single Week Solver (Rolling Horizon)
# -----------------------------
@st.cache_data
def solve_schedule_week(week, parameters, worker_memory):
    (shop_names, worker_names, shift_names, day_names, shift_availability, 
     shift_regular_time, shift_extra_time, shift_forecasts, min_workers_dict, 
     contract_hours, conversion_rate, extra_cost, leave_requests, is_special_worker,
     worker_allowed_shops, worker_roles) = parameters
    
    num_workers = len(worker_names)
    model = cp_model.CpModel()
    
    # Decision variables: shifts[(n, shop, day, shift)]
    shifts = {}
    for n in range(num_workers):
        for shop in shop_names:
            for d, day in enumerate(day_names):
                for s in shift_names:
                    var = model.NewBoolVar(f"week{week}_shift_n{n}_{shop}_{day}_{s}")
                    # Enforce shop availability.
                    if not shift_availability.get((shop, day, s), False):
                        model.Add(var == 0)
                    # Worker-shop allowance.
                    if shop not in worker_allowed_shops[worker_names[n]]:
                        model.Add(var == 0)
                    # If on the first day of the week the worker has reached max consecutive days, force off.
                    if d == 0 and worker_memory.get(worker_names[n], {}).get("consecutive_days", 0) >= MAX_CONSECUTIVE_DAYS:
                        model.Add(var == 0)
                    shifts[(n, shop, d, s)] = var
    
    # Enforce leave requests.
    for n in range(num_workers):
        for shop in shop_names:
            for d, day in enumerate(day_names):
                for s in SHIFT_NAMES:
                    if leave_requests[worker_names[n]].get((day, s), False):
                        model.Add(shifts[(n, shop, d, s)] == 0)
    
    # Capacity constraints per shop shift.
    worker_count = {}
    for shop in shop_names:
        max_capacity = DEFAULT_SHOP_CAPACITY[shop]["max"]
        for d, day in enumerate(day_names):
            for s in SHIFT_NAMES:
                if shift_availability.get((shop, day, s), False):
                    min_workers = min_workers_dict[(shop, day, s)]
                    count_var = model.NewIntVar(min_workers, max_capacity, f"week{week}_worker_count_{shop}_{day}_{s}")
                    model.Add(count_var == sum(shifts[(n, shop, d, s)] for n in range(num_workers)))
                    worker_count[(shop, d, s)] = count_var
                else:
                    count_var = model.NewIntVar(0, 0, f"week{week}_worker_count_{shop}_{day}_{s}")
                    model.Add(count_var == 0)
                    worker_count[(shop, d, s)] = count_var
    
    # Each worker can work at most one shop per time slot.
    for n in range(num_workers):
        for d, day in enumerate(day_names):
            for s in SHIFT_NAMES:
                model.Add(sum(shifts[(n, shop, d, s)] for shop in shop_names) <= 1)
    
    # Enforce pharmacist >= salesperson in each shop/day/shift.
    for shop in shop_names:
        for d, day in enumerate(day_names):
            for s in SHIFT_NAMES:
                if shift_availability.get((shop, day, s), False):
                    pharmacist_count = sum(shifts[(n, shop, d, s)] for n in range(num_workers) if worker_roles[n] == "Pharmacist")
                    salesperson_count = sum(shifts[(n, shop, d, s)] for n in range(num_workers) if worker_roles[n] == "Salesperson")
                    model.Add(pharmacist_count >= salesperson_count)
    
    # Compute work time (in quarter-hour increments).
    regular_work_time = {}
    extra_work_time = {}
    for n in range(num_workers):
        reg_expr = []
        extra_expr = []
        for shop in shop_names:
            for d, day in enumerate(day_names):
                for s in SHIFT_NAMES:
                    if shift_availability.get((shop, day, s), False):
                        reg_expr.append(shift_regular_time[(shop, day, s)] * shifts[(n, shop, d, s)])
                        extra_expr.append(shift_extra_time[(shop, day, s)] * shifts[(n, shop, d, s)])
        regular_work_time[n] = sum(reg_expr)
        extra_work_time[n] = sum(extra_expr)
        contract_quarters = int(contract_hours[n] * 4)
        if not is_special_worker[n]:
            model.Add(regular_work_time[n] == contract_quarters)
        else:
            model.Add(regular_work_time[n] <= contract_quarters)
    
    # Maximum consecutive shifts constraint (within the week).
    total_slots = len(day_names) * len(SHIFT_NAMES)
    for n in range(num_workers):
        shift_slots = []
        for d, day in enumerate(day_names):
            for s in SHIFT_NAMES:
                shift_slots.append(sum(shifts[(n, shop, d, s)] for shop in shop_names))
        for i in range(total_slots - MAX_CONSECUTIVE_SHIFTS):
            model.Add(sum(shift_slots[i:i+MAX_CONSECUTIVE_SHIFTS+1]) <= MAX_CONSECUTIVE_SHIFTS)
    
    # Objective: Maximize revenue minus extra cost penalty.
    extra_cost_per_quarter = [c / 4 for c in extra_cost]
    revenue_terms = []
    for n in range(num_workers):
        for shop in shop_names:
            for d, day in enumerate(day_names):
                for s in SHIFT_NAMES:
                    revenue_terms.append(shift_forecasts[(shop, day, s)] * conversion_rate[n] * shifts[(n, shop, d, s)])
    revenue = sum(revenue_terms)
    extra_cost_penalty = sum(extra_cost_per_quarter[n] * extra_work_time[n] for n in range(num_workers))
    model.Maximize(revenue - extra_cost_penalty)
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    schedule = {shop: {} for shop in shop_names}
    worker_summary = []
    statistics = {"Revenue": solver.ObjectiveValue()}
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for shop in shop_names:
            for d, day in enumerate(day_names):
                if day not in schedule[shop]:
                    schedule[shop][day] = {}
                for s in SHIFT_NAMES:
                    if not shift_availability.get((shop, day, s), False):
                        continue
                    assigned = [worker_names[n] for n in range(num_workers) if solver.Value(shifts[(n, shop, d, s)]) == 1]
                    schedule[shop][day][s] = assigned
        for n in range(num_workers):
            reg_val = solver.Value(regular_work_time[n])
            extra_val = solver.Value(extra_work_time[n])
            summary = {
                "Worker": worker_names[n],
                "Role": worker_roles[n],
                "Contract Hours": contract_hours[n],
                "Regular Hours Worked": reg_val / 4.0,
                "Extra Hours Worked": extra_val / 4.0,
                "Total Hours Worked": (reg_val + extra_val) / 4.0
            }
            worker_summary.append(summary)
        
        # Update worker memory for the next week.
        for n, worker in enumerate(worker_names):
            daily_work = []
            for d, day in enumerate(day_names):
                worked = any(
                    solver.Value(shifts[(n, shop, d, s)]) == 1
                    for shop in shop_names
                    for s in SHIFT_NAMES
                )
                daily_work.append(1 if worked else 0)
            count = 0
            for val in reversed(daily_work):
                if val == 1:
                    count += 1
                else:
                    break
            worker_memory[worker] = worker_memory.get(worker, {"consecutive_days": 0, "saturday_shifts": 0})
            worker_memory[worker]["consecutive_days"] = count
            if "Saturday" in day_names:
                sat_index = day_names.index("Saturday")
                saturday_worked = any(
                    solver.Value(shifts[(n, shop, sat_index, s)]) == 1
                    for shop in shop_names
                    for s in SHIFT_NAMES
                )
                if saturday_worked:
                    worker_memory[worker]["saturday_shifts"] += 1
        return schedule, worker_summary, statistics, worker_memory
    else:
        st.error(f"No solution found for week {week+1}")
        return None, None, None, worker_memory

# -----------------------------
# Rendering Functions
# -----------------------------
@st.cache_data
def render_calendar_for_worker(schedule, day_names, day_labels, shift_names, shop_names, worker_names):
    def generate_worker_shift_color(index, total):
        hue = int(360 * index / total + 30)
        return f"hsl({hue}, 70%, 65%)"
    
    html = ['<div style="overflow-x: auto;">', '<table class="calendar-table">']
    header = '<tr><th class="calendar-header">Worker - Shop</th>'
    for label in day_labels:
        header += f'<th class="calendar-header">{label}</th>'
    header += '</tr>'
    html.append(header)
    
    for worker in worker_names:
        for shop in shop_names:
            row = f'<tr><td class="worker-name-cell">{worker} - {shop}</td>'
            for day in day_names:
                cell_html = ""
                for s_index, s in enumerate(shift_names):
                    assigned = schedule.get(shop, {}).get(day, {}).get(s, [])
                    if worker in assigned:
                        color = generate_worker_shift_color(s_index, len(shift_names))
                        cell_html += f'<div class="shift-pill" style="background-color: {color};">{s}</div>'
                    # Leave empty if not assigned.
                row += f'<td>{cell_html}</td>'
            row += '</tr>'
            html.append(row)
    html.append('</table></div>')
    return "\n".join(html)

@st.cache_data
def render_calendar_for_employer(schedule, day_names, day_labels, shift_names, shop, worker_names):
    def generate_pastel_color(i, total):
        hue = int(360 * i / total)
        return f"hsl({hue}, 60%, 65%)"
    worker_colors = {worker: generate_pastel_color(i, len(worker_names)) for i, worker in enumerate(worker_names)}
    
    html = ['<div style="overflow-x: auto;">', '<table class="calendar-table">', '<thead>', '<tr><th class="calendar-header"></th>']
    for label in day_labels:
        html.append(f'<th class="calendar-header">{label}</th>')
    html.append('</tr></thead><tbody>')
    for s in shift_names:
        row = f'<tr><td class="worker-name-cell">{s}</td>'
        for day in day_names:
            assigned = schedule.get(shop, {}).get(day, {}).get(s, [])
            cell_html = ""
            for worker in assigned:
                color = worker_colors.get(worker, "#607d8b")
                cell_html += f'<div class="shift-box" style="background-color: {color};">{worker}</div>'
            row += f'<td>{cell_html}</td>'
        row += '</tr>'
        html.append(row)
    html.append('</tbody></table></div>')
    return "\n".join(html)

# -----------------------------
# Main Application
# -----------------------------
def main():
    st.set_page_config(page_title="Rolling Horizon Multi-Week Scheduling", layout="wide")
    st.markdown('<meta name="viewport" content="width=device-width, initial-scale=1.0">', unsafe_allow_html=True)
    st.title("Rolling Horizon Multi-Week Shift Scheduling Optimization")
    st.markdown("This application computes the optimal schedule week by week, carrying over previous weeks’ assignments.")
    
    # -----------------------------
    # Session State Initialization
    # -----------------------------
    if "current_week" not in st.session_state:
        st.session_state.current_week = 0
    # Always show these inputs so they can be changed.
    max_weeks = st.sidebar.number_input(
        "Number of Planning Weeks", 
        min_value=1, max_value=6, 
        value=st.session_state.get("max_weeks", 4), step=1
    )
    st.session_state.max_weeks = max_weeks

    selected_date = st.sidebar.date_input(
        "Select first day", 
        value=st.session_state.get("selected_date", datetime.date.today()), 
        min_value=datetime.date.today()
    )
    st.session_state.selected_date = selected_date

    if "week_solutions" not in st.session_state:
        st.session_state.week_solutions = {}
    if "week_worker_summary" not in st.session_state:
        st.session_state.week_worker_summary = {}
    if "week_statistics" not in st.session_state:
        st.session_state.week_statistics = {}
    if "week_parameters" not in st.session_state:
        st.session_state.week_parameters = {}
    if "worker_memory" not in st.session_state:
        st.session_state.worker_memory = {}
    if "current_week_solved" not in st.session_state:
        st.session_state.current_week_solved = False

    # Calculate first Monday based on selected date.
    first_monday = st.session_state.selected_date - datetime.timedelta(days=st.session_state.selected_date.weekday())
    
    # -----------------------------
    # Parameter Input for Current Week
    # -----------------------------
    current_week = st.session_state.current_week
    week_key = f"week_{current_week}"
    st.sidebar.header(f"Parameters for Week {current_week+1}")
    # Render parameter UI.
    params = get_parameters(week_key=week_key)
    st.session_state.week_parameters[current_week] = params
    
    # If worker_memory is empty, initialize it using the current list of workers.
    if not st.session_state.worker_memory:
        st.session_state.worker_memory = {worker: {"consecutive_days": 0, "saturday_shifts": 0} for worker in params[1]}

    # -----------------------------
    # Solve Week Button
    # -----------------------------
    if st.button("Solve Week", key="solve_week_button"):
        schedule, worker_summary, statistics, updated_memory = solve_schedule_week(current_week, params, st.session_state.worker_memory)
        if schedule is not None:
            st.session_state.week_solutions[current_week] = schedule
            st.session_state.week_worker_summary[current_week] = worker_summary
            st.session_state.week_statistics[current_week] = statistics
            st.session_state.worker_memory = updated_memory
            st.session_state.current_week_solved = True

    # -----------------------------
    # Navigation Buttons: Previous / Next Week
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    def prev_week():
        if st.session_state.current_week > 0:
            st.session_state.current_week -= 1
            st.session_state.current_week_solved = True  # assume previous week is solved

    def next_week():
        if st.session_state.current_week_solved and st.session_state.current_week < st.session_state.max_weeks - 1:
            from_week = st.session_state.current_week
            st.session_state.current_week += 1
            to_week = st.session_state.current_week
            copy_week_parameters(from_week, to_week)
            st.session_state.current_week_solved = False

    with col1:
        st.button("<< Previous Week", on_click=prev_week, key="prev_week", disabled=(st.session_state.current_week == 0))
    with col3:
        st.button("Next Week >>", on_click=next_week, key="next_week", 
                  disabled=(not st.session_state.current_week_solved or st.session_state.current_week >= st.session_state.max_weeks - 1))

    # -----------------------------
    # Display Schedule for the Current Week
    # -----------------------------
    if st.session_state.current_week in st.session_state.week_solutions:
        current_schedule = st.session_state.week_solutions[st.session_state.current_week]
        week_start_date = first_monday + datetime.timedelta(weeks=st.session_state.current_week)
        week_end_date = week_start_date + datetime.timedelta(days=6)
        st.header(f"Week {st.session_state.current_week+1} Schedule: {week_start_date.strftime('%b %d, %Y')} - {week_end_date.strftime('%b %d, %Y')}")
        week_dates = [week_start_date + datetime.timedelta(days=i) for i in range(7)]
        day_labels = [f"{day} {date.strftime('%b %d')}" for day, date in zip(DAY_NAMES, week_dates)]
        
        st.markdown(CALENDAR_CSS, unsafe_allow_html=True)
        st.subheader("Worker Timetable")
        detailed_html = render_calendar_for_worker(
            current_schedule, 
            DAY_NAMES, 
            day_labels, 
            SHIFT_NAMES, 
            SHOP_NAMES, 
            st.session_state.week_parameters[st.session_state.current_week][1]
        )
        st.markdown(detailed_html, unsafe_allow_html=True)
        
        st.subheader("Shop Timetables")
        for shop in SHOP_NAMES:
            st.markdown(f"**{shop}**")
            employer_html = render_calendar_for_employer(
                current_schedule, 
                DAY_NAMES, 
                day_labels, 
                SHIFT_NAMES, 
                shop, 
                st.session_state.week_parameters[st.session_state.current_week][1]
            )
            st.markdown(employer_html, unsafe_allow_html=True)
        
        st.header("Worker Summary")
        st.table(pd.DataFrame(st.session_state.week_worker_summary[st.session_state.current_week]))

if __name__ == '__main__':
    main()
