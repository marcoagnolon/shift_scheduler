import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import datetime

# --- Global Constants ---
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
    "Piazza Milano": {"min": 2, "max": 6},
    "Piazza Aurora": {"min": 2, "max": 3}
}

# Customer forecasts remain fixed from defaults.
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
MAX_CONSECUTIVE_SHIFTS = 10  # Added constant for maximum consecutive shifts constraint

# CSS for calendar rendering
CALENDAR_CSS = """
<style>
.calendar-table {
    width: 100%;
    border-collapse: collapse;
}

/* Updated header style: dark background with white text */
.calendar-header {
    background-color: #333;
    color: #fff;
    text-align: center;
    padding: 8px;
    border: 1px solid #ddd;
}

/* Bold the worker names */
.worker-name-cell {
    font-weight: bold;
    padding: 8px;
    border: 1px solid #ddd;
}

/* Style for shift pills */
.shift-pill {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 12px;
    margin: 2px;
    color: #fff;
}

.morning-pill {
    background-color: #5F8B4C;
}

.afternoon-pill {
    background-color: #FF9A9A;
}

/* Style for shift boxes in employer view */
.shift-box {
    padding: 4px;
    border-radius: 4px;
    margin: 2px;
    color: #fff;
    display: inline-block;
}

/* Alternate row colors using two very similar shades of green */
.calendar-table tr:nth-child(odd) {
    background-color: #505A5B;
}
.calendar-table tr:nth-child(even) {
    background-color: #343F3E;
}
</style>
"""



##############################################
# Helper: Global Shop Settings (for all shifts)
##############################################
def get_shop_settings():
    with st.sidebar.expander("Global Shop Settings", expanded=True):
        st.subheader("Global Settings for Each Shift")
        global_shift_settings = {}
        for shift in SHIFT_NAMES:
            # Set defaults based on shift (using your default schedule)
            if shift.lower() == "morning":
                default_open = datetime.time(8, 30)
                default_close = datetime.time(12, 45)
            elif shift.lower() == "afternoon":
                default_open = datetime.time(15, 30)
                default_close = datetime.time(21, 0)
            else:
                default_open = datetime.time(8, 30)
                default_close = datetime.time(21, 0)
            # Use default min workers (here set to 2 as per your defaults)
            default_min_workers = 2
            st.markdown(f"### {shift} Shift Global Settings")
            shift_open_time = st.time_input(f"{shift} Opening Time", value=default_open, key=f"global_open_{shift}")
            shift_close_time = st.time_input(f"{shift} Closing Time", value=default_close, key=f"global_close_{shift}")
            shift_min_workers = st.number_input(f"{shift} Minimum Workers", value=default_min_workers, min_value=0, step=1, key=f"global_min_{shift}")
            global_shift_settings[shift] = {
                "open_time": shift_open_time,
                "close_time": shift_close_time,
                "min_workers": shift_min_workers
            }
        
        shop_shift_schedule_ui = {}
        shop_capacity_ui = {}
        # Apply the global settings for each shop.
        for shop in SHOP_NAMES:
            shop_shift_schedule_ui[shop] = {}
            shop_capacity_ui[shop] = {}
            for shift in SHIFT_NAMES:
                shop_shift_schedule_ui[shop][shift] = (
                    global_shift_settings[shift]["open_time"],
                    global_shift_settings[shift]["close_time"]
                )
                shop_capacity_ui[shop][shift] = global_shift_settings[shift]["min_workers"]
    return shop_shift_schedule_ui, shop_capacity_ui

def get_shop_settings_per_shop():
    shop_shift_schedule_ui = {}
    shop_capacity_ui = {}
    with st.sidebar.expander("Shop–Specific Global Settings", expanded=True):
        for shop in SHOP_NAMES:
            st.subheader(f"{shop} Global Settings")
            shop_shift_schedule_ui[shop] = {}
            shop_capacity_ui[shop] = {}
            for shift in SHIFT_NAMES:
                st.markdown(f"**{shift} Shift for {shop}**")
                # Use default values from the dictionaries:
                default_open, default_close = DEFAULT_SHOP_SHIFT_SCHEDULE[shop][shift]
                default_min_workers = DEFAULT_SHOP_CAPACITY[shop]["min"]
                open_time = st.time_input(f"{shop} {shift} Opening Time", value=default_open, key=f"{shop}_{shift}_open")
                close_time = st.time_input(f"{shop} {shift} Closing Time", value=default_close, key=f"{shop}_{shift}_close")
                min_workers = st.number_input(f"{shop} {shift} Minimum Workers", value=default_min_workers, min_value=0, step=1, key=f"{shop}_{shift}_min")
                shop_shift_schedule_ui[shop][shift] = (open_time, close_time)
                shop_capacity_ui[shop][shift] = min_workers
    return shop_shift_schedule_ui, shop_capacity_ui


##############################################
# Helper: Shop–Specific Day–Shift Overrides (Optional)
##############################################
def get_day_shift_overrides_per_shop(shop_capacity_ui):
    """
    Optionally override the global settings for specific shop/day–shift combinations.
    """
    day_shift_overrides = {}
    for shop in SHOP_NAMES:
        with st.sidebar.expander(f"{shop} - Day–Shift Overrides", expanded=False):
            st.markdown(f"Overrides for {shop}")
            for day in DAY_NAMES:
                st.markdown(f"**{day}**")
                for shift in SHIFT_NAMES:
                    override = st.checkbox(f"Override settings for {shop} {day} – {shift}?", key=f"{shop}_{day}_{shift}_override")
                    if override:
                        is_closed = st.checkbox(f"Mark {shop} {day} – {shift} as closed?", key=f"{shop}_{day}_{shift}_closed")
                        if is_closed:
                            day_shift_overrides[(shop, day, shift)] = {"closed": True}
                        else:
                            override_open = st.time_input(f"{shop} {day} – {shift} Opening Time", value=datetime.time(8, 30), key=f"{shop}_{day}_{shift}_open")
                            override_close = st.time_input(f"{shop} {day} – {shift} Closing Time", value=datetime.time(21, 0), key=f"{shop}_{day}_{shift}_close")
                            override_min_workers = st.number_input(
                                f"{shop} {day} – {shift} Minimum Workers",
                                value=shop_capacity_ui[shop][shift],
                                min_value=0, step=1,
                                key=f"{shop}_{day}_{shift}_min"
                            )
                            day_shift_overrides[(shop, day, shift)] = {
                                "closed": False,
                                "open_time": override_open,
                                "close_time": override_close,
                                "min_workers": override_min_workers
                            }
    return day_shift_overrides



##############################################
# Helper: Day–Shift Overrides (Optional)
##############################################
def get_day_shift_overrides():
    """
    Optionally override the global settings for specific day–shift combinations.
    You can either provide custom opening/closing times and minimum workers,
    or mark the day–shift as closed.
    """
    day_shift_overrides = {}
    with st.sidebar.expander("Day–Shift Overrides", expanded=True):
        st.markdown("Override global settings for specific day–shift combinations (if needed).")
        for day in DAY_NAMES:
            st.markdown(f"**{day}**")
            for shift in SHIFT_NAMES:
                override = st.checkbox(f"Override settings for {day} – {shift}?", key=f"override_{day}_{shift}")
                if override:
                    is_closed = st.checkbox(f"Mark {day} – {shift} as closed?", key=f"closed_{day}_{shift}")
                    if is_closed:
                        day_shift_overrides[(day, shift)] = {"closed": True}
                    else:
                        override_open = st.time_input(f"{day} – {shift} Opening Time", value=datetime.time(8, 30), key=f"{day}_{shift}_open")
                        override_close = st.time_input(f"{day} – {shift} Closing Time", value=datetime.time(21, 0), key=f"{day}_{shift}_close")
                        override_min_workers = st.number_input(f"{day} – {shift} Minimum Workers", value=2, min_value=0, step=1, key=f"{day}_{shift}_min")
                        day_shift_overrides[(day, shift)] = {
                            "closed": False,
                            "open_time": override_open,
                            "close_time": override_close,
                            "min_workers": override_min_workers
                        }
    return day_shift_overrides

##############################################
# Full get_parameters() Function (Updated)
##############################################
def get_parameters():
    shop_names = SHOP_NAMES
    day_names = DAY_NAMES

    # 1. Retrieve the shop-specific global settings.
    shop_shift_schedule_ui, shop_capacity_ui = get_shop_settings_per_shop()

    # 2. Retrieve any shop-specific day–shift overrides.
    day_shift_overrides = get_day_shift_overrides_per_shop(shop_capacity_ui)
    

    # Dictionaries to store the computed configuration.
    shift_availability = {}   # True if the shop/day/shift is active; False if closed.
    shift_regular_time = {}   # Regular work time in quarter-hour increments.
    shift_extra_time = {}     # Extra time beyond the regular period.
    min_workers_dict = {}     # Minimum required workers per shop/day/shift.
    shift_forecasts = {}      # Customer forecast per shop/day/shift.

    min_duration_minutes = MIN_REGULAR_SHIFT_DURATION_HOURS * 60

    # Loop over each shop, day, and shift.
    for shop in shop_names:
        for day in day_names:
            for shift in SHIFT_NAMES:
                # Check if an override exists for this shop/day–shift.
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
                    # Use shop-specific global settings if no override exists.
                    start_time, end_time = shop_shift_schedule_ui[shop][shift]
                    min_workers_override = shop_capacity_ui[shop][shift]
                    shift_availability[(shop, day, shift)] = True

                # Compute the shift duration (in minutes).
                dt_start = datetime.datetime.combine(datetime.date.today(), start_time)
                dt_end = datetime.datetime.combine(datetime.date.today(), end_time)
                if dt_end < dt_start:
                    dt_end += datetime.timedelta(days=1)
                total_minutes = (dt_end - dt_start).seconds // 60

                # Validate that the shift meets the minimum duration.
                if total_minutes < min_duration_minutes:
                    st.error(f"Invalid time configuration for {shop} {day} – {shift}: "
                             f"duration is {total_minutes/60:.2f} hours, but must be at least {MIN_REGULAR_SHIFT_DURATION_HOURS} hours.")
                    st.stop()

                total_quarters = total_minutes // 15
                if total_quarters > 16:
                    shift_regular_time[(shop, day, shift)] = 16
                    shift_extra_time[(shop, day, shift)] = total_quarters - 16
                else:
                    shift_regular_time[(shop, day, shift)] = total_quarters
                    shift_extra_time[(shop, day, shift)] = 0

                min_workers_dict[(shop, day, shift)] = min_workers_override
                # Use fixed customer forecast from defaults.
                shift_forecasts[(shop, day, shift)] = DEFAULT_SHOP_SHIFT_FORECAST[shop][shift]

    # --- Worker Parameters (Configured via the sidebar) ---
    worker_names_str = st.sidebar.text_input(
        "Worker Names (comma separated)",
        "Chiara, Elisabetta, Erika, Claudia, Kety, Luana, Irene, Michela"
    )
    worker_names = [w.strip() for w in worker_names_str.split(",") if w.strip()]

    st.sidebar.header("Worker Parameters")
    contract_hours = []
    conversion_rate = []
    extra_cost = []
    leave_requests = {}
    is_special_worker = []
    worker_allowed_shops = {}
    worker_roles = []  # New list to store the role for each worker

    # Define a scale factor for special worker extra cost.
    SPECIAL_WORKER_COST_SCALE = 1000000
    default_normal_ec = 15  # Normal worker default extra cost.

    for worker in worker_names:
        with st.sidebar.expander(f"{worker}"):
            allowed = st.multiselect(
                f"Select shops where {worker} can work",
                options=shop_names,
                default=shop_names,
                key=f"{worker}_shops"
            )
            worker_allowed_shops[worker] = allowed

            # For each worker...
            special = st.checkbox(f"{worker} is a Special Worker", value=False, key=f"{worker}_special")
            is_special_worker.append(special)

            default_contract = 40
            ch = st.number_input(
                f"{worker} Contract Hours", value=default_contract, min_value=0, step=1, key=f"{worker}_contract"
            )

            if not special:
                default_cr = 100
                default_ec = default_normal_ec
                cr = st.number_input(
                    f"{worker} Conversion Rate ($ per customer)", value=default_cr, min_value=0, step=1, key=f"{worker}_conv"
                )
                ec = st.number_input(
                    f"{worker} Extra Cost ($ per extra hour)", value=default_ec, min_value=0, step=1, key=f"{worker}_cost"
                )
            else:
                st.markdown("**Special Worker settings:** Conversion Rate and Extra Cost are set automatically.")
                # For special workers, conversion rate is fixed to 0 and extra cost is scaled.
                cr = 0
                ec = default_normal_ec * SPECIAL_WORKER_COST_SCALE

            leave = {}
            st.markdown("Leave Requests (independent of shop):")
            for d in day_names:
                for s in SHIFT_NAMES:
                    leave[(d, s)] = st.checkbox(
                        f"Leave on {d} {s}",
                        value=False,
                        key=f"{worker}_leave_{d}_{s}"
                    )
            # New role input for each worker.
            role = st.radio(f"{worker} Role", options=["Pharmacist", "Salesperson"], key=f"{worker}_role")
            worker_roles.append(role)

            contract_hours.append(ch)
            conversion_rate.append(cr)
            extra_cost.append(ec)
            leave_requests[worker] = leave

    return (shop_names, worker_names, SHIFT_NAMES, day_names, shift_availability, 
            shift_regular_time, shift_extra_time, shift_forecasts, min_workers_dict, 
            contract_hours, conversion_rate, extra_cost, leave_requests, is_special_worker,
            worker_allowed_shops, worker_roles)


##############################################
# Compute Next Week Off for Worker
##############################################
def compute_next_week_off_for_worker(worker, schedule, shift_availability, day_names, shift_names, max_consec, shop_names):
    """
    Computes the next forced off slot (day and shift) for the worker in the next week,
    based on the worker's current weekly assignments across all shops and the maximum
    allowed consecutive shifts (max_consec).

    Parameters:
      worker: The worker's name.
      schedule: A dictionary with structure schedule[shop][day][shift] = list of workers.
      shift_availability: A dictionary with keys (shop, day, shift) indicating if the shift is open.
      day_names: List of day names (e.g. ["Monday", "Tuesday", ...]).
      shift_names: List of shift names (e.g. ["Morning", "Afternoon"]).
      max_consec: Maximum allowed consecutive shifts (as individual shift slots).
      shop_names: List of shop names.

    Returns:
      A string with the day and shift (e.g., "Thursday Afternoon") when the worker is forced off next week,
      or a message if no forced rest occurs within a week.
    """
    # Build the worker's shift assignment list for the week (0 = off, 1 = work)
    shift_slots = []
    for day in day_names:
        for shift in shift_names:
            # Check if the shift is available in any shop
            available = any(shift_availability.get((shop, day, shift), False) for shop in shop_names)
            if available:
                # If available, mark the slot as worked if the worker is assigned in any shop
                worked = any(worker in schedule.get(shop, {}).get(day, {}).get(shift, []) for shop in shop_names)
                shift_slots.append(1 if worked else 0)
            else:
                # If the shift is closed in all shops, consider it off.
                shift_slots.append(0)
    
    # Count the consecutive worked slots (1's) from the end of the week.
    L = 0
    for val in reversed(shift_slots):
        if val == 1:
            L += 1
        else:
            break

    # Determine the (1-indexed) position of the forced off shift in next week.
    next_week_off_slot = max_consec - L + 1

    # Total number of shift slots per week.
    slots_per_week = len(day_names) * len(shift_names)
    if next_week_off_slot > slots_per_week:
        return "No forced rest within week"
    
    # Convert next_week_off_slot (1-indexed) to 0-indexed slot index.
    slot_index = next_week_off_slot - 1
    next_day_index = slot_index // len(shift_names)
    next_shift_index = slot_index % len(shift_names)
    next_day = day_names[next_day_index]
    next_shift = shift_names[next_shift_index]
    return f"{next_day} {next_shift}"

##############################################
# Solve Scheduling Problem
##############################################
def solve_schedule(shop_names, worker_names, worker_roles, shift_names, day_names, shift_availability, 
                   shift_regular_time, shift_extra_time, shift_forecasts, min_workers_dict, 
                   contract_hours, conversion_rate, extra_cost, leave_requests, is_special_worker,
                   worker_allowed_shops):
    num_workers = len(worker_names)
    model = cp_model.CpModel()
    
    # Decision variables: shifts[(n, shop, d, s)] = 1 if worker n works in shop on day d, shift s.
    shifts = {}
    for n in range(num_workers):
        for shop in shop_names:
            for d, day in enumerate(day_names):
                for s in shift_names:
                    var = model.NewBoolVar(f"shift_n{n}_{shop}_{day}_{s}")
                    # If the shop's shift is closed, force to 0.
                    if not shift_availability.get((shop, day, s), False):
                        model.Add(var == 0)
                    # If worker is not allowed in this shop, force to 0.
                    if shop not in worker_allowed_shops[worker_names[n]]:
                        model.Add(var == 0)
                    shifts[(n, shop, d, s)] = var
    
    # Enforce leave requests (independent of shop).
    for n in range(num_workers):
        for shop in shop_names:
            for d, day in enumerate(day_names):
                for s in shift_names:
                    if leave_requests[worker_names[n]].get((day, s), False):
                        model.Add(shifts[(n, shop, d, s)] == 0)
    
    # Enforce capacity constraints per shop shift.
    worker_count = {}
    for shop in shop_names:
        # Get the maximum capacity for this shop from default constants.
        max_capacity = DEFAULT_SHOP_CAPACITY[shop]["max"]
        for d, day in enumerate(day_names):
            for s in shift_names:
                if shift_availability.get((shop, day, s), False):
                    min_workers = min_workers_dict[(shop, day, s)]
                    count_var = model.NewIntVar(min_workers, max_capacity, f"worker_count_{shop}_{day}_{s}")
                    model.Add(count_var == sum(shifts[(n, shop, d, s)] for n in range(num_workers)))
                    worker_count[(shop, d, s)] = count_var
                else:
                    count_var = model.NewIntVar(0, 0, f"worker_count_{shop}_{day}_{s}")
                    model.Add(count_var == 0)
                    worker_count[(shop, d, s)] = count_var
    
    # Each worker can work at most one shop per time slot (day, shift).
    for n in range(num_workers):
        for d, day in enumerate(day_names):
            for s in shift_names:
                model.Add(sum(shifts[(n, shop, d, s)] for shop in shop_names) <= 1)
    
    # Enforce pharmacist >= salesperson constraint for each shop/day/shift.
    for shop in shop_names:
        for d, day in enumerate(day_names):
            for s in shift_names:
                if shift_availability.get((shop, day, s), False):
                    pharmacist_count = sum(shifts[(n, shop, d, s)] for n in range(num_workers) if worker_roles[n] == "Pharmacist")
                    salesperson_count = sum(shifts[(n, shop, d, s)] for n in range(num_workers) if worker_roles[n] == "Salesperson")
                    model.Add(pharmacist_count >= salesperson_count)
    
    # Compute regular and extra work time for each worker (over all shops, in quarter increments).
    regular_work_time = {}
    extra_work_time = {}
    for n in range(num_workers):
        reg_expr = []
        extra_expr = []
        for shop in shop_names:
            for d, day in enumerate(day_names):
                for s in shift_names:
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
    
    # Maximum consecutive shifts constraint (across day/shift slots regardless of shop).
    total_slots = len(day_names) * len(shift_names)
    for n in range(num_workers):
        shift_slots = []
        for d, day in enumerate(day_names):
            for s in shift_names:
                shift_slots.append(sum(shifts[(n, shop, d, s)] for shop in shop_names))
        for i in range(total_slots - MAX_CONSECUTIVE_SHIFTS):
            model.Add(sum(shift_slots[i:i+MAX_CONSECUTIVE_SHIFTS+1]) <= MAX_CONSECUTIVE_SHIFTS)
    
    # Revenue and extra cost objective.
    extra_cost_per_quarter = [c / 4 for c in extra_cost]
    revenue_terms = []
    for n in range(num_workers):
        for shop in shop_names:
            for d, day in enumerate(day_names):
                for s in shift_names:
                    revenue_terms.append(shift_forecasts[(shop, day, s)] * conversion_rate[n] * shifts[(n, shop, d, s)])
    revenue = sum(revenue_terms)
    extra_cost_penalty = sum(extra_cost_per_quarter[n] * extra_work_time[n] for n in range(num_workers))
    model.Maximize(revenue - extra_cost_penalty)
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # Build schedule dictionary: schedule[shop][day][shift] = list of workers assigned.
    schedule = {shop: {} for shop in shop_names}
    worker_summary = []
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
            # next_possible_off_day = compute_next_possible_off_day(
            #     worker_names[n],
            #     schedule,
            #     day_names,
            #     SHIFT_NAMES,
            #     shop_names,
            #     MAX_CONSECUTIVE_SHIFTS
            # )
            # summary["Next Possible Off Day"] = next_possible_off_day

            worker_summary.append(summary)
        
        statistics = {"Revenue": solver.ObjectiveValue()}
        return schedule, worker_summary, statistics
    else:
        return None, None, None


# def compute_consecutive_shifts_at_end(worker, schedule, day_names, shift_names, shop_names):
#     """
#     Count consecutive shift slots (ordered by day then shift)
#     at the end of the current week where the worker was assigned.
#     """
#     shift_slots = []
#     # Create a flattened list of shift slots for the week.
#     for day in day_names:
#         for s in shift_names:
#             # The worker is considered "working" in this slot if assigned in any shop.
#             works = any(worker in schedule.get(shop, {}).get(day, {}).get(s, [])
#                         for shop in shop_names)
#             shift_slots.append(1 if works else 0)
#     # Count consecutive worked slots from the end.
#     count = 0
#     for worked in reversed(shift_slots):
#         if worked:
#             count += 1
#         else:
#             break
#     return count

# def compute_next_possible_off_day(worker, schedule, day_names, shift_names, shop_names, MAX_CONSECUTIVE_SHIFTS):
#     """
#     Determines the next day–shift (as a tuple) in the following week on which
#     the worker would be forced to have an off slot because they've reached
#     the maximum consecutive shifts limit.

#     The function calculates the number of consecutive shift slots (ordered as
#     day-shift pairs) worked at the end of the current week. Then, based on the
#     remaining shift slots allowed (MAX_CONSECUTIVE_SHIFTS minus the consecutive
#     shifts already worked), it computes the index of the next forced off slot in
#     the next week. This index is then mapped to the corresponding day and shift.
    
#     For example, if MAX_CONSECUTIVE_SHIFTS = 10 and there are 2 shifts per day,
#     a worker who worked 3 consecutive shift slots at the end of the week can work
#     7 additional slots. The forced off slot would then be the 8th slot of next week,
#     which (with 2 shifts per day) corresponds to Thursday Afternoon.
#     """
#     # Get the number of consecutive shift slots worked at the end of the current week.
#     consecutive_slots = compute_consecutive_shifts_at_end(worker, schedule, day_names, shift_names, shop_names)
#     # Determine how many additional shift slots the worker can work before a forced off.
#     allowed_additional = MAX_CONSECUTIVE_SHIFTS - consecutive_slots
    
#     # If already at or above the limit, the next off slot is the first slot of next week.
#     if allowed_additional <= 0:
#         return day_names[0], shift_names[0]
    
#     # The forced off slot in next week is at index 'allowed_additional'
#     forced_index = allowed_additional  # 0-indexed position among the shift slots in next week.
#     shifts_per_day = len(shift_names)
#     forced_day_index = forced_index // shifts_per_day
#     forced_shift_index = forced_index % shifts_per_day
    
#     # Cap indices if they extend beyond the week.
#     if forced_day_index >= len(day_names):
#         forced_day_index = len(day_names) - 1
#         forced_shift_index = len(shift_names) - 1

#     return f"{day_names[forced_day_index]} {shift_names[forced_shift_index]}", 




##############################################
# Render Calendars for Worker and Employer
##############################################
def render_calendar_for_worker(schedule, day_names, shift_names, shop_names, worker_names):
    """
    Detailed worker timetable:
    - Each row is a worker–shop combination.
    - Columns are days.
    - In each cell, if a worker is assigned to a shop on that day, display a pill (colored by shift name).
    """
    html = ['<table class="calendar-table">']
    # Header row.
    header = '<tr><th class="calendar-header">Worker - Shop</th>'
    for day in day_names:
        header += f'<th class="calendar-header">{day}</th>'
    header += '</tr>'
    html.append(header)
    
    for worker in worker_names:
        for shop in shop_names:
            row = f'<tr><td class="worker-name-cell">{worker} - {shop}</td>'
            for day in day_names:
                cell_html = ""
                for s in shift_names:
                    assigned = schedule.get(shop, {}).get(day, {}).get(s, [])
                    if worker in assigned:
                        css_class = "morning-pill" if s.lower() == "morning" else "afternoon-pill" if s.lower() == "afternoon" else "shift-pill"
                        cell_html += f'<div class="shift-pill {css_class}">{s}</div>'
                row += f'<td>{cell_html}</td>'
            row += '</tr>'
            html.append(row)
    html.append('</table>')
    return "\n".join(html)

def render_calendar_for_employer(schedule, day_names, shift_names, shop, worker_names):
    """
    Employer view for a specific shop:
    - Rows: shifts
    - Columns: days
    - Each cell shows colored boxes for each worker assigned.
    """
    def generate_color(i, total):
        hue = int(360 * i / total)
        return f"hsl({hue}, 70%, 50%)"
    worker_colors = {worker: generate_color(i, len(worker_names)) for i, worker in enumerate(worker_names)}
    
    html = ['<table class="calendar-table">', '<thead>', '<tr><th class="calendar-header"></th>']
    for day in day_names:
        html.append(f'<th class="calendar-header">{day}</th>')
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
    html.append('</tbody></table>')
    return "\n".join(html)

##############################################
# Main Application
##############################################
def main():
    st.set_page_config(page_title="Multi-Shop Shift Scheduling Optimization", layout="wide")
    st.title("Multi-Shop Shift Scheduling Optimization")
    st.markdown(
        """
        This platform computes an optimal shift schedule for multiple shops.
        Shop parameters (opening/closing times, capacity, and customer forecasts) 
        can now be configured via the web interface.
        Click **Solve Schedule** to view detailed worker assignments, shop timetables, and solver statistics.
        """
    )
    
    (shop_names, worker_names, shift_names, day_names, shift_availability, 
     shift_regular_time, shift_extra_time, shift_forecasts, min_workers_dict, 
     contract_hours, conversion_rate, extra_cost, leave_requests, is_special_worker,
     worker_allowed_shops, worker_roles) = get_parameters()
    
    if st.button("Solve Schedule"):
        with st.spinner("Solving the optimization model..."):
            schedule, worker_summary, statistics = solve_schedule(
                shop_names, worker_names, worker_roles, shift_names, day_names, shift_availability, 
                shift_regular_time, shift_extra_time, shift_forecasts, min_workers_dict, 
                contract_hours, conversion_rate, extra_cost, leave_requests, is_special_worker,
                worker_allowed_shops
            )
        if schedule:
            st.success("Optimal schedule found!")
            st.markdown(CALENDAR_CSS, unsafe_allow_html=True)
            
            st.header("Worker Timetable for employer (Rows: Worker - Shop)")
            detailed_html = render_calendar_for_worker(schedule, day_names, shift_names, shop_names, worker_names)
            st.markdown(detailed_html, unsafe_allow_html=True)
            
            st.header("Shop Timetables for shifts")
            for shop in shop_names:
                st.subheader(f"Employer Calendar View for {shop}")
                employer_html = render_calendar_for_employer(schedule, day_names, shift_names, shop, worker_names)
                st.markdown(employer_html, unsafe_allow_html=True)
                
            st.header("Worker Summary")
            st.table(pd.DataFrame(worker_summary))
            
            st.header("Solver Statistics")
            stat_df = pd.DataFrame(list(statistics.items()), columns=["Statistic", "Value"])
            st.table(stat_df)
        else:
            st.error("No optimal solution found!")

if __name__ == '__main__':
    main()
