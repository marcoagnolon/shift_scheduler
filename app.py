import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import datetime

def get_parameters():
    st.sidebar.header("Global Model Parameters")
    
    st.sidebar.subheader("Shift Configuration")
    # --- Worker Names ---
    worker_names_str = st.sidebar.text_input(
        "Worker Names (comma separated)",
        "Chiara, Elisabetta, Erika, Claudia, Kety, Luana"
    )
    worker_names = [w.strip() for w in worker_names_str.split(",") if w.strip()]
    
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    shift_names = ["Morning", "Afternoon"]

    # Default opening times for each shift: set to a 4-hour period.
    default_times = {
        "Morning": (datetime.time(8, 30), datetime.time(12, 45)),
        "Afternoon": (datetime.time(15, 30), datetime.time(21, 00))
    }

    # Dictionaries to store shift configurations
    shift_availability = {}
    shift_regular_time = {}
    shift_extra_time = {}
    # New dictionary: per shift minimum workers (default = 4)
    min_workers_dict = {}

    with st.sidebar.expander("Shift Scheduler and Configuration"):
        for day in day_names:
            st.markdown(f"### {day}")
            for shift in shift_names:
                # Set default availability: closed on Saturday Afternoon and all Sunday shifts.
                default_val = True
                if (day == "Saturday" and shift == "Afternoon") or (day == "Sunday"):
                    default_val = False
                
                # Shift Availability Checkbox
                availability = st.checkbox(f"{shift} Shift Open", value=default_val, key=f"{day}_{shift}_open")
                shift_availability[(day, shift)] = availability

                # New: Minimum workers required for this shift.
                if availability:
                    min_workers = st.number_input(
                        f"Minimum workers for {day} {shift}",
                        value=4, min_value=0, step=1, key=f"{day}_{shift}_min"
                    )
                else:
                    min_workers = 0
                min_workers_dict[(day, shift)] = min_workers

                st.markdown(f"**{shift} Shift Time Configuration**")
                # Let the user choose the opening hours by selecting a start and end time.
                start_time = st.time_input(
                    f"{day} {shift} Start Time",
                    value=default_times[shift][0],
                    key=f"{day}_{shift}_start"
                )
                end_time = st.time_input(
                    f"{day} {shift} End Time",
                    value=default_times[shift][1],
                    key=f"{day}_{shift}_end"
                )
                dt_start = datetime.datetime.combine(datetime.date.today(), start_time)
                dt_end = datetime.datetime.combine(datetime.date.today(), end_time)
                if dt_end < dt_start:
                    dt_end += datetime.timedelta(days=1)
                total_minutes = (dt_end - dt_start).seconds // 60
                total_quarters = total_minutes // 15

                # If the shift duration is more than 4 hours (16 quarters),
                # use 4 hours as regular time and the remainder as extra time.
                if total_quarters > 16:
                    shift_regular_time[(day, shift)] = 16
                    shift_extra_time[(day, shift)] = total_quarters - 16
                else:
                    shift_regular_time[(day, shift)] = total_quarters
                    shift_extra_time[(day, shift)] = 0

    st.sidebar.subheader("Customer Count")
    # --- Shift Forecasts (in a collapsible expander) ---
    with st.sidebar.expander("Shift Forecasts (Customer Count)"):
        shift_forecasts = {}
        for day in day_names:
            for shift in shift_names:
                if shift_availability[(day, shift)]:
                    default_forecast = 50 if shift == "Morning" else 60
                    forecast = st.number_input(
                        f"{day} {shift} Forecast",
                        value=default_forecast, min_value=0, key=f"{day}_{shift}_forecast"
                    )
                else:
                    st.markdown(f"{day} {shift} Forecast: N/A (Shift Closed)")
                    forecast = 0
                shift_forecasts[(day, shift)] = forecast
    
    # --- Worker Parameters ---
    st.sidebar.header("Worker Parameters")
    contract_hours = []
    conversion_rate = []
    extra_cost = []
    max_morning_shifts = []
    max_afternoon_shifts = []
    worker_max_extra_hours = []  # Worker-specific maximum extra hours allowed (in hours)
    leave_requests = []  # Each worker gets a dictionary: keys (day, shift) -> bool.
    st.sidebar.markdown("Set the following for each worker:")
    for worker in worker_names:
        with st.sidebar.expander(f"{worker}"):
            ch = st.number_input(
                f"{worker} Contract Hours", value=36, min_value=0, step=1, key=f"{worker}_contract"
            )
            cr = st.number_input(
                f"{worker} Conversion Rate ($ per customer)", value=100, min_value=0, step=1, key=f"{worker}_conv"
            )
            ec = st.number_input(
                f"{worker} Extra Cost ($ per extra hour)", value=15, min_value=0, step=1, key=f"{worker}_cost"
            )
            # Maximum shifts allowed.
            morning_open_count = sum(1 for day in day_names if shift_availability[(day, "Morning")])
            afternoon_open_count = sum(1 for day in day_names if shift_availability[(day, "Afternoon")])
            max_am = st.number_input(
                f"{worker} Max Morning Shifts", value=morning_open_count, min_value=0, max_value=morning_open_count, step=1, key=f"{worker}_max_am"
            )
            max_pm = st.number_input(
                f"{worker} Max Afternoon Shifts", value=afternoon_open_count, min_value=0, max_value=afternoon_open_count, step=1, key=f"{worker}_max_pm"
            )
            # Worker-specific maximum extra hours allowed.
            max_extra = st.number_input(
                f"{worker} Maximum Extra Hours", value=10.0, min_value=0.0, step=0.5, key=f"{worker}_max_extra"
            )
            # Leave requests: allow marking days/shifts the worker is unavailable.
            leave = {}
            st.markdown("Leave Requests:")
            for day in day_names:
                for shift in shift_names:
                    leave[(day, shift)] = st.checkbox(
                        f"Leave on {day} {shift}", value=False, key=f"{worker}_leave_{day}_{shift}"
                    )
            contract_hours.append(ch)
            conversion_rate.append(cr)
            extra_cost.append(ec)
            max_morning_shifts.append(max_am)
            max_afternoon_shifts.append(max_pm)
            worker_max_extra_hours.append(max_extra)
            leave_requests.append(leave)

    # Return the new min_workers_dict variable along with the other parameters.
    return (worker_names, shift_availability, shift_regular_time, shift_extra_time, shift_forecasts, 
            contract_hours, conversion_rate, extra_cost, max_morning_shifts, max_afternoon_shifts, 
            leave_requests, worker_max_extra_hours, min_workers_dict)


def solve_schedule(worker_names, shift_availability, shift_regular_time, shift_extra_time, shift_forecasts, contract_hours, 
                   conversion_rate, extra_cost, max_morning_shifts, max_afternoon_shifts, leave_requests, worker_max_extra_hours,
                   min_workers_dict):
    num_workers = len(worker_names)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    shift_names = ["Morning", "Afternoon"]
    num_days = len(day_names)
    
    model = cp_model.CpModel()
    
    # Decision variables: shifts[(n, d, s)] = 1 if worker n works on day d, shift s.
    shifts = {}
    for n in range(num_workers):
        for d in range(num_days):
            for s in shift_names:
                if not shift_availability[(day_names[d], s)]:
                    shifts[(n, d, s)] = model.NewBoolVar(f"shift_n{n}_d{d}_{s}")
                    model.Add(shifts[(n, d, s)] == 0)
                else:
                    shifts[(n, d, s)] = model.NewBoolVar(f"shift_n{n}_d{d}_{s}")
    
    # Enforce leave requests for each worker.
    for n in range(num_workers):
        for d in range(num_days):
            for s in shift_names:
                if leave_requests[n][(day_names[d], s)]:
                    model.Add(shifts[(n, d, s)] == 0)
    
    # For each open shift, enforce a minimum number of workers using the per-shift min_workers.
    worker_count = {}
    for d in range(num_days):
        for s in shift_names:
            if shift_availability[(day_names[d], s)]:
                min_workers = min_workers_dict[(day_names[d], s)]
                worker_count[(d, s)] = model.NewIntVar(min_workers, num_workers, f"worker_count_d{d}_{s}")
                model.Add(worker_count[(d, s)] == sum(shifts[(n, d, s)] for n in range(num_workers)))
            else:
                worker_count[(d, s)] = model.NewIntVar(0, 0, f"worker_count_d{d}_{s}")
                model.Add(worker_count[(d, s)] == 0)
    
    # Compute regular and extra work time for each worker (in quarter increments).
    regular_work_time = {}
    extra_work_time = {}
    for n in range(num_workers):
        reg_time_expr = []
        extra_time_expr = []
        for d in range(num_days):
            for s in shift_names:
                if shift_availability[(day_names[d], s)]:
                    reg_time_expr.append(shift_regular_time[(day_names[d], s)] * shifts[(n, d, s)])
                    extra_time_expr.append(shift_extra_time[(day_names[d], s)] * shifts[(n, d, s)])
        regular_work_time[n] = sum(reg_time_expr)
        extra_work_time[n] = sum(extra_time_expr)
        
        # Ensure the worker meets contract hours using regular time.
        contract_quarters = int(contract_hours[n] * 4)
        model.Add(regular_work_time[n] >= contract_quarters)
        
        # Calculate the total extra time as the sum of:
        # 1) Any regular work done beyond the contract hours.
        # 2) The extra work time from extra time shifts.
        # Enforce that this sum does not exceed the worker's maximum extra hours.
        extra_allowable = int(worker_max_extra_hours[n] * 4)
        model.Add((regular_work_time[n] - contract_quarters) + extra_work_time[n] <= extra_allowable)
    
    # Revenue (per customer conversion) minus extra cost penalty.
    revenue = sum(
        shift_forecasts[(day_names[d], s)] * conversion_rate[n] * shifts[(n, d, s)]
        for n in range(num_workers) for d in range(num_days) for s in shift_names
    )
    extra_cost_per_increment = [c / 4 for c in extra_cost]  # cost per quarter increment.
    extra_cost_penalty = sum(
        extra_cost_per_increment[n] * extra_work_time[n] for n in range(num_workers)
    )
    model.Maximize(revenue - extra_cost_penalty)
    
    # Solve the model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    schedule = {}
    worker_summary = []
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Build schedule dictionary: schedule[day] = { "Morning": [workers], "Afternoon": [workers] }
        for d in range(num_days):
            day_schedule = {}
            for s in shift_names:
                if not shift_availability[(day_names[d], s)]:
                    continue  # Skip closed shifts.
                assigned_workers = [worker_names[n] for n in range(num_workers) if solver.Value(shifts[(n, d, s)]) == 1]
                day_schedule[s] = assigned_workers
            schedule[day_names[d]] = day_schedule
        
        # Build worker summary.
        for n in range(num_workers):
            contract_quarters = int(contract_hours[n] * 4)
            reg_increments = solver.Value(regular_work_time[n])
            extra_increments = solver.Value(extra_work_time[n])
            # Compute overtime as the regular work beyond contract plus extra shift time.
            overtime_quarters = max(reg_increments - contract_quarters, 0) + extra_increments
            worker_summary.append({
                "Worker": worker_names[n],
                "Contract Hours": contract_hours[n],
                "Regular Hours Worked": reg_increments / 4.0,
                "Extra Hours Worked": overtime_quarters / 4.0,
                "Total Hours Worked": reg_increments / 4.0 + overtime_quarters / 4.0
            })
        
        statistics = {
            "Revenue": solver.ObjectiveValue(),
        }
        return schedule, worker_summary, statistics
    else:
        return None, None, None

def create_timetable(schedule):
    """
    Create a timetable DataFrame with rows for shifts and columns for days.
    Each cell shows the list of assigned workers (or 'N/A' if the shift is closed).
    """
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    shift_names = ["Morning", "Afternoon"]
    
    timetable_data = {day: [] for day in day_names}
    
    for day in day_names:
        day_sched = schedule.get(day, {})
        for shift in shift_names:
            if shift in day_sched:
                timetable_data[day].append(", ".join(day_sched[shift]) if day_sched[shift] else "")
            else:
                timetable_data[day].append("N/A")
    
    timetable_df = pd.DataFrame(timetable_data, index=shift_names)
    return timetable_df

def render_calendar_for_worker(schedule, day_names, shift_names, worker_names):
    """
    Build an HTML table with:
    - Rows = workers
    - Columns = days
    - Each cell has zero or more 'pills' (one per assigned shift).
    """
    worker_day_assignments = {
        worker: {day: [] for day in day_names} for worker in worker_names
    }
    for day in day_names:
        if day not in schedule:
            continue
        for shift in shift_names:
            assigned_workers = schedule[day].get(shift, [])
            for w in assigned_workers:
                worker_day_assignments[w][day].append(shift)
    
    html = []
    html.append('<table class="calendar-table">')
    html.append('<thead>')
    html.append('<tr>')
    html.append('<th class="calendar-header"></th>')
    for day in day_names:
        html.append(f'<th class="calendar-header">{day}</th>')
    html.append('</tr>')
    html.append('</thead>')
    html.append('<tbody>')
    for worker in worker_names:
        html.append('<tr>')
        html.append(f'<td class="worker-name-cell">{worker}</td>')
        for day in day_names:
            shifts_for_day = worker_day_assignments[worker][day]
            cell_html = ""
            for shift in shifts_for_day:
                pill_class = "morning-pill" if shift == "Morning" else "afternoon-pill"
                cell_html += f'<div class="shift-pill {pill_class}">{shift}</div>'
            html.append(f'<td>{cell_html}</td>')
        html.append('</tr>')
    html.append('</tbody>')
    html.append('</table>')
    return "\n".join(html)

def render_calendar_for_employer(schedule, day_names, shift_names, worker_names):
    """
    Build an HTML table with:
    - Rows = shifts (Morning, Afternoon)
    - Columns = days
    - Each cell displays a pill for each assigned worker with a unique color.
    """
    def generate_color(i, total):
        hue = int(360 * i / total)
        return f"hsl({hue}, 70%, 50%)"
    
    worker_colors = {worker: generate_color(i, len(worker_names)) for i, worker in enumerate(worker_names)}
    
    html = []
    html.append('<table class="calendar-table">')
    html.append('<thead>')
    html.append('<tr>')
    html.append('<th class="calendar-header"></th>')
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        html.append(f'<th class="calendar-header">{day}</th>')
    html.append('</tr>')
    html.append('</thead>')
    html.append('<tbody>')
    for shift in shift_names:
        html.append('<tr>')
        html.append(f'<td class="worker-name-cell">{shift}</td>')
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            assigned_workers = schedule.get(day, {}).get(shift, [])
            if assigned_workers:
                pills_html = ""
                for worker in assigned_workers:
                    color = worker_colors.get(worker, "#607d8b")
                    pill_html = f'<div class="shift-box" style="background-color: {color};">{worker}</div>'
                    pills_html += pill_html
                html.append(f'<td>{pills_html}</td>')
            else:
                html.append('<td></td>')
        html.append('</tr>')
    html.append('</tbody>')
    html.append('</table>')
    return "\n".join(html)

CALENDAR_CSS = """
<style>
.calendar-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1rem;
    table-layout: fixed;
}
.calendar-table th, .calendar-table td {
    border: 1px solid #ddd;
    padding: 0.5rem;
    vertical-align: top;
    text-align: center;
    color: #333;
}
.calendar-header {
    background-color: #f5f5f5;
    font-weight: bold;
    color: #333;
}
.worker-name-cell {
    background-color: #fafafa;
    font-weight: bold;
    text-align: left;
    color: #333;
    width: 130px;
}
.shift-pill {
    display: inline-block;
    padding: 0.3rem 0.6rem;
    margin: 0.2rem;
    border-radius: 15px;
    color: #fff;
    font-size: 0.85rem;
    white-space: nowrap;
}
.morning-pill {
    background-color: #009688;
}
.afternoon-pill {
    background-color: #ff9800;
}
.shift-box {
    display: inline-block;
    padding: 0.4rem 0.8rem;
    border-radius: 15px;
    background-color: #607d8b;
    color: #fff;
    font-size: 0.85rem;
    margin: 0.2rem;
    white-space: nowrap;
}
</style>
"""

def main():
    st.set_page_config(page_title="Shift Scheduling Optimization", layout="wide")
    st.title("Shop Shift Scheduling Optimization")
    st.markdown(
        """
        This interactive web platform uses **Google OR-Tools** to compute an optimal shift schedule for your shop.
        Adjust the global settings (shift availability, time configuration, and worker parameters) as well as each worker’s hard constraints (leaves and max shifts).
        Once you click **Solve Schedule**, the model is solved and the timetable, worker summary, and solver statistics are displayed.
        """
    )
    
    # Get all parameters from the sidebar.
    (worker_names, shift_availability, shift_regular_time, shift_extra_time, shift_forecasts, 
     contract_hours, conversion_rate, extra_cost, max_morning_shifts, max_afternoon_shifts, 
     leave_requests, worker_max_extra_hours, min_workers_dict) = get_parameters()
    
    if st.button("Solve Schedule"):
        with st.spinner("Solving the optimization model..."):
            schedule, worker_summary, statistics = solve_schedule(
                worker_names, shift_availability, shift_regular_time, shift_extra_time, shift_forecasts, contract_hours,
                conversion_rate, extra_cost, max_morning_shifts, max_afternoon_shifts, leave_requests, worker_max_extra_hours,
                min_workers_dict
            )
        if schedule:
            st.success("Optimal schedule found!")
            
            # Inject CSS.
            st.markdown(CALENDAR_CSS, unsafe_allow_html=True)
            
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            shift_names = ["Morning", "Afternoon"]
            st.header("Calendar View (Workers as Rows)")
            calendar_html_v1 = render_calendar_for_worker(schedule, day_names, shift_names, worker_names)
            st.markdown(calendar_html_v1, unsafe_allow_html=True)
            
            st.header("Calendar View (Shifts as Rows)")
            calendar_html_v2 = render_calendar_for_employer(schedule, day_names, shift_names, worker_names)
            st.markdown(calendar_html_v2, unsafe_allow_html=True)
            
            st.header("Worker Summary")
            st.table(pd.DataFrame(worker_summary))
            
            st.header("Solver Statistics")
            stat_df = pd.DataFrame(list(statistics.items()), columns=["Statistic", "Value"])
            st.table(stat_df)
        else:
            st.error("No optimal solution found!")

if __name__ == '__main__':
    main()



# max extra hour -> set high to prevent no optimal solution

# to add a bonus worker, just set:
#  - contract hours = 0
#  - extra worker hours = high wrt others
#  - cost = very high wrt others
#  - conversion rate = 0