import matplotlib.pyplot as plt
import numpy as np

from utils.stag_hunt_utils import StagHuntLog
from utils.escalation_utils import EscalationLog

def convert_name_to_label(name):
    if "x" in name:
        return name.split("_")[-1]
    if "mp" in name:
        return f"mp = {name.split("_")[2]}"
    if "bp" in name:
        return f"bp = {name.split('_')[2][1:]}x"
    return name
    

def plot_figure(plot_data, filename, fig_title, n_rows=2, fig_num=1, y_limits_top=None, y_limits_bottom=None, single_plot=False, figsize=(12, 8), legend_rows=2):

    n_axis = len(plot_data[0][0])
    n_cols = int(np.ceil(n_axis/n_rows))

    plt.style.use('seaborn-v0_8-whitegrid')

    if single_plot:
        n_rows=1
        figsize=(6, 5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, num=fig_num)
    if single_plot:
        axes = [axes]
    else:
        axes = axes.flatten()
    

    for axis, (title, _, _, x_label, y_label) in zip(axes, plot_data[0][0]):
        axis.grid(True)
        axis.set_xlabel(x_label, fontsize=12)
        axis.set_ylabel(y_label, fontsize=12)

    colors = plt.cm.tab20.colors
    for i, (data, name) in enumerate(plot_data):
        color1, color2 = colors[2*i], colors[2*i+1]
        for j, (_, values, trend, _, _) in enumerate(data):
            axes[j].plot(values, color=color2, alpha=0.2)
            axes[j].plot(trend, color=color1, alpha=1, linewidth=2, label=convert_name_to_label(name) if j==0 else None)
    
    fig.legend(
        loc='lower center', 
        bbox_to_anchor=(0.5, 0), 
        ncol=int(np.ceil(len(plot_data)/legend_rows)), 
        fontsize=12 if single_plot else 14, title=fig_title, title_fontsize=16, frameon=True, 
    )

    if y_limits_top is not None:
        for i, axis in enumerate(axes):
            if y_limits_top[i] is not None:
                axis.set_ylim(top=y_limits_top[i])

    if y_limits_bottom is not None:
        for i, axis in enumerate(axes):
            if y_limits_bottom[i] is not None:
                axis.set_ylim(bottom=y_limits_bottom[i])

    fig.tight_layout(rect=[0, 0.1, 1, 1], h_pad=2.0)

    plt.savefig(filename, dpi=300, bbox_inches='tight')


def get_plot_data(names, base_folder, log_class, common_rewards=False):
    file_names = [f"{base_folder}/logs/{n}_ag1.pkl" for n in names]
    logs_data = [log_class.load(fn).get_plot_data() for fn in file_names]
    
    if common_rewards:
        file_names_other = [f"{base_folder}/logs/{n}_ag2.pkl" for n in names]
        logs_data_other = [log_class.load(fn).get_plot_data() for fn in file_names_other]

        for ld, ld_other in zip(logs_data, logs_data_other):
            common_rewards = (ld[0][1]+ld_other[0][1])/2
            common_rewards_avg = (ld[0][2]+ld_other[0][2])/2
            ld[0] = (ld[0][0], common_rewards, common_rewards_avg, ld[0][3], ld[0][4])

    plot_data = [(log, name) for log, name in zip(logs_data, names)]

    return plot_data

def get_single_plot_data(names, base_folder, log_class, idx=0, common_rewards=False):
    file_names = [f"{base_folder}/logs/{n}_ag1.pkl" for n in names]
    logs_data = [log_class.load(fn).get_plot_data() for fn in file_names]

    if common_rewards:
        file_names_other = [f"{base_folder}/logs/{n}_ag2.pkl" for n in names]
        logs_data_other = [log_class.load(fn).get_plot_data() for fn in file_names_other]

        for ld, ld_other in zip(logs_data, logs_data_other):
            common_rewards = (ld[0][1]+ld_other[0][1])/2
            common_rewards_avg = (ld[0][2]+ld_other[0][2])/2
            ld[0] = (ld[0][0], common_rewards, common_rewards_avg, ld[0][3], ld[0][4])

    plot_data = [((log[idx],), name) for log, name in zip(logs_data, names)]

    return plot_data


fig_num=1

###########################
# Hunt Reward Study
###########################

name = "hunt_rewards_study"
title = ""

env, log_class = "stag_hunt", StagHuntLog

base_folder = f"savings/{env}"
names = [
        "dqn_mp_-0.5",
        "dqn_mp_-0.5",
        "dqn_mp_-1.5",
        "dqn_mp_-2.5",
        "dqn_mp_-3.5",
        "dqn_mp_-5.0"
    ]

plot_data = get_plot_data(names, base_folder, log_class)

plot_figure(plot_data, f"{base_folder}/plots/{name}.png", title, n_rows=2, fig_num=fig_num,
            y_limits_top=[None, None, None, 125], y_limits_bottom=[-160, None, None, None])

fig_num+=1

###########################
# Hunt Reward Study Shared
###########################

name = "hunt_shared_rewards_study"
title = ""

env, log_class = "stag_hunt", StagHuntLog

base_folder = f"savings/{env}"
names = [
        "dqn_mp_-0.5_R",
        "dqn_mp_-0.5_R",
        "dqn_mp_-1.5_R",
        "dqn_mp_-2.5_R",
        "dqn_mp_-3.5_R",
        "dqn_mp_-5.0_R"
    ]

plot_data = get_plot_data(names, base_folder, log_class, common_rewards=True)

plot_figure(plot_data, f"{base_folder}/plots/{name}.png", title, n_rows=2, fig_num=fig_num, #figsize=(12, 10),
            y_limits_top=[None, None, None, 125], y_limits_bottom=[-200, None, None, None])

fig_num+=1

plot_data = get_single_plot_data(names, base_folder, log_class, common_rewards=True)

plot_figure(plot_data, f"{base_folder}/plots/{name}_single.png", title, fig_num=fig_num, n_rows=1, single_plot=True,
            y_limits_top=[None], y_limits_bottom=[-250])

fig_num+=1


###########################
# Hunt Size Study
###########################

name = "hunt_size_study"
title = ""

env, log_class = "stag_hunt", StagHuntLog

base_folder = f"savings/{env}"
names = [
        "dqn_mp_-1.5_5x5",
        "dqn_mp_-1.5_10x10",
        "dqn_mp_-1.5_15x15",
        "dqn_mp_-1.5_20x20",
    ]

plot_data = get_plot_data(names, base_folder, log_class)

plot_figure(plot_data, f"{base_folder}/plots/{name}.png", title, n_rows=2, fig_num=fig_num, #figsize=(12, 10),
            y_limits_top=[None, None, None, 125], y_limits_bottom=[-150, None, None, None])

fig_num+=1

plot_data = get_single_plot_data(names, base_folder, log_class)

plot_figure(plot_data, f"{base_folder}/plots/{name}_single.png", title, fig_num=fig_num, n_rows=1, single_plot=True,
            y_limits_top=[None], y_limits_bottom=[-150])

fig_num+=1

###########################
# Escalation Reward Study
###########################

name = "escalation_rewards_study"
title = ""

env, log_class = "escalation", EscalationLog

base_folder = f"savings/{env}"
names = [
        "dqn_bp_-0.1",
        "dqn_bp_-0.5",
        "dqn_bp_-1.0",
        "dqn_bp_-2.0",
    ]

plot_data = get_plot_data(names, base_folder, log_class)

plot_figure(plot_data, f"{base_folder}/plots/{name}.png", title, n_rows=1, fig_num=fig_num, figsize=(12, 4), legend_rows=1,
            y_limits_top=[None, None, None], y_limits_bottom=[-50, None, None])

fig_num+=1




###########################
# Escalation Shared Reward Study
###########################

name = "escalation_shared_rewards_study"
title = ""

env, log_class = "escalation", EscalationLog

base_folder = f"savings/{env}"
names = [
        "dqn_bp_-0.1_R",
        "dqn_bp_-0.5_R",
        "dqn_bp_-1.0_R",
        "dqn_bp_-2.0_R",
        
    ]

plot_data = get_plot_data(names, base_folder, log_class, common_rewards=True)

plot_figure(plot_data, f"{base_folder}/plots/{name}.png", title, n_rows=1, fig_num=fig_num, figsize=(12, 4), legend_rows=1,
            y_limits_top=[None, None, None], y_limits_bottom=[-5, None, None])

fig_num+=1


