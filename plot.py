import matplotlib.pyplot as plt

from shared.plot_utils import plot_comparison

dir_name = 'default'
root = f'runs/{dir_name}'
tasks = ['mujoco:HalfCheetah']
agents = ['TQC']
fig, axes = plt.subplots(1, len(tasks), figsize=(5*len(tasks), 4))
save_dir = None

if len(tasks) == 1:
    plot_comparison(root, agents, tasks[0], ax=axes)
else:
    for ax, task in zip(axes, tasks):
        plot_comparison(root, agents, task, ax=ax)

fig.tight_layout()

if save_dir is None:
    plt.show()
else:
    plt.savefig(save_dir, dpi=300)