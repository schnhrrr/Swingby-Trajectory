# %%
import numpy as np
from ..config.config_3d import vanilla3d_config
from ..runner import run_experiment
from ..plotter import TrajectoryPlotter

vanilla3d_config["optimizer"]["n_adam"] = 2000
vanilla3d_config["optimizer"]["n_lbfgs"] = 10000

idx = []
w_bc_list = []
min_bc_loss = []
min_total_loss = []

# weights = np.exp(np.linspace(np.log(1e0), np.log(1e2), 333))
weights = np.exp(np.linspace(np.log(1e-3), np.log(1e1), 666))

for i, w_bc in enumerate(weights):
    vanilla3d_config["optimizer"]["w_bc"] = w_bc
    result = run_experiment(vanilla3d_config)
    print(i, " ", w_bc)
    idx.append(i + 1)
    min_bc_loss.append(min(result["result"].loss_bc))
    min_total_loss.append(min(result["result"].loss))
    w_bc_list.append(w_bc)

    data = np.column_stack([idx, w_bc_list, min_bc_loss, min_total_loss])
    np.savetxt(
        "weights3d_dynamic_2108.csv",
        data,
        delimiter=",",
        fmt="%.4f",
        header="Idx,w_bc,min-bc-loss,min-total-loss",
        comments="",
    )

# %%
import matplotlib.pyplot as plt

data = np.genfromtxt("weights3d_dynamic_2108.csv", delimiter=",", skip_header=True)
w_bc = data[:, 1]
loss_bc = data[:, 2]
loss_total = data[:, 3]

min_loss = min(loss_total)
min_loss_idx = np.where(loss_total == min_loss)

min_bc_loss = min(loss_bc[500:])
min_bc_loss_idx = np.where(loss_bc == min_bc_loss)

plt.figure()
plt.scatter(w_bc, loss_total, label="Smallest Total Loss")
plt.scatter(w_bc, loss_bc, label="Smallest BC Loss")
plt.scatter(
    w_bc[min_bc_loss_idx[0][-1]],
    min_bc_loss,
    marker="x",
    color="r",
    label=r"$\omega_{BC}=" + f"{w_bc[min_bc_loss_idx[0][-1]]}$",
    s=50,
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\omega_{BC}$")
plt.ylabel("Smallest total loss in training process")
plt.legend()
plt.savefig("weight_search3d_dynamic.pdf")

# %%
