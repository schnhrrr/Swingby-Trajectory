# %%
import numpy as np
from ..config.config_2d import vanilla2d_config
from ..runner import run_experiment

vanilla2d_config["optimizer"]["n_adam"] = 2000
vanilla2d_config["optimizer"]["n_lbfgs"] = 10_000
weights = np.exp(np.linspace(np.log(1e-3), np.log(1e1), 666))

idx = []
w_bc_list = []
min_bc_loss = []
min_total_loss = []

if static_experiment := True:
    vanilla2d_config["optimizer"]["n_adam"] = 2000
    del vanilla2d_config["extra_parameters"]
    weights = np.exp(np.linspace(np.log(1e-3), np.log(1e3), 1000))

for i, w_bc in enumerate(weights):
    print(i + 1)
    idx.append(i + 1)
    vanilla2d_config["optimizer"]["w_bc"] = w_bc
    result = run_experiment(vanilla2d_config)
    min_bc_loss.append(min(result["result"].loss_bc))
    min_total_loss.append(min(result["result"].loss))
    w_bc_list.append(w_bc)
    if i % 10 == 0:
        data = np.column_stack([idx, w_bc_list, min_bc_loss, min_total_loss])
        np.savetxt(
            "weights2d_1109.csv",
            data,
            delimiter=",",
            fmt="%.4f",
            header="Idx,w_bc,min-bc-loss,min-total-loss",
            comments="",
        )

# %%
import matplotlib.pyplot as plt

data = np.genfromtxt("weights2d_1109.csv", delimiter=",", skip_header=True)

w_bc = data[:, 1]
loss_bc = data[:, 2]
loss_total = data[:, 3]

min_loss = min(loss_total[300:])
min_loss_idx = np.where(loss_total == min_loss)

min_bc_loss = min(loss_bc[500:])
min_bc_loss_idx = np.where(loss_bc == min_bc_loss)

plt.figure()
plt.plot(w_bc, loss_total, label="Smallest total loss")
# plt.plot(w_bc, loss_bc, label='Smallest BC Loss')
plt.scatter(
    w_bc[min_loss_idx[0][-1]],
    min_loss,
    marker="x",
    color="r",
    label=r"$\omega_{BC}=" + f"{w_bc[min_loss_idx[0][-1]]}$",
    s=50,
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\omega_{BC}$")
plt.ylabel("Smallest total loss in training process")
plt.legend()
plt.savefig("weight_search2d_static.pdf")


# %%
