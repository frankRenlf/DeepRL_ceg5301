# This code is mainly excerpted from the repo: https://github.com/Kchu/DeepRL_PyTorch.
import os
import pickle
import matplotlib.pyplot as plt


# paths for result log
dirc = "./data/plots/"

RESULT_PATH = [dirc + "dqn_result_o_Pendulum.pkl"]


# model load with check
result = []
for i in range(len(RESULT_PATH)):
    if os.path.isfile(RESULT_PATH[i]):
        pkl_file = open(RESULT_PATH[i], "rb")
        result.append(pickle.load(pkl_file))
        pkl_file.close()
    else:
        print("Can not find:", RESULT_PATH[i])

# plot the figure
print("Load complete!")
print("Plotting the curves!")

plt.plot(range(len(result[0])), result[0], label="dqn")
# plt.plot(range(len(result[1])), result[1], label="IQN")

plt.legend()
plt.xlabel("Iteration times(Thousands)")
plt.ylabel("Score")
plt.tight_layout()
plt.grid()
plt.savefig(dirc + "dqn_result_o_Pendulum.png")
plt.show()
plt.close()
