import numpy as np
import matplotlib.pyplot as plt


def f1():
    rms_list = []
    with open("./toolkit/test.txt") as f:
        for line in f:
            if "e_loss" in line:
                rms = line.split("]")[1].split(" ")[-1]
                rms_f = float(rms)
                if rms_f > 1e-9:
                    rms_list.append(rms_f)

    rms_np = np.array(rms_list)
    rms_np_lg = np.log10(rms_np)
    plt.figure()
    x_lenth = range(len(rms_np_lg))
    plt.plot(x_lenth, rms_np_lg)
    plt.show()

    print(line)


if __name__ == "__main__":
    f1()
