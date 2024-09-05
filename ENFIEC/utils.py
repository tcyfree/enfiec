import matplotlib.pyplot as plt


def draw_loss(Loss_list, ehoch, save_name=None):
    plt.cla()
    x = range(0, ehoch)
    y1 = Loss_list
    plt.title("Train loss per epoch", fontsize=20)
    plt.plot(x, y1, ".-", color='b')
    plt.xlabel("epoches", fontsize=20)
    plt.ylabel("Train loss", fontsize=20)
    plt.grid()
    plt.savefig(save_name)
    plt.show()
