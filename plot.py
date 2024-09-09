import numpy as np
import matplotlib.pyplot as plt


def show(log_dir, color, test=False):
    ades, fdes = [], []
    with open('{:}/log'.format(log_dir), 'r') as f:
        for line in f:
            if test == False:
                if 'ade, fde' in line:
                    ade, fde = line.split(':')[-1].split()
                    ades.append(float(ade))
                    fdes.append(float(fde))
            else:
                if 'test' in line:#test:0.2279	0.4309
                    ade, fde = line.split(':')[-1].split()
                    ades.append(float(ade))
                    fdes.append(float(fde))
    if 'sdd' in log_dir and test == False:
        ades = 50 * np.array(ades)
        fdes = 50 * np.array(fdes)
    plt.plot(ades, color=color, label=log_dir)
    plt.plot(fdes, color=color)


def show_log_old(test):
    plt.figure(figsize=(30,20))
    colors = plt.cm.viridis(np.linspace(0, 1, 10))

    plt.subplot(2, 3, 1)
    plt.title('eth')
    plt.plot([0.39 for _ in range(40)], color='r', label='MID')
    plt.plot([0.66 for _ in range(40)], color='r')
    plt.plot([0.407 for _ in range(40)], color='b', label='reproduce_mid')
    plt.plot([0.723 for _ in range(40)], color='b')
    plt.plot([0.39-0.001 for _ in range(40)], color='y', label='sota')
    plt.plot([0.58 for _ in range(40)], color='y')
    show('rl/log/eth/Feb02_11_35best_baseline', colors[0], test)


    plt.legend()


    plt.subplot(2, 3, 2)
    plt.title('hotel')
    plt.plot([0.13 for _ in range(40)], color='r', label='MID')
    plt.plot([0.22 for _ in range(40)], color='r')
    plt.plot([0.143 for _ in range(40)], color='b', label='reproduce_mid')
    plt.plot([0.234 for _ in range(40)], color='b')
    plt.plot([0.11 for _ in range(40)], color='y', label='sota')
    plt.plot([0.17 for _ in range(40)], color='y')
    show('rl/log/hotel/Feb02_11_39best_baseline', colors[0], test)

    plt.legend()


    plt.subplot(2, 3, 3)
    plt.title('univ')
    plt.plot([0.22 for _ in range(40)], color='r', label='MID')
    plt.plot([0.45 for _ in range(40)], color='r')
    plt.plot([0.223 for _ in range(40)], color='b', label='reproduce_mid')
    plt.plot([0.432 for _ in range(40)], color='b')
    plt.plot([0.22-0.001 for _ in range(40)], color='y', label='sota')
    plt.plot([0.41 for _ in range(40)], color='y')
    show('rl/log/univ/Feb02_11_39best_baseline', colors[0], test)


    plt.legend()


    plt.subplot(2, 3, 4)
    plt.title('zara1')
    plt.plot([0.17 for _ in range(40)], color='r', label='MID')
    plt.plot([0.30 for _ in range(40)], color='r')
    plt.plot([0.186 for _ in range(40)], color='b', label='reproduce_mid')
    plt.plot([0.353 for _ in range(40)], color='b')
    plt.plot([0.17-0.001 for _ in range(40)], color='y', label='sota')
    plt.plot([0.26 for _ in range(40)], color='y')
    show('rl/log/zara1/Feb02_11_39best_baseline', colors[0], test)

    plt.legend()


    plt.subplot(2, 3, 5)
    plt.title('zara2')
    plt.plot([0.13 for _ in range(40)], color='r', label='MID')
    plt.plot([0.27 for _ in range(40)], color='r')
    plt.plot([0.140 for _ in range(40)], color='b', label='reproduce_mid')
    plt.plot([0.277 for _ in range(40)], color='b')
    plt.plot([0.12 for _ in range(40)], color='y', label='sota')
    plt.plot([0.22 for _ in range(40)], color='y')
    show('rl/log/zara2/Feb02_11_40best_baseline', colors[0], test)

    plt.legend()


    plt.subplot(2, 3, 6)
    plt.title('sdd')
    plt.plot([7.61 for _ in range(40)], color='r', label='MID')
    plt.plot([14.3 for _ in range(40)], color='r')
    plt.plot([7.98 for _ in range(40)], color='b', label='reproduce_mid')
    plt.plot([14.43 for _ in range(40)], color='b')
    plt.plot([8.48 for _ in range(40)], color='y', label='sota')
    plt.plot([11.66 for _ in range(40)], color='y')
    show('rl/log/sdd/Feb02_11_40best_baseline', colors[0], test)

    plt.legend()


    plt.savefig('image_old_{:}.pdf'.format('test' if test else 'train'))



if __name__ == '__main__':
    show_log_old(True)
    show_log_old(False)