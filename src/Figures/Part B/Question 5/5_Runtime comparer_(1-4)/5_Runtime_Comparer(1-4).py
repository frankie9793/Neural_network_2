import pylab as plt

def main():

    runtimes = [645.566, 271.263, 2977.84, 3292.08]


    plt.figure(1)
    plt.plot(range(4), runtimes, marker='x', linestyle='-')
    plt.xticks(range(4),['CNN_char','CNN_word','RNN_char','RNN_word'])
    plt.xlabel('Type')
    plt.ylabel('Runtimes/ms')
    plt.title('Runtimes')
    #plt.savefig('./figures/PARTB_QNS3_a.png')

    plt.show()

if __name__ == '__main__':
  main()

