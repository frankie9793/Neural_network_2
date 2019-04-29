import pylab as plt

def read_data():

    with open("TestAcc_GRU.txt", "r") as ins:
        array_GRU = []
        for line in ins:
            line = line.split()
            temp =float(line[4])
            array_GRU.append(temp)

    ins.close()

    with open("TestAcc_2Layers.txt", "r") as ins1:
        array_VANILLA = []
        for line in ins1:
            line = line.split()
            temp =float(line[4])
            array_VANILLA.append(temp)

    ins1.close()

    return array_GRU, array_VANILLA

def main():

    data_GRU,data_2Layers= read_data()

    no_epochs = 100

    plt.figure(1)
    plt.plot(range(no_epochs), data_GRU, label='Single')
    plt.plot(range(no_epochs), data_2Layers, label='Double',color='Green')
    #plt.plot(range(no_epochs), data_LSTM, label='LSTM',color='Red')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.title('Test Accuracy vs Epochs')

    plt.show()
    # you may also want to remove whitespace characters like `\n` at the end of each line

if __name__ == '__main__':
  main()