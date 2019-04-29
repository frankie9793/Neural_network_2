import pylab as plt

def read_data():

    with open("TestAcc_GRU.txt", "r") as ins:
        array_GRU = []
        for line in ins:
            line = line.split()
            temp =float(line[4])
            array_GRU.append(temp)

    ins.close()

    with open("TestAcc_VANILLA.txt", "r") as ins1:
        array_VANILLA = []
        for line in ins1:
            line = line.split()
            temp =float(line[4])
            array_VANILLA.append(temp)

    ins1.close()

    with open("TestAcc_LSTM.txt", "r") as ins2:
        array_LSTM = []
        for line in ins2:
            line = line.split()
            temp = float(line[4])
            array_LSTM.append(temp)

    ins2.close()


    return array_GRU, array_VANILLA,array_LSTM

def main():

    data_GRU,data_Vanilla, data_LSTM = read_data()

    no_epochs = 100

    plt.figure(1)
    plt.plot(range(no_epochs), data_GRU, label='GRU')
    plt.plot(range(no_epochs), data_Vanilla, label='VANILLA',color='Green')
    plt.plot(range(no_epochs), data_LSTM, label='LSTM',color='Red')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.title('Test Accuracy vs Epochs')

    plt.show()
    # you may also want to remove whitespace characters like `\n` at the end of each line

if __name__ == '__main__':
  main()