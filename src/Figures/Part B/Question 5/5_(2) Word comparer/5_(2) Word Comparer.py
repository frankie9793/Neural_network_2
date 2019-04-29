import pylab as plt

def read_data():

    with open("TestAcc_NOdropout.txt", "r") as ins:
        array_CNN_character = []
        for line in ins:
            line = line.split()
            temp =float(line[4])
            array_CNN_character.append(temp)

    ins.close()

    with open("TestAcc_WITHdropout.txt", "r") as ins1:
        array_CNN_word = []
        for line in ins1:
            line = line.split()
            temp =float(line[4])
            array_CNN_word.append(temp)

    ins1.close()

    return array_CNN_character, array_CNN_word,

def main():

    NO_dropout, WITH_dropout= read_data()

    no_epochs = 100

    plt.figure(1)
    plt.plot(range(no_epochs), NO_dropout, label='Without dropouot')
    plt.plot(range(no_epochs), WITH_dropout, label='With dropout', color='Red')
    #plt.plot(range(no_epochs), data_LSTM, label='LSTM',color='Red')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.title('Test Accuracy vs Epochs')

    plt.show()
    # you may also want to remove whitespace characters like `\n` at the end of each line

if __name__ == '__main__':
  main()