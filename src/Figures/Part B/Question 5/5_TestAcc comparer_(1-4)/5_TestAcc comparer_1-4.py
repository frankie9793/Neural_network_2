import pylab as plt

def read_data():

    with open("CNN_character.txt", "r") as ins:
        array_CNN_character = []
        for line in ins:
            line = line.split()
            temp =float(line[4])
            array_CNN_character.append(temp)

    ins.close()

    with open("CNN_word.txt", "r") as ins1:
        array_CNN_word = []
        for line in ins1:
            line = line.split()
            temp =float(line[4])
            array_CNN_word.append(temp)

    ins1.close()

    with open("RNN_character.txt", "r") as ins2:
        array_RNN_character = []
        for line in ins2:
            line = line.split()
            temp =float(line[4])
            array_RNN_character.append(temp)

    ins2.close()

    with open("RNN_word.txt", "r") as ins3:
        array_RNN_word = []
        for line in ins3:
            line = line.split()
            temp =float(line[4])
            array_RNN_word.append(temp)

    ins3.close()

    return array_CNN_character, array_CNN_word, array_RNN_character, array_RNN_word

def main():

    CNNC, CNNW, RNNC, RNNW= read_data()

    no_epochs = 100

    plt.figure(1)
    plt.plot(range(no_epochs), CNNC, label='CNN_Character')
    plt.plot(range(no_epochs), CNNW, label='CNN_Word',color='Green')
    plt.plot(range(no_epochs), RNNC, label='RNN_Character', color='Red')
    plt.plot(range(no_epochs), RNNW, label='RNN_Word',color="Orange")
    #plt.plot(range(no_epochs), data_LSTM, label='LSTM',color='Red')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.title('Test Accuracy vs Epochs')

    plt.show()
    # you may also want to remove whitespace characters like `\n` at the end of each line

if __name__ == '__main__':
  main()