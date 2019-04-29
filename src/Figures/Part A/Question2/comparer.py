import pylab as plt

def read_data():

    with open("Test1.txt", "r") as ins:
        test1 = []
        for line in ins:
            line = line.split()
            temp =float(line[4])
            test1.append(temp)

    ins.close()

    with open("Test2.txt", "r") as ins1:
        test2 = []
        for line in ins1:
            line = line.split()
            temp =float(line[4])
            test2.append(temp)

    ins1.close()

    with open("Test3.txt", "r") as ins2:
        test3 = []
        for line in ins2:
            line = line.split()
            temp =float(line[4])
            test3.append(temp)

    ins2.close()

    with open("Test4.txt", "r") as ins3:
        test4 = []
        for line in ins3:
            line = line.split()
            temp =float(line[4])
            test4.append(temp)

    ins3.close()

    with open("Test5.txt", "r") as ins4:
        test5 = []
        for line in ins4:
            line = line.split()
            temp =float(line[4])
            test5.append(temp)

    ins4.close()

    with open("Test6.txt", "r") as ins5:
        test6 = []
        for line in ins5:
            line = line.split()
            temp =float(line[4])
            test6.append(temp)

    ins5.close()

    with open("Test7.txt", "r") as ins6:
        test7 = []
        for line in ins6:
            line = line.split()
            temp =float(line[4])
            test7.append(temp)

    ins6.close()

    with open("Test8.txt", "r") as ins7:
        test8 = []
        for line in ins7:
            line = line.split()
            temp =float(line[4])
            test8.append(temp)

    ins7.close()

    with open("Test9.txt", "r") as ins8:
        test9 = []
        for line in ins8:
            line = line.split()
            temp =float(line[4])
            test9.append(temp)

    ins8.close()

    with open("Test10.txt", "r") as ins9:
        test10 = []
        for line in ins9:
            line = line.split()
            temp =float(line[4])
            test10.append(temp)

    ins9.close()



    return test1,test2 ,test3, test4, test5, test6, test7, test8, test9, test10

def main():

    test1, test2, test3, test4, test5, test6, test7, test8, test9, test10= read_data()

    no_epochs = 1000

    plt.figure(1,figsize=(30,20))
    plt.plot(range(no_epochs),test1 , label='Size=10')
    plt.plot(range(no_epochs), test2, label='Size=20')
    plt.plot(range(no_epochs), test3, label='Size=30')
    plt.plot(range(no_epochs), test4, label='Size=40')
    plt.plot(range(no_epochs), test5, label='Size=50')
    plt.plot(range(no_epochs), test6, label='Size=60')
    plt.plot(range(no_epochs), test7, label='Size=70')
    plt.plot(range(no_epochs), test8, label='Size=80')
    plt.plot(range(no_epochs), test9, label='Size=90')
    plt.plot(range(no_epochs), test10, label='Size=100')

    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.title('Test Accuracy vs Epochs')

    plt.show()
    # you may also want to remove whitespace characters like `\n` at the end of each line

if __name__ == '__main__':
  main()