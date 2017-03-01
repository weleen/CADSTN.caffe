#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import os
import sys

def parse_log_file(log_file, path_to_png):
    """parse log file"""
    assert os.path.isfile(log_file), 'log file {} is not exist!'.format(log_file)

    write_train = open(path_to_png + '.train','w')
    write_train.write('Iteration\tLoss\n')
    write_test = open(path_to_png + '.test', 'w')
    write_test.write('Iteration\tLoss\n')
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for ind, line in enumerate(lines):
            if 'loss' in line and 'Train' in line and '#0' in line: # train iteration
                # print line
                train_iteration = int(lines[ind - 1] \
                        [(lines[ind - 1].find('Iteration') + 10) : (lines[ind - 1].find('('))])
                # print 'train iter ', train_iteration
                train_loss = float(line[(line.find('loss = ') + 7) : (line.find('('))])
                # print 'loss ', train_loss
                write_train.write(str(train_iteration)+'\t'+str(train_loss)+'\n')
            elif 'loss' in line and 'Test' in line and '#0' in line: # test iteration
                # print line
                test_iteration = int(lines[ind - 1] \
                        [(lines[ind - 1].find('Iteration') + 10) : (lines[ind - 1].find(','))])
                # print 'test iter ', test_iteration
                test_loss = float(line[(line.find('loss = ') + 7) : (line.find('('))])
                # print 'loss ', test_loss
                write_test.write(str(test_iteration)+'\t'+str(test_loss)+'\n')
    print('read lines over :)')
    write_train.close()
    write_test.close()
           
def plot_train_and_test(path_to_png):
    """plot train and  test vs iteration"""
    assert os.path.isfile(path_to_png + '.train'), 'train log miss'
    assert os.path.isfile(path_to_png + '.test'), 'test log miss'
    read_train = open(path_to_png + '.train','r')
    read_test = open(path_to_png + '.test', 'r')

    train_ = read_train.readlines()
    test_ = read_test.readlines()
    
    train_iter = []
    train_loss = []
    for ind, line in enumerate(train_):
        if ind == 0:
            continue
        iteration, loss = map(float, line.split())
        train_iter.append(iteration)
        train_loss.append(loss)
    
    test_iter = []
    test_loss = []
    for ind, line in enumerate(test_):
        if ind == 0:
            continue
        iteration, loss = map(float, line.split())
        test_iter.append(iteration)
        test_loss.append(loss)

    print 'start plot'
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(train_iter, train_loss, color='red', linewidth=2)
    ax1.set_title('train vs iter')
    ax2.plot(test_iter, test_loss, color='b', linewidth=2)
    ax2.set_title('test vs iter')
    
    plt.savefig(path_to_png)
    plt.show()
    print('end')
    
    read_train.close()
    read_test.close()


if __name__ == '__main__':
    argc = len(sys.argv)
    log_path = sys.argv[1]
    path_to_png = sys.argv[2]

    parse_log_file(log_path, path_to_png)
    plot_train_and_test(path_to_png)
