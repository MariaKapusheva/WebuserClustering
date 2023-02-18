"""run_clustering.py -- the main program that does the data IO (by reading 
the 4 data files),handles the cmd line interaction with the programmer 
and calls three clustering methods."""

import sys
from kmeans import KMeans
from kohonen import Kohonen

def main():
    if len(sys.argv) == 5:
        train, test, requests, clients, dim = read_data(*sys.argv[1:])
    else:
        print("No files where defined (python run_clustering.py [traindata, testdata, requests, clients]), using defaults")
        train, test, requests, clients, dim = read_data()

    clustering_algorithm = None
    while True:
        try:
            algorithm = int(input("Run K-means (1), Kohonen SOM (2) or Quit(3) ? "))
        except ValueError:
            continue
            
        if algorithm == 1:
            clustering_algorithm = kmeans_init(train, test, dim)
        elif algorithm == 2:
            clustering_algorithm = kohonen_init(train, test, dim)
        elif algorithm == 3:
            exit()
        else:
            continue

        if not clustering_algorithm:
            continue

        input("Perform the actual training! (hit enter)")
        print("Training ...")
        clustering_algorithm.train()
        print("Training finished!")
        input("Perform the testing! (hit enter)")
        print("Testing ...")
        clustering_algorithm.test()
        print("Testing finished!")

        while True:
            try:
                output = int(input("Show output print_test(1), vector members(2), vector prototypes(3), Quit(4) or set prefetch threshold(5)? "))
            except ValueError:
                continue

            if output == 1:
                clustering_algorithm.print_test()
            elif output == 2:
                clustering_algorithm.print_members()
            elif output == 3:
                clustering_algorithm.print_prototypes()
            elif output == 4:
                break
            elif output == 5:
                try:
                    prefetch_threshold = float(input("Prefetch threshold = "))
                    clustering_algorithm.prefetch_threshold = prefetch_threshold
                    print("Testing algorithm with newly set prefetch threshold...")
                    clustering_algorithm.test()
                    print("Testing finished!")
                except Exception as e:
                    print("ERROR while setting prefetch threshold: ", e)
                    exit()

def read_data(train_filename="train.dat", test_filename="test.dat",
              requests_filename="requests.dat", clients_filename="clients.dat"):
    train, dim = read_train(train_filename)
    test = read_test(test_filename, dim)
    requests = read_requests(requests_filename)
    clients = read_clients(clients_filename)

    if dim != len(requests):
        print("ERROR: the number of dimensions in the training data does not match the number of requests in " + requests_filename)
        exit()

    return train, test, requests, clients, dim

def read_train(filename):
    train_data = []
    try:
        with open(filename, "r") as f:
            for line in f:
                train_data.append(list(map(float, line.rstrip("\n").split())))
    except Exception as e:
        print("Error while reading train data: ", e)
        exit()
        
    dim = 0
    for data in train_data:
        if dim == 0:
            dim = len(data)
        else:
            if dim != len(data):
                print("ERROR: Varying dimensions in train data.")
                exit()

    return train_data, dim

def read_test(filename, dim):
    test_data = []
    try:
        with open(filename, "r") as f:
            for line in f:
                test_data.append(list(map(float, line.rstrip("\n").split())))
    except Exception as e:
        print("Error while reading test data: ", e)
        exit()

    for data in test_data:
        if len(data) != dim:
            print("ERROR: Dimensions in test data do not correspond to those in the train data.")
            exit()

    return test_data

def read_requests(filename):
    request_data = []
    try:
        with open(filename, "r") as f:
            for line in f:
                request_data.append(line.rstrip("\n"))
    except Exception as e:
        print("Error while reading requests data: ", e)
        exit()
        
    return request_data

def read_clients(filename):
    clients = []
    try:
        with open(filename, "r") as f:
            for line in f:
                clients.append(line.rstrip("\n"))
    except Exception as e:
        print("Error while reading clients data: ", e)
        exit()
        
    return clients

def kmeans_init(train, test, dim):
    k = None
    while not isinstance(k, int):
        try:
            k = int(input("How many clusters (k) ? "))
            return KMeans(k, train, test, dim)
        except Exception as e:
            print("ERROR while trying to initialize KMeans: ", e)

def kohonen_init(train, test, dim):
    n = None
    while not isinstance(n, int):
        try:
            n = int(input("Map size (N*N) ? "))
        except Exception as e:
            print("ERROR while trying to initialize Kohonen: ", e)

    epochs = None
    while not isinstance(epochs, int):
        try:
            epochs = int(input("Number of training epochs ? "))
            return Kohonen(n, epochs, train, test, dim)
        except Exception as e:
            print("ERROR while trying to initialize Kohonen: ", e)


if __name__ == "__main__":
    main()