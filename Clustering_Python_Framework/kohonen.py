import random
from math import sqrt


class Cluster:
    """This class represents the clusters, it contains the
    prototype and a set with the ID's (which are Integer objects) 
    of the datapoints that are member of that cluster."""

    def __init__(self, dim):
        self.prototype = [0.0 for _ in range(dim)]
        self.current_members = set()


class Kohonen:
    def __init__(self, n, epochs, traindata, testdata, dim):
        self.n = n
        self.epochs = epochs
        self.traindata = traindata
        self.testdata = testdata
        self.dim = dim

        # A 2-dimensional list of clusters. Size == N x N
        self.clusters = [[Cluster(dim) for _ in range(n)] for _ in range(n)]
        # Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.5
        self.initial_learning_rate = 0.8
        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0

        ## Best Matching Unit
        self.BMU = Cluster(dim)
        ## the coordinate of the BMU in the map
        self.BMU_x = 0
        self.BMU_y = 0

        ## the upper and lower bounds of x and y for the BMU neighbourhood
        self.x_upper = 0
        self.x_lower = 0
        self.y_upper = 0
        self.y_lower = 0

        ##
        self.owner = Cluster(dim)
        self.hits = 0
        self.prefetch = 0
        self.requests = 0

    def euclidean_distance(self, a, b):
        ## finds the Euclidean distance
        return sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(a, b)))

    def clear_cluster(self):
        ## clears the prototypes for each node of the map
        for i in range(self.n):
            for j in range(self.n):
                cluster = self.clusters[i][j]
                cluster.current_members.clear()

    ## Finds the cluster that the vector belongs to
    def find_owner(self, client):
        for i in range(self.n):
            for j in range(self.n):
                if client in self.clusters[i][j].current_members:
                    self.owner = self.clusters[i][j]

    ## Evaluates the performance
    def evaluate(self, client):
        # iterate along all dimensions
        for i in range(self.dim):
            # and count prefetched htmls
            if self.owner.prototype[i] > self.prefetch_threshold:
                self.prefetch = self.prefetch + 1
            # count number of hits
            if self.owner.prototype[i] > self.prefetch_threshold and self.testdata[client][i] > self.prefetch_threshold:
                self.hits = self.hits + 1
            # count number of requests
            if (self.testdata[client][i] > self.prefetch_threshold):
                self.requests = self.requests + 1

    def find_BMU(self, train_vector):
        self.BMU = self.clusters[0][0].prototype
        for j in range(self.n):
            for k in range(self.n):
                if self.euclidean_distance(train_vector, self.clusters[j][k].prototype) < self.euclidean_distance(
                        self.BMU, train_vector):
                    self.BMU = self.clusters[j][k].prototype
                    self.BMU_x = j
                    self.BMU_y = k

    def find_neighbor_bounds(self, r):
        self.x_upper = min(self.n - 1, self.BMU_x + int(r))
        self.x_lower = max(0, self.BMU_x - int(r))
        self.y_upper = min(self.n - 1, self.BMU_y + int(r))
        self.y_lower = max(0, self.BMU_y - int(r))

    def adjust_neighbors(self, r, learning_rate, train_vector):
        x = self.BMU_x
        y = self.BMU_y
        for x in range(self.x_lower, self.x_upper):
            for y in range(self.y_lower, self.y_upper):
                clusters = self.clusters[x][y]
                neighbors_prototype = []
                for dim in range(self.dim):
                    neighbors_prototype.append(
                        float((1 - learning_rate) * clusters.prototype[dim] + learning_rate * train_vector[dim]))
                self.clusters[x][y].prototype = neighbors_prototype

    def train(self):
        train_data = self.traindata

        ## Initialize the map with random vectors from the training data
        for i in range(self.n):
            for j in range(self.n):
                self.clusters[i][j].prototype = train_data[random.randint(0, len(train_data) - 1)]
        # Repeat 'epochs' times:
        for t in range(self.epochs):
            # Step 2: Calculate the square size and the learning rate, these decrease linearly with the number of epochs.
            r = (1 - t / self.epochs) * (self.n / 2)
            learning_rate = self.initial_learning_rate * (1 - t / self.epochs)

            self.clear_cluster()

            #     Step 3: Every input vector is presented to the map (always in the same order)
            for i in range(len(train_data)):
                #     For each vector its Best Matching Unit is found, and :
                self.find_BMU(train_data[i])

                #     Step 4: All nodes within the neighbourhood of the BMU are changed, you don't have to use distance relative learning.
                self.find_neighbor_bounds(r)

                self.adjust_neighbors(r, learning_rate, train_data[i])

                self.clusters[self.BMU_x][self.BMU_y].current_members.add(i)

    def test(self):
        # iterate along all clients
        for cl in range(len(self.traindata)):
            # for each client find the cluster of which it is a member
            self.find_owner(cl)
            # get the actual test data (the vector) of this client
            self.evaluate(cl)

        # set the global variables hitrate and accuracy to their appropriate value
        self.hitrate = self.hits / self.requests
        self.accuracy = self.hits / self.prefetch

    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate + self.accuracy)
        print()

    def print_members(self):
        for i in range(self.n):
            for j in range(self.n):
                print("Members cluster[" + str(i) + "][" + str(j) + "] :", self.clusters[i][j].current_members)
                print()

    def print_prototypes(self):
        # np.set_printoptions(precision=4)
        for i in range(self.n):
            for j in range(self.n):
                print("Prototype cluster[" + str(i) + "][" + str(j) + "] :", self.clusters[i][j].prototype)
                print()
