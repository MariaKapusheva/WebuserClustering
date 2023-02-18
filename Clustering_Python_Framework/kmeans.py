import random
from math import sqrt
"""kmeans.py"""

class Cluster:
    """This class represents the clusters, it contains the
    prototype (the mean of all it's members) and memberlists with the
    ID's (which are Integer objects) of the datapoints that are member
    of that cluster. You also want to remember the previous members so
    you can check if the clusters are stable."""
    def __init__(self, dim):
        self.prototype = [0.0 for _ in range(dim)]
        self.current_members = set()
        self.previous_members = set()
        self.previous_prototype = [0.0 for _ in range(dim)]

class KMeans:
    def __init__(self, k, traindata, testdata, dim):
        self.k = k
        self.traindata = traindata
        self.copy_traindata = self.traindata
        print(self.traindata)
        self.testdata = testdata
        self.dim = dim

        # Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.95
        # An initialized list of k clusters
        self.clusters = [Cluster(dim) for _ in range(k)]

        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0

        self.first_try_distances = True
        self.first_try_distances_count = 0
        self.first_try_members = True
        self.first_try_members_count = 0
        self.accumulative_count = 0
        self.member_count = 0
        self.counter = 0
        self.n_vectors = []
        self.random_indices = set()

    ## Computes the mean of a cluster by summing over all cluster members and then dividing by their number.
    def calculate_cluster_average(self, cluster_index):
        vector_average_sum = 0
        n_members = len(self.clusters[cluster_index].current_members)
        for i in range(n_members):
            vector = self.clusters[cluster_index].current_members
            vector_average_sum += sum(vector[i])/200
        cluster_average = vector_average_sum/n_members
        return cluster_average

    ## Computes the mean of a data vector by summing over all vector members and then dividing by their number.
    def calculate_vector_average(self, cluster_index, vector_index):
        vector_average = 0
        n_members = len(self.clusters[cluster_index].current_members)
        vector = self.clusters[cluster_index].current_members
        vector_average = sum(vector[vector_index])/200
        return vector_average

    ## finds the Euclidean distance
    def euclidean_distance(self, a, b):
        return sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(a, b)))

    ## Computes the new data points for the prototype for a given cluster
    def compute_prototype_values(self, cluster_members):
        vector_mean = 0
        prototype_members = []
        ## For each data point within a vector from that cluster
        for n_data_point in range(200):
            ## For each data vector from that cluster
            for n_vector in range(len(cluster_members)):
                ## Add the data points up
                vector_mean += cluster_members[n_vector][n_data_point]
            ## Divide by the number of data vectors within the cluster
            vector_mean /= len(cluster_members)
            prototype_members.append(vector_mean)
            vector_mean = 0
        return prototype_members

    def compute_euclidean_distances(self):
        ## Initialize k * 70 matrix
        vector_distance = [[0] * (self.k) for i in range(len(self.traindata))]
        for cluster_index in range(len(self.clusters)):
            ## Initialization step (first time)
            if self.first_try_distances:
                ## Assign the randomized vectors to a prototype for each cluster
                self.clusters[cluster_index].prototype = self.traindata[cluster_index]
                ## For each data vector in traindata
                for vector_index in range(len(self.traindata)):
                    ## Compute the Euclidean distance of each vector to a prototype, and add it to the matrix
                    vector_distance[vector_index][cluster_index] = self.euclidean_distance(
                        self.clusters[cluster_index].prototype,
                        self.traindata[vector_index])
            ## For repeating steps 2-3
            ## The algorithm follows similar logic to the one above, except now it takes into account how many vectors
            ## are in each updated cluster with member.count, and how many vectors there are in total with
            ## accumulative.count
            else:
                cluster_vector = self.clusters[cluster_index].current_members
                while self.accumulative_count < self.n_vectors[cluster_index]:
                    for index in range(self.k):
                        vector_distance[self.accumulative_count][index] = self.euclidean_distance(
                            self.clusters[index].prototype,
                            cluster_vector[self.member_count])
                    self.accumulative_count += 1
                    self.member_count += 1
                self.member_count = 0
        self.accumulative_count = 0
        self.first_try_distances = False
        self.n_vectors = []
        return vector_distance

    def compute_new_cluster_members(self, cluster_index, vector_distance):
        cluster_members = []
        ## Initialization step (first time)
        if self.first_try_members:
            ## For each vector in the data
            for vector_index in range(len(self.traindata)):
                ## Get the index of the smallest Euclidean distance within the matrix for that particular vector
                cluster_vector_index = vector_distance[vector_index].index(min(vector_distance[vector_index]))
                ## If the index of that particular vector corresponds to the current cluster index, add it to the cluster
                if cluster_vector_index == cluster_index:
                    cluster_members.append(self.traindata[vector_index])
            ## Assign the current cluster members
            self.clusters[cluster_index].current_members = cluster_members
            ## Assign the new prototype data points
            self.clusters[cluster_index].prototype = self.compute_prototype_values(cluster_members)
            self.first_try_members_count += 1
        ## For repeating steps 2-3
        ## The algorithm follows similar logic to the one for initialization
        else:
            cluster_vector = self.clusters[cluster_index].current_members
            for vector_index in range(len(self.clusters[cluster_index].current_members)):
                cluster_vector_index = vector_distance[self.counter].index(min(vector_distance[self.counter]))
                if cluster_vector_index == cluster_index:
                    cluster_members.append(cluster_vector[vector_index])
                self.counter += 1

            self.clusters[cluster_index].current_members = cluster_members
            self.compute_prototype_values(cluster_members)
            self.clusters[cluster_index].prototype = self.compute_prototype_values(cluster_members)

        if self.first_try_members_count == self.k:
            self.first_try_members = False

    def train(self):
        cluster_centroid = []
        switch = True
        switch_count = 0
        # random.shuffle(self.traindata)
        ## Generate a list of random indices without replacement

        while (len(self.random_indices) < len(self.traindata)):
            self.random_indices.add(random.randint(0, len(self.traindata)-1))
        self.random_indices = list(self.random_indices)
        random.shuffle(self.random_indices)

        for vector_index in range(len(self.traindata)):
            self.traindata[self.random_indices[vector_index]] = self.copy_traindata[vector_index]

        ## Find the Euclidean distances of the data vectors in regards to the random vectors assigned to the prototype
        ## clusters
        vector_distance = self.compute_euclidean_distances()

        ## Find and place the vectors in their respective cluster depending on their Euclidean distances
        ## and then compute the new centroids for each cluster
        print("PROTOTYPES")
        for cluster_index in range(len(self.clusters)):
            self.compute_new_cluster_members(cluster_index, vector_distance)
            cluster_centroid.append(self.calculate_cluster_average(cluster_index))
            self.clusters[cluster_index].previous_prototype = self.clusters[cluster_index].prototype
            print(self.clusters[cluster_index].prototype)
        ## while switch == True
        while switch:
            ## Compute the euclidean distances of current prototype vector to cluster vectors
            accumulation = 0
            for cluster_index in range(len(self.clusters)):
                accumulation += len(self.clusters[cluster_index].current_members)
                self.n_vectors.append(accumulation)

            vector_distance = self.compute_euclidean_distances()
            print("PROTOTYPES")
            for cluster_index in range(len(self.clusters)):
                ## Previous members & prototype = current members & prototype before current members & prototype change
                self.clusters[cluster_index].previous_members = self.clusters[cluster_index].current_members
                self.clusters[cluster_index].previous_prototype = self.clusters[cluster_index].prototype
                print(self.clusters[cluster_index].prototype)
                ## Find and place the vectors in their respective cluster depending on their Euclidean distances
                self.compute_new_cluster_members(cluster_index, vector_distance)
                ## If membership of patterns for each cluster does not change, end the while loop
                if self.clusters[cluster_index].current_members == self.clusters[cluster_index].previous_members:
                    if self.clusters[cluster_index].prototype == self.clusters[cluster_index].previous_prototype:
                        switch_count += 1
                        if switch_count == self.k:
                            switch = False
            ## Reset the switch counter after each epoch
            self.counter = 0
            ## Reset if not all membership patterns remained the same
            switch_count = 0
        pass

    def test(self):
        post_threshold_prototype = []
        prefetch_count = 0
        number_of_requests_count = 0
        hit_count = 0
        for cluster_index in range(len(self.clusters)):
            post_threshold_prototype.append(self.clusters[cluster_index].prototype)
            for data_point_index in range(len(post_threshold_prototype[cluster_index])):
                if post_threshold_prototype[cluster_index][data_point_index] > self.prefetch_threshold:
                    post_threshold_prototype[cluster_index][data_point_index] = 1
                    prefetch_count += 1
                else:
                    post_threshold_prototype[cluster_index][data_point_index] = 0

        for vector_index in range(len(self.testdata)):
            user_ID = self.random_indices[vector_index]
            cluster_vector = self.traindata[user_ID]
            for cluster_index in range(len(self.clusters)):
                for cluster_vector_index in range(len(self.clusters[cluster_index].current_members)):
                    if cluster_vector == self.clusters[cluster_index].current_members[cluster_vector_index]:
                        cluster_ID = cluster_index
                        break
                else:
                    continue
                break
            for data_point_index in range(len(post_threshold_prototype[cluster_ID])):
                if post_threshold_prototype[cluster_ID][data_point_index] == self.testdata[vector_index][data_point_index]:
                    hit_count += 1
                if self.testdata[vector_index][data_point_index] == 1:
                    number_of_requests_count += 1

        self.hitrate = hit_count / number_of_requests_count
        self.accuracy = hit_count / prefetch_count





        # iterate along all clients. Assumption: the same clients are in the same order as in the testData
        # for each client find the cluster of which it is a member
        # get the actual testData (the vector) of this client
        # iterate along all dimension
        # and count prefetched htmls
        # count number of hits
        # count number of requests
        # set the variables hitrate and accuracy to their appropriate value
        pass


    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate+self.accuracy)
        print()

    def print_members(self):
        for i, cluster in enumerate(self.clusters):
            print("Members cluster["+str(i)+"] :", cluster.current_members)
            print()

    def print_prototypes(self):
        for i, cluster in enumerate(self.clusters):
            print("Prototype cluster["+str(i)+"] :", cluster.prototype)
            print()

    def partition(self):
        random.shuffle(self.traindata)
        return [self.traindata[i::self.k] for i in range(self.k)]

