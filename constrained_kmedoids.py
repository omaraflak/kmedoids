"""
This version of KMedoids gives equally sized clusters (will not be optimal).
"""
import random
import numpy as np

class KMedoids:
    def __init__(self, distance_matrix, n_clusters=2, start_prob=0.90, end_prob=0.99):
        if not 0 <= start_prob < end_prob <= 1:
            raise ValueError('start_prob must be smaller than end_prob.')
        if not n_clusters < len(distance_matrix):
            raise ValueError('number of clusters must not exceed number of data points.')

        self.distance_matrix = distance_matrix
        self.n_clusters = n_clusters
        self.n_points = len(distance_matrix)
        self.n_range = set(range(self.n_points))
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.clusters = None
        self.medoids = None

    def initialize_medoids(self):
        # K-means++ initialization
        medoids = {random.randint(0, self.n_points - 1)} # nosec
        while len(medoids) != self.n_clusters:
            distances = np.array([
                [point, self.get_closest_medoid(medoids, point)[1]]
                for point in self.n_range
                if point not in medoids
            ])
            distances_sorted = distances[distances[:, 1].argsort()]
            start_index = int(self.start_prob * len(distances))
            end_index = round(self.end_prob * (len(distances) - 1))
            new_medoid = int(distances_sorted[random.randint(start_index, end_index)][0]) # nosec
            medoids.add(new_medoid)
        return medoids

    def get_distance(self, point1, point2):
        return self.distance_matrix[point1][point2]

    def get_closest_medoid(self, medoids, point):
        closest_medoid = None
        closest_distance = float('inf')

        for medoid in medoids:
            distance = self.get_distance(point, medoid)
            if distance < closest_distance:
                closest_medoid = medoid
                closest_distance = distance

        return closest_medoid, closest_distance

    def get_closest_point(self, medoid, exception):
        closest_point = None
        closest_distance = float('inf')

        points = self.n_range - exception
        for point in points:
            distance = self.get_distance(point, medoid)
            if distance < closest_distance:
                closest_point = point
                closest_distance = distance

        return closest_point, closest_distance

    def associate_medoids_to_closest_point(self, medoids):
        clusters = {medoid: {medoid} for medoid in medoids}
        already_associated_points = set(medoids)
        associated_points = len(medoids)
        while associated_points != self.n_points:
            for medoid in medoids:
                point, _ = self.get_closest_point(medoid, already_associated_points)
                if point is not None:
                    clusters[medoid].add(point)
                    already_associated_points.add(point)
                    associated_points += 1
        return clusters

    def get_medoid_cost(self, medoid, clusters):
        return np.mean([self.get_distance(point, medoid) for point in clusters[medoid]])

    def get_configuration_cost(self, medoids, clusters):
        return np.sum([self.get_medoid_cost(medoid, clusters) for medoid in medoids])

    def get_non_medoids(self, medoids):
        return self.n_range - medoids

    def run(self, max_iterations=10, tolerance=0.01):
        # 1- Initialize: select k of the n data points as the medoids.
        self.medoids = self.initialize_medoids()

        # 2- Associate each medoid to the closest data point.
        self.clusters = self.associate_medoids_to_closest_point(self.medoids)

        # 3- While the cost of the configuration decreases:
        # 3.1- For each medoid m, for each non-medoid data point o:
        # 3.1.1- Swap m and o, associate each medoid to the closest data point,
        #        recompute the cost (sum of distances of points to their medoid)
        # 3.1.2- If the total cost of the configuration increased in the previous step, undo the swap
        cost_change = float('inf')
        current_cost = self.get_configuration_cost(self.medoids, self.clusters)
        for _ in range(max_iterations):
            if cost_change > tolerance:
                cost_change = 0
                for m in self.medoids:
                    for o in self.get_non_medoids(self.medoids):
                        new_medoids = {o} | (self.medoids - {m})
                        new_clusters = self.associate_medoids_to_closest_point(new_medoids)
                        new_cost = self.get_configuration_cost(new_medoids, new_clusters)
                        if new_cost < current_cost:
                            self.medoids = new_medoids
                            self.clusters = new_clusters
                            cost_change = current_cost - new_cost
                            current_cost = new_cost
                            break
            else:
                break

