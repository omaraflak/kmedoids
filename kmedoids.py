import numpy as np
import random

class KMedoids:
	def __init__(self, distance_matrix, n_clusters=2, start_prob=0.80, end_prob=0.98):
		self.distance_matrix = distance_matrix
		self.n_clusters = n_clusters
		self.n_points = len(distance_matrix)
		self.start_prob = start_prob
		self.end_prob = end_prob
		self.clusters = None
		self.medoids = None

	def initialize_medoids(self):
		# K-means++ initialization
		medoids = [random.randint(0, self.n_points - 1)]
		while len(medoids) != self.n_clusters:
			distances = [self.get_closest_medoid(medoids, point)[1] for point in range(self.n_points)]
			distances_index = np.argsort(distances)
			start_index = round(self.start_prob * len(distances_index))
			end_index = round(self.end_prob * (len(distances_index) - 1))
			new_medoid = distances_index[random.randint(start_index, end_index)]
			medoids.append(new_medoid)
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

	def associate_points_to_closest_medoid(self, medoids):
		clusters = {medoid: [medoid] for medoid in medoids}
		for point in range(self.n_points):
			medoid, _ = self.get_closest_medoid(medoids, point)
			clusters[medoid].append(point)
		return clusters

	def get_medoid_cost(self, medoid, clusters):
		return np.mean([self.get_distance(point, medoid) for point in clusters[medoid]])

	def get_configuration_cost(self, medoids, clusters):
		return np.sum([self.get_medoid_cost(medoid, clusters) for medoid in medoids])

	def get_non_medoids(self, medoids, clusters):
		return [pt for points in clusters.values() for pt in points if pt not in medoids]

	# https://en.wikipedia.org/wiki/K-medoids
	def run(self, max_iterations=10, tolerance=0.01):
		# 1- Initialize: select k of the n data points as the medoids.
		self.medoids = self.initialize_medoids()

		# 2- Associate each data point to the closest medoid.
		self.clusters = self.associate_points_to_closest_medoid(self.medoids)

		# 3- While the cost of the configuration decreases:
		# 		3.1- For each medoid m, for each non-medoid data point o:
		# 				3.1.1- Swap m and o, associate each data point to the closest medoid, recompute the cost (sum of distances of points to their medoid)
		#				3.1.2- If the total cost of the configuration increased in the previous step, undo the swap
		cost_change = float('inf')
		current_cost = self.get_configuration_cost(self.medoids, self.clusters)
		for epoch in range(max_iterations):
			if cost_change > tolerance:
				cost_change = 0
				for m in self.medoids:
					for o in self.get_non_medoids(self.medoids, self.clusters):
						new_medoids = [o] + [med for med in self.medoids if med != m]
						new_clusters = self.associate_points_to_closest_medoid(new_medoids)
						new_cost = self.get_configuration_cost(new_medoids, new_clusters)
						if new_cost < current_cost:
							self.medoids = new_medoids
							self.clusters = new_clusters
							cost_change = current_cost - new_cost
							current_cost = new_cost
							break
			else:
				break