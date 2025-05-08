# naive_bayes_classifier.py

import csv
import math
from collections import defaultdict, Counter

class NaiveBayesClassifier:
    def __init__(self):
        self.priors = Counter()
        self.discrete_likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.continuous_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.feature_types = []
        self.feature_names = []
        self.classes = set()
        self.total_samples = 0

    def fit(self, filepath):
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            # Extract features and their types ("name",C or D,...)
            self.feature_names = [header[i] for i in range(0, len(header)-1, 2)]
            self.feature_types = [header[i+1] for i in range(0, len(header)-1, 2)]

            data = []
            for row in reader:
                x = row[:-1]
                y = row[-1]
                self.total_samples += 1
                self.priors[y] += 1
                self.classes.add(y)
                data.append((x, y))

            for x, y in data:
                for i, val in enumerate(x):
                    fname = self.feature_names[i]
                    if self.feature_types[i] == 'D':
                        self.discrete_likelihoods[y][fname][val] += 1
                    else:
                        val = float(val)
                        mu, var = self.continuous_stats[y][fname]
                        n = self.discrete_likelihoods[y][fname].get('_count', 0) + 1
                        new_mu = mu + (val - mu) / n
                        new_var = var + (val - mu) * (val - new_mu)
                        self.continuous_stats[y][fname] = [new_mu, new_var]
                        self.discrete_likelihoods[y][fname]['_count'] = n

        # Finalize variance calculation
        for y in self.classes:
            for fname in self.feature_names:
                if self.feature_types[self.feature_names.index(fname)] == 'C':
                    mu, var = self.continuous_stats[y][fname]
                    n = self.discrete_likelihoods[y][fname]['_count']
                    self.continuous_stats[y][fname] = [mu, var / (n - 1)]

    def gaussian_prob(self, x, mu, var):
        if var == 0:
            return 1e-9
        return (1.0 / math.sqrt(2 * math.pi * var)) * math.exp(-((x - mu) ** 2) / (2 * var))

    def predict(self, input_features):
        posteriors = {}
        for y in self.classes:
            prior = self.priors[y] / self.total_samples
            likelihood = 1.0
            for i, val in enumerate(input_features):
                fname = self.feature_names[i]
                if self.feature_types[i] == 'D':
                    freq = self.discrete_likelihoods[y][fname].get(val, 0) + 1  # Laplace smoothing
                    total = sum(self.discrete_likelihoods[y][fname].values()) + len(self.discrete_likelihoods[y][fname])
                    likelihood *= freq / total
                else:
                    val = float(val)
                    mu, var = self.continuous_stats[y][fname]
                    likelihood *= self.gaussian_prob(val, mu, var)
            posteriors[y] = prior * likelihood

        total_post = sum(posteriors.values())
        for y in posteriors:
            posteriors[y] /= total_post

        return posteriors


def main():
    nb = NaiveBayesClassifier()
    nb.fit("IRIS.csv")
    result = nb.predict(['5.9', '3.0', '5.1', '1.8'])
    print(result)

if __name__ == "__main__":
    main()

# Example usage:
# nb = NaiveBayesClassifier()
# nb.fit('iris_dataset.csv')
# result = nb.predict(['5.9', '3.0', '5.1', '1.8'])
# print(result)
