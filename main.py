import csv
import math
from collections import defaultdict, Counter

class NaiveBayesClassifier:
    def __init__(self):
        self.priors = Counter() #Counter μετράει τον αριθμό εμφανήσεων ενος αντικειμένου
        self.discrete_likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) #Δημιουργεί ένα dict τριών στρώσεων
        self.continuous_stats = defaultdict(lambda: defaultdict(lambda: [0, 0])) #Δημιουργεί ένα dict δύο στρώσεων
        self.feature_types = [] #Τύποι των features χωρίζονται σε D(discrete) & C(continuous)
        self.feature_names = [] #Ονόματα των features π.χ. Sepal Width
        self.classes = set()
        self.total_samples = 0

    def fit(self, filepath):
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            # Διαβάζει τα feature και τον τύπο τους π.χ. ("Sepal width",C or D,...)
            self.feature_names = [header[i] for i in range(0, len(header)-1, 2)]
            self.feature_types = [header[i+1] for i in range(0, len(header)-1, 2)]

            data = []
            for row in reader:
                x = row[:-1] # Τα features
                y = row[-1] # Η κατηγορία του δεδομένου
                self.total_samples += 1
                self.priors[y] += 1
                self.classes.add(y)
                data.append((x, y))
                
            # Υπολογίζει τις πιθανότητες για τα διακριτά χαρακτηριστικά και τις στατιστικές για τα συνεχόμενα χαρακτηριστικά
            for x, y in data:
                for i, value in enumerate(x):
                    fname = self.feature_names[i]
                    if self.feature_types[i] == 'D':
                        self.discrete_likelihoods[y][fname][value] += 1
                    else:
                        value = float(value)
                        mean, var = self.continuous_stats[y][fname]
                        n = self.discrete_likelihoods[y][fname].get('_count', 0) + 1
                        new_mean = mean + (value - mean) / n
                        new_var = var + (value - mean) * (value - new_mean)
                        self.continuous_stats[y][fname] = [new_mean, new_var]
                        self.discrete_likelihoods[y][fname]['_count'] = n

        # Κανονικοποιεί τις στατιστικές των συνεχών χαρακτηριστικών διαιρώντας με το πλήθος των δειγμάτων
        for y in self.classes:
            for fname in self.feature_names:
                if self.feature_types[self.feature_names.index(fname)] == 'C':
                    mean, var = self.continuous_stats[y][fname]
                    n = self.discrete_likelihoods[y][fname]['_count']
                    self.continuous_stats[y][fname] = [mean, var / (n - 1)]

    def gaussian_prob(self, x, mean, var):
        if var == 0:
            return 1e-9 # Αποφυγή διαίρεσης με το μηδέν
        return (1.0 / math.sqrt(2 * math.pi * var)) * math.exp(-((x - mean) ** 2) / (2 * var))

    def predict(self, input_features):
        
        # Yπολογίζει τις πιθανότητες για κάθε κατηγορία και επιστρέφει την κατηγορία με τη μεγαλύτερη πιθανότητα
        posteriors = {}
        for y in self.classes:
            prior = self.priors[y] / self.total_samples
            likelihood = 1.0
            for i, value in enumerate(input_features):
                fname = self.feature_names[i]
                if self.feature_types[i] == 'D':
                    freq = self.discrete_likelihoods[y][fname].get(value, 0) + 1  # Laplace smoothing
                    total = sum(self.discrete_likelihoods[y][fname].values()) + len(self.discrete_likelihoods[y][fname]) # Laplace smoothing
                    likelihood *= freq / total
                else:
                    value = float(value)
                    mean, var = self.continuous_stats[y][fname]
                    likelihood *= self.gaussian_prob(value, mean, var)
            posteriors[y] = prior * likelihood # Υπολογισμός της πιθανοφάνειας

        total_post = sum(posteriors.values())
        for y in posteriors:
            posteriors[y] /= total_post # Κανονικοποίηση των πιθανοτήτων
        # uncomment για να δεις την δομή των πινάκων
        # print(self.priors,"\n-----\n", self.discrete_likelihoods, "\n-----\n", self.continuous_stats)

        return posteriors


def main():
    nb = NaiveBayesClassifier()
    nb.fit("artificial_dataset.csv")
    user_input = input("Enter features (example: 5.1, 3.5, 1.4, 0.2): ").split(",")
    result = nb.predict(user_input)
    print(result)

if __name__ == "__main__":
    main()