from sklearn.neural_network import MLPClassifier

# Format Data
# Input data and answer key will be defined by the user...will be
# left blank for now until we find out how data will inputted
X = []

# Answer key
y = []

# Define Model
mlp = MLPClassifier(hidden_layer_sizes=(4,),
                    activation='sigmoid', solver='adam',
                    learning_rate_init=0.03, max_iter=2000,
                    random_state=1)

# Train
mlp.fit(X, y)

# Output
# storing variables here for now until we know where and how to
# output it.
predictions = mlp.predict(X)
accuracy = mlp.score(X, y) * 100
weights = mlp.coefs_
