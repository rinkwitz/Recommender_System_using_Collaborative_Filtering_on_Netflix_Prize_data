import math
import numpy as np
import pandas as pd

# params:
num_iterations = 1000     # number of times doing gradient descent
num_features = 30        # length of feature vector
alpha = 0.001           # learning rate for movie vectors
llambda = 1.0          # regularization for movie vectors


# load data:
#df_movie_titles = pd.read_csv("netflix-prize-data/movie_titles.csv", names=["id", "year", "title"], encoding="iso-8859-1")
df = pd.DataFrame(np.load("data_small.npy"), columns=["movie_id", "user_id", "rating", "year", "month", "day"])
df = df[["movie_id", "user_id", "rating"]]
df.sort_values(by=["movie_id", "user_id"], ascending=[True, True], inplace=True)
data = df.values
dict_userid_to_index = {int(user_id):index for index, user_id in enumerate(df["user_id"].unique())}
dict_index_to_user_id = {index:int(user_id) for index, user_id in enumerate(df["user_id"].unique())}

# init matrices:
num_movies = len(df["movie_id"].unique())
num_users = len(df["user_id"].unique())
X = np.random.uniform(-0.2, 0.2, (num_features, num_movies))
Theta = np.random.uniform(-0.2, 0.2, (num_features, num_users))
Y = np.zeros((num_movies, num_users))
R = np.zeros((num_movies, num_users))
for row in data:
    if row[2] != 0.0:
        i = int(row[0]) - 1
        j = dict_userid_to_index[int(row[1])]
        Y[i, j] = row[2]
        R[i, j] = 1.0
Y = Y - np.mean(Y, axis=1, keepdims=True)               # mean normalization over user ratings
Lambda = np.ones((num_features, 1)) * llambda
d = {"X": X, "Theta": Theta, "loss": math.inf}
history_loss = {}
history_lr = {}

# optional for continuing optimization:
#X = np.load("X.npy")
#Theta = np.load("Theta.npy")

# vectorized gradient descent:
lr_adjustments = 0
for num_iteration in range(num_iterations):
    X_old, Theta_old = X.copy(), Theta.copy()
    X = X_old - alpha * (np.dot(Theta_old, ((np.dot(X_old.T, Theta_old) - Y) * R).T) + Lambda * X_old)
    Theta = Theta_old - alpha * (np.dot(X_old, (np.dot(X_old.T, Theta_old) - Y) * R) + Lambda * Theta_old)
    loss = 0.5 * np.sum(((np.dot(X.T, Theta) - Y) ** 2) * R) + 0.5 * llambda * np.sum(X ** 2) + 0.5 * llambda * np.sum(Theta ** 2)

    # dynamically adjust learning rate:
    if loss / d["loss"] > 1.2:
        alpha /= 3
        X = X_old.copy()                # reset to old X
        Theta = Theta_old.copy()        # reset to old Theta
        print("num_iteration: {}\tloss: {}\t --> adjusting learning rate".format(num_iteration + 1 - lr_adjustments, loss))
        lr_adjustments += 1
    else:
        d["X"] = X
        d["Theta"] = Theta
        d["loss"] = loss
        print("num_iteration: {}\tloss: {}\tlr: {}".format(num_iteration + 1 - lr_adjustments, loss, alpha))

    history_loss[num_iteration + 1 - lr_adjustments] = loss
    history_lr[num_iteration + 1 - lr_adjustments] = alpha

    # saving weights to hard disk:
    if (num_iteration + 1 - lr_adjustments) % 10 == 0:
        print("saving weights to hard disk ... ")
        np.save("X.npy", X)
        np.save("Theta.npy", Theta)
        np.save("history_loss.npy", history_loss)
        np.save("history_lr.npy", history_lr)