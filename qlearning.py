# based off code from bitwise-ben/Fruit
import os
from random import sample as rsample

import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD, RMSprop

from matplotlib import pyplot as plt


GRID_SIZE_X = 10
GRID_SIZE_Y = 10

def episode():
    """
    Coroutine of episode.
    Action has to be explicitly send to this coroutine.
    """
    x, y, z = (
        np.random.randint(0, GRID_SIZE_X), 0,  # X,Y of boulder
        np.random.randint(1, GRID_SIZE_X - 1)  # X of basket
    )
    while True:
        X = np.zeros((GRID_SIZE_X, GRID_SIZE_Y))  # Reset grid
        # Draw boulder
        X[y, x] = 1.
        bar = list(range(z - 1, z + 2))
        X[-1, bar] = 1.  # Draw basket

        # End of game is known when fruit is at second to last line of grid.
        # End represents either a win or a loss
        end = int(y >= GRID_SIZE_Y - 2)
        if end and x not in bar:
            end *= -1

        action = yield X[np.newaxis], end
        if end:
            break

        # check if this is GRID_SIZE_Y or GRID_SIZE_X
        z = min(max(z + action, 1), GRID_SIZE_Y - 2)
        y += 1


def experience_replay(batch_size):
    """
    Coroutine of experience replay.

    Provide a new experience by calling send, which in turn yields
    a random batch of previous replay experiences.
    """
    memory = []
    while True:
        experience = yield rsample(memory, batch_size) if batch_size <= len(memory) else None
        memory.append(experience)

def create_model():
    # Recipe of deep reinforcement learning model
    model = Sequential()
    model.add(Convolution2D(16, nb_row=3, nb_col=3, input_shape=(1, GRID_SIZE_X, GRID_SIZE_Y), activation='relu'))
    model.add(Convolution2D(16, nb_row=3, nb_col=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(3))
    model.compile(RMSprop(), 'MSE')
    return model

def save_img():
    if 'images' not in os.listdir('.'):
        os.mkdir('images')
    frame = 0
    while True:
        screen = (yield)
        plt.imshow(screen[0], interpolation='none')
        plt.savefig('images/%03i.png' % frame)
        frame += 1

def same_imgs(model):
    img_saver = save_img()
    next(img_saver)

    for _ in range(10):
        g = episode()
        S, _ = next(g)
        img_saver.send(S)
        try:
            while True:
                act = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0] - 1
                S, _ = g.send(act)
                img_saver.send(S)

        except StopIteration:
            pass

    img_saver.close()

if __name__ == '__main__':

    num_epochs = 600
    batch_size = 128
    epsilon = .8
    gamma = .8

    model = create_model()
    exp_replay = experience_replay(batch_size)
    # Start experience-replay coroutine
    next(exp_replay)

    for i in range(num_epochs):
        ep = episode()
        S, won = next(ep)  # Start coroutine of single entire episode
        loss = 0.
        try:
            while True:
                action = np.random.randint(-1, 2)
                if np.random.random() > epsilon:
                    # Get the index of the maximum q-value of the model.
                    # Subtract one because actions are either -1, 0, or 1
                    action = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0] - 1

                S_prime, won = ep.send(action)
                experience = (S, action, won, S_prime)
                S = S_prime

                batch = exp_replay.send(experience)
                if batch:
                    inputs = []
                    targets = []
                    for s, a, r, s_prime in batch:
                        # The targets of unchosen actions are the q-values of the model,
                        # so that the corresponding errors are 0. The targets of chosen actions
                        # are either the rewards, in case a terminal state has been reached,
                        # or future discounted q-values, in case episodes are still running.
                        t = model.predict(s[np.newaxis]).flatten()
                        t[a + 1] = r
                        if not r:
                            t[a + 1] = r + gamma * model.predict(s_prime[np.newaxis]).max(axis=-1)
                        targets.append(t)
                        inputs.append(s)

                    loss += model.train_on_batch(np.array(inputs), np.array(targets))

        except StopIteration:
            pass

        if (i + 1) % 100 == 0:
            print('Epoch %i, loss: %.6f' % (i + 1, loss))


    save_imgs(model)

