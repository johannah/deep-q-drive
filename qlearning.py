# based off code from bitwise-ben/Fruit
import os
from random import sample as rsample

import sys
sys.setrecursionlimit(40000)
import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD, RMSprop

from matplotlib import pyplot as plt
import argparse
import pickle


def episode(grid_size_x, grid_size_y):
    # always start boulder at 0
    y = 0
    car_wside = 1
    car_length = 1
    track_side = 3
    x = np.random.randint(track_side-1, grid_size_x-track_side)
    x = 6
    xcar = x
    score = 0
    # create track array
    X = np.ones((grid_size_x, grid_size_y))
    road = list(range(x-track_side-1, x+track_side))
    X[0:grid_size_y,road] = 0.0

    die_color = 0.75
    while True:

        # iterate to newest track
        X = np.roll(X, 1, axis=0)
        road = list(range(x-track_side-1, x+track_side))
        # reset initial
        X[0,:] = 1.0
        X[0,road] = 0.0
        # draw initialized car
        if xcar < 0:
            score = -1000
            X[grid_size_y-1, 0] = die_color
        elif xcar > grid_size_x-1:
            score = -1000
            X[grid_size_y-1, grid_size_x-1] = die_color
        elif not xcar in road:
            score = -1000
            X[grid_size_y-1, xcar] = die_color
        else:
            score += 0.1
            X[grid_size_y-1, xcar] = 0.2

        action = yield X[np.newaxis], score
        if score < 0:
            break
        if score > 2:
            score = 1000
            break

        #xcar = min(max(xcar + action, 1), grid_size_x - car_wside)
        # action can be -1,0,1
        xcar = xcar + action

        # check if this is grid_size_y or grid_size_x
        # move the pixel down the screen
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

def create_model(grid_size_x, grid_size_y):
    # Recipe of deep reinforcement learning model
    # the sequential model is a linear stack of layers
    model = Sequential()
    model.add(Convolution2D(16, nb_row=3, nb_col=3,
                            input_shape=(1, grid_size_x, grid_size_y),
                            activation='relu'))
    model.add(Convolution2D(16, nb_row=3, nb_col=3, activation='relu'))
    model.add(Flatten())
    #model.add(Dense(grid_size_x*grid_size_y, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(3))
    model.compile(RMSprop(), 'MSE')
    return model

def save_img(screen, frame, msg):
    plt.title(msg)
    plt.imshow(screen, interpolation='none')
    plt.savefig('images/%03i.png' % frame)

def save_imgs(model, grid_size_x, grid_size_y, gif_path):
    print("Creating snapshots for gif")
    image_path = 'images'
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    frame = 0
    score = 0
    for _ in range(1):
        g = episode(grid_size_x, grid_size_y)
        S, this_score = next(g)
        # save initial image, score 0
        save_img(S[0], frame, 'init')
        frame += 1
        try:
            while True:
                action = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0] - 1
                S, score_add = g.send(action)
                score += score_add
                save_img(S[0], frame, 'score: %05i' %score)
                frame += 1
                if abs(score_add):
                    # this means that the game has ended put an extra frame
                    if score_add > 0:
                        save_img(S[0], frame, 'Win')
                        save_img(S[0], frame, 'Win')
                        save_img(S[0], frame, 'Win')
                    else:
                        save_img(S[0], frame, 'Lose')
                        save_img(S[0], frame, 'Lose')
                        save_img(S[0], frame, 'Lose')
                    frame += 1

        except StopIteration:
            pass
    ipath = os.path.join(image_path, '*.png')
    cmd = "convert -delay 10 -loop 0 %s %s" %(ipath, gif_path)
    print("Creating gif", cmd)
    os.system(cmd)

def save_model(epoch_num, model, losses, grid_size_x, grid_size_y, model_path):
    pfile = os.path.join(model_path, 'epoch_%03d.pkl'%epoch_num)
    print('Saving epoch %i, loss: %.6f ---- to: %s' % (epoch_num, losses[-1], pfile))
    pickle.dump({"model":model,
                 'grid_size_x':grid_size_x, 'grid_size_y':grid_size_y,
                 'losses':losses,
                 'epoch':epoch_num},
                open(pfile, mode='wb'))


def train_model(model, model_path, save_every, losses, epoch_start, num_epochs, grid_size_x, grid_size_y):
    epsilon = .8
    gamma = .8
    batch_size = 128
    # create dir for saved files if it doesnt exist
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    print("Saving models every %s epochs to directory: %s" %(save_every, model_path))

    exp_replay = experience_replay(batch_size)
    # Start experience-replay coroutine
    next(exp_replay)

    for i in range(epoch_start, num_epochs):
        ep = episode(grid_size_x, grid_size_y)
        # start coroutine of single entire episode
        S, won = next(ep)
        # initialize loss
        loss = 0.0
        try:
            while True:
                # create random action
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

        losses.append(loss)
        if i % save_every == 0:
            if i > epoch_start:
                save_model(i, model, losses, grid_size_x, grid_size_y, model_path)
    save_model(i, model, losses, grid_size_x, grid_size_y, model_path)
    return model

def load_model_from_path(load_model_path):
    print("attempting to load from: %s" %load_model_path)
    if os.path.exists(load_model_path):
        lp = pickle.load(open(load_model_path, mode='rb'))
        return lp['model'], lp['losses'], lp['grid_size_x'], lp['grid_size_y'], lp['epoch']
    else:
        print("Error: %s does not exist" %load_model_path)
        sys.exit()


if __name__ == '__main__':
    grid_size_x = 20
    grid_size_y = 20
    parser = argparse.ArgumentParser(description='read input for model')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--do_make_gif', action='store_true')
    parser.add_argument('--gif_path', type=str,  default='example.gif')
    parser.add_argument('--save_model_path', type=str,  default='models')
    parser.add_argument('--load_model_path', type=str, default='')
    args = parser.parse_args()

    num_epochs = args.num_epochs
    save_every = args.save_every
    load_model_path = args.load_model_path
    do_make_gif = args.do_make_gif
    gif_path = args.gif_path
    if do_make_gif:
        print("Will save images at completion and create gif: %s" %gif_path)

    if load_model_path != '':
        # if we were given a model, continue
        model, losses, grid_size_x, grid_size_y, epoch_start = load_model_from_path(load_model_path)
        print("Loading model with values: epoch_start:%s loss:%s" %(epoch_start, losses[-1]))
    else:
        # else: create a model from scratch
        epoch_start = 0
        losses = []
        model = create_model(grid_size_x, grid_size_y)

    save_model_path = args.save_model_path
    do_epochs = num_epochs-epoch_start
    if (do_epochs-1) > 0:
        print("Running for %s epochs" %do_epochs)
        model = train_model(model, save_model_path, save_every, losses, epoch_start, num_epochs, grid_size_x, grid_size_y)
    if do_make_gif:
        save_imgs(model, grid_size_x, grid_size_y, gif_path)

    plt.title("Loss")
    plt.plot(losses)
    plt.savefig('model_losses_%03i.png' %(len(losses)))

