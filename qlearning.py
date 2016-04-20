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
from glob import glob

win_score = 100

def get_turn():
    change = np.random.randint(3, 5)
    turn_by = np.random.randint(-1, 2)
    return change, turn_by

def episode(grid_size_x, grid_size_y):
    # always start boulder at 0
    car_wside = 1
    car_length = 1
    track_side = 3
    lose_score = -win_score
    score = 0
    die_color = 0.75
    wrecked = False
    count = 0

    def update_road(count, turn_by, change, x, road):
        if count==change:
            change, turn_by = get_turn()
            count = 0
        else:
            count += 1

        if int(turn_by):
            x = x+turn_by
            if (track_side+2)<x<((grid_size_x-2)-track_side):
                road = list(range(x-track_side-1, x+track_side))
            else:
                count = change
        return road, count, x, turn_by, change

    # initial center for road
    x = np.random.randint(track_side+2, (grid_size_x-2)-track_side)
    road = list(range(x-track_side-1, x+track_side))
    # create track array
    X = np.ones((grid_size_x, grid_size_y))
    change, turn_by = get_turn()

    for yv in range(grid_size_y)[::-1]:
        road, count, x, turn_by, change = update_road(count, turn_by, change, x, road)
        X[yv, road] = 0.0

    # start car in the center of the road
    score = 0
    my_road = list(np.where(X[grid_size_y-1,:]<1)[0])
    xcar = my_road[track_side]
    end = 0
    while not wrecked:
        # reset initial
        X[0,:] = 1.0
        X[0,road] = 0.0
        # draw initialized car
        my_road = list(np.where(X[grid_size_y-1,:]<1)[0])
        if xcar < 0:
            score = lose_score
            X[grid_size_y-1, 0] = die_color
        elif xcar > grid_size_x-1:
            score = lose_score
            X[grid_size_y-1, grid_size_x-1] = die_color
        elif xcar not in my_road:
            score = lose_score
            X[grid_size_y-1, xcar] = die_color
        else:
            score += 1
            X[grid_size_y-1, xcar] = 0.2
        if abs(score) ==  win_score:
            wrecked = True
            end = 1
        action = yield X[np.newaxis], score
        # action can be -1,0,1
        xcar = xcar + action
        # iterate to newest track
        X = np.roll(X, 1, axis=0)
        road, count, x, turn_by, change = update_road(count, turn_by, change, x, road)

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
    # the sequential model is a linear stack of layers
    model = Sequential()
    model.add(Convolution2D(32, nb_row=3, nb_col=3,
                            input_shape=(1, grid_size_x, grid_size_y),
                            activation='relu'))
    model.add(Convolution2D(32, nb_row=3, nb_col=3, activation='relu'))
    model.add(Convolution2D(32, nb_row=3, nb_col=3, activation='relu'))
    model.add(Flatten())
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
    else:
        todel = glob(os.path.join(image_path, '*.png'))
        for f in todel:
            os.remove(f)
    frame = 0
    score = 0
    for _ in range(5):
        g = episode(grid_size_x, grid_size_y)
        S, this_score = next(g)
        # save initial image, score 0
        save_img(S[0], frame, 'score: %05i' %score)
        frame += 1
        try:
            while True:
                action = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0] - 1
                S, score = g.send(action)
                print(score, action)
                save_img(S[0], frame, 'score: %05i' %score)
                frame += 1
                if (score < 0) or (score > win_score):
                    save_img(S[0], frame, 'score: %05i' %score)
                    frame += 1
                    save_img(S[0], frame, 'score: %05i' %score)
                    frame += 1
                    save_img(S[0], frame, 'score: %05i' %score)
                    frame += 1
        except StopIteration:
            pass
    ipath = os.path.join(image_path, '*.png')
    cmd = "convert -delay 10 -loop 0 %s %s" %(ipath, gif_path)
    print("Creating gif", cmd)
    os.system(cmd)

def save_model(epoch_num, model, losses, all_scores, grid_size_x, grid_size_y, model_path):
    pfile = os.path.join(model_path, 'epoch_%03d.pkl'%epoch_num)
    print('Saving epoch %i, loss: %.6f ---- to: %s' % (epoch_num, losses[-1], pfile))
    pickle.dump({"model":model,
                 'grid_size_x':grid_size_x, 'grid_size_y':grid_size_y,
                 'losses':losses,
                 'all_scores':all_scores,
                 'epoch':epoch_num},
                open(pfile, mode='wb'))

def train_model(model, model_path, save_every, losses, epoch_start, num_epochs, grid_size_x, grid_size_y):
    epsilon = 0.05
    gamma = .8
    batch_size = 128
    # create dir for saved files if it doesnt exist
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    print("Saving models every %s epochs to directory: %s" %(save_every, model_path))
    exp_replay = experience_replay(batch_size)
    # Start experience-replay coroutine
    next(exp_replay)
    all_scores = []
    for i in range(epoch_start, num_epochs):
        ep = episode(grid_size_x, grid_size_y)
        # start coroutine of single entire episode
        S, score = next(ep)
        # initialize loss
        loss = 0.0
        scores = []
        try:
            while True:
                if np.random.random() < epsilon:
                    # sometimes a random action happens instead (explore)
                    # create random action
                    action = np.random.randint(-1, 2)
                else:
                    # Get the index of the maximum q-value of the model.
                    # Subtract one because actions are either -1, 0, or 1
                    action = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0] - 1

                # get next state
                S_prime, score = ep.send(action)
                score = score/float(win_score)
                scores.append(score)
                experience = (S, action, score, S_prime)

                # set S for next time
                S = S_prime
                batch = exp_replay.send(experience)
                if batch:
                    inputs = []
                    targets = []
                    for s, a, r, s_prime in batch:
                        # 1) do a feed forward pass for current state s to get
                        # predicted Q values for all actions
                        t = model.predict(s[np.newaxis]).flatten()
                        # use a+1 since actions can be -1,0,1
                        if (0 > r) or (r >= win_score):
                            t[a+1] = r
                        else:
                            # if we are not at an end state:
                            t[a+1] = r + gamma * model.predict(s_prime[np.newaxis]).max(axis=-1)

                        targets.append(t)
                        inputs.append(s)
                    loss += model.train_on_batch(np.array(inputs), np.array(targets))
        except StopIteration:
            pass
        losses.append(loss)
        print("epoch: %s loss: %s" %(i,loss))
        all_scores.append(scores[-2])
        if i % save_every == 0:
            if i > epoch_start:
                save_model(i, model, losses, all_scores, grid_size_x, grid_size_y, model_path)
    save_model(i, model, losses, all_scores, grid_size_x, grid_size_y, model_path)
    return model, losses, all_scores

def load_model_from_path(load_model_path):
    print("attempting to load from: %s" %load_model_path)
    if os.path.exists(load_model_path):
        lp = pickle.load(open(load_model_path, mode='rb'))
        return lp['model'], lp['losses'], lp['all_scores'], lp['grid_size_x'], lp['grid_size_y'], lp['epoch']
    else:
        print("Error: %s does not exist" %load_model_path)
        sys.exit()

if __name__ == '__main__':
    grid_size_x = 15
    grid_size_y = 15
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
        model, losses, all_scores, grid_size_x, grid_size_y, epoch_start = load_model_from_path(load_model_path)
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
        model, losses, all_scores = train_model(model, save_model_path, save_every, losses, epoch_start, num_epochs, grid_size_x, grid_size_y)

    plt.figure()
    plt.title("Loss")
    print("LOSSES", len(losses), losses)
    losses = np.array(losses)
    plt.plot(losses/max(losses))
    plt.savefig('model_losses_%03i.png' %(len(losses)))


    plt.figure()
    plt.title("Scores")
    all_scores = np.array(all_scores)
    plt.plot(all_scores)
    plt.savefig('model_scores_%03i.png' %(len(all_scores)))

    plt.clf()
    plt.figure()

    if do_make_gif:
        save_imgs(model, grid_size_x, grid_size_y, gif_path)

