import numpy as np
from scipy.special import expit
import time
import random
from operator import add
import math

games_per_cycle = 200
player_n = 4


# input neurons
in_neurons = (player_n + 1) * 5 * 6
# neurons in hidden layer
hid_neurons = 256
# Mutation rate
mr = 0.001
# Weight magnitude
wm = 1
# num of actions
output_neurons = 5

cards = []
players = [None for i in range(player_n)]

class Card:

    def __init__(self, name, c_id):
        self.maa = name[:1]
        self.num = int(name[1:])
        tmp = list(c_id)
        self.c_id = [int(i) for i in tmp]

class Player:

    hand = []
    not_played = [2, 2, 2, 2, 2]
    
    def __init__(self, place, neurons_in, neurons_mid, neurons_out,
                 w1=None, w2=None, w3=None, b1=None, b2=None, b3=None):
        self.place = place
        self.neurons_mid = neurons_mid
        
        if w1 is not None:
            # self.neurons_mid = int(neurons_mid+random.randint(-1, 1))
            self.w1 = np.clip(np.resize(w1, (self.neurons_mid, neurons_in)) + np.random.uniform(-mr, mr, (self.neurons_mid, neurons_in)), -wm, wm)
            #self.w2 = np.clip(w2 + np.random.uniform(-mr, mr, (self.neurons_mid, neurons_mid)), -wm, wm)
            self.w3 = np.clip(np.resize(w3, (neurons_out, self.neurons_mid)) + np.random.uniform(-mr, mr, (neurons_out, self.neurons_mid)), -wm, wm)
            self.b1 = np.clip(np.resize(b1, (self.neurons_mid)) + np.random.uniform(-mr, mr, self.neurons_mid), -wm/4, wm/4)
            #self.b2 = np.clip(b2 + np.random.uniform(-mr, mr, neurons_mid), -wm, wm)
            self.b3 = np.clip(b3 + np.random.uniform(-mr, mr, neurons_out), -wm, wm)
            return
        
        self.w1 = np.zeros((neurons_mid, neurons_in))
        #self.w2 = np.zeros((neurons_mid, neurons_mid))
        self.w3 = np.zeros((neurons_out, neurons_mid))
        self.b1 = np.zeros(neurons_mid)
        #self.b2 = np.random.randint(-wm, wm, size=neurons_mid)
        self.b3 = np.zeros(neurons_out)


    def choose_card(self, a0):
        a1 = self.network_step(a0, self.w1, self.b1)
        #a2 = self.network_step(a1, self.w2, self.b2)
        output = self.network_step(a1, self.w3, self.b3) # a1 to a2
        return output

    def network_step(self, a, w, b):
        return expit(np.dot(w, a) + b)


class Tikki:

    table = [[0 for i in range(5)] for j in range(player_n)]

    def __init__(self, cycle, w1=None, w2=None, w3=None, b1=None, b2=None, b3=None, neurons_mid=hid_neurons):
        for i in range(player_n):
            if w1 is not None:# and i != 0:
                players[i] = Player(i, in_neurons, neurons_mid, output_neurons,
                                           w1=w1, w2=w2, w3=w3, b1=b1, b2=b2, b3=b3)
            else:
                players[i] = Player(i,  in_neurons, hid_neurons, output_neurons)

    def cycle(self):
        random.shuffle(cards)
        for i in range(player_n):
            players[i].hand = cards[(i*5):(i*5+5)]
            players[i].not_played = [2, 2, 2, 2, 2]
        confidence = []
        leader = random.randint(0, player_n-1)
        suit = ""
        for turn in range(5):
            for per in range(player_n):
                p = players[(leader + per)%player_n]
                input_dat = self.input_dat(p.place)
                if per == 0:
                    weights = [a + b for a, b in zip(p.not_played, p.choose_card(input_dat))]
                    self.table[p.place][turn] = p.hand[weights.index(max(weights))]
                    p.not_played[weights.index(max(weights))] = 0
                    suit = self.table[p.place][turn].maa
                    confidence.append(math.modf(max(weights))[0])
                else:
                    correct_suit = [0, 0, 0, 0, 0]
                    for i in range(5):
                        if p.hand[i].maa == suit:
                            correct_suit[i] = 1
                    cor = [a + b for a, b in zip(p.not_played, correct_suit)]
                    weights = [a + b for a, b in zip(cor, p.choose_card(input_dat))]
                    self.table[p.place][turn] = p.hand[weights.index(max(weights))]
                    p.not_played[weights.index(max(weights))] = 0
                    confidence.append(math.modf(max(weights))[0])
            leading_card = 0
            for i in range(player_n):
                if self.table[i][turn].maa == suit and self.table[i][turn].num > leading_card:
                    leading_card = self.table[i][turn].num
                    leader = i
        return leader, sum(confidence)/len(confidence)
        

    def input_dat(self, place):
        dat = []
        for i in range(5):
            dat.append(players[place].hand[i].c_id)
        for j in range(player_n):
            for i in range(5):
                if self.table[j][i] == 0:
                    dat.append([0, 0, 0, 0, 0, 0])
                else:
                    dat.append(self.table[j][i].c_id)
        rer = np.array(dat)
        return rer.flatten()

def training_cycle(cycle, p=None):
    t1 = time.time()
    wins = [0 for i in range(player_n)]
    if p is not None:
        print()
        game = Tikki(cycle, w1=p.w1, w3=p.w3, b1=p.b1, b3=p.b3, neurons_mid=p.neurons_mid) # w2=p.w2, , , b2=p.b2
    else:
        game = Tikki(cycle)
    for i in range(games_per_cycle):
        winner, confidence = game.cycle()
        wins[winner] += 1
    t2 = time.time()
    p = players[wins.index(max(wins))]
    return p, wins, t2-t1, confidence

def load_cards():
    file = open("pakka", 'r')
    lines = file.readlines()
    for line in lines:
        l = line.split()
        if l[0] == "Kortti":
            continue
        else:
            cards.append(Card(l[0], l[1]))

def human_vs_machine(p):
    players_ = []
    table = [[0 for i in range(5)] for j in range(player_n)]
    for i in range(player_n):
        if int(input("Human or machine (1/0)")):
            players_.append("Human")
        else:
            players_.append(Player(i, in_neurons, hid_neurons, output_neurons,
                                  w1=p.w1, w3=p.w3, b1=p.b1, b3=p.b3))

    for p in players_:
        if p != "Human":
            print("Look file 'pakka' for card indexies")
            p.hand = [cards[int(input(("Card " + str(i+1) + " index: ")))] for i in range(5)]
    leader = random.randint(0, player_n-1)

    suit = ""
    for turn in range(5):
        print(leader)
        for per in range(player_n):
            if players_[(leader+per)%player_n] == "Human":
                ind = int(input("Index of card to be played"))
                table[(leader+per)%player_n][turn] = cards[ind]
                if per == 0:
                    while True:
                        suit = input("Played suit (h, r, i, p): ")
                        if suit == "h" or suit == "r" or suit == "i" or suit == "p":
                            break
            else:
                input_dat = test_input_dat(p.place, table, p=players_)
                p = players_[(leader+per)%player_n]
                print(p.place)
                if per == 0:
                    weights = [a + b for a, b in zip(p.not_played, p.choose_card(input_dat))]
                    table[p.place][turn] = p.hand[weights.index(max(weights))]
                    p.not_played[weights.index(max(weights))] = 0
                    suit = table[p.place][turn].maa
                    print(table[p.place][turn].maa, table[p.place][turn].num)
                    print("Played suit: " + suit)
                else:
                    correct_suit = [0, 0, 0, 0, 0]
                    for i in range(5):
                        if p.hand[i].maa == suit:
                            correct_suit[i] = 1
                    cor = [a + b for a, b in zip(p.not_played, correct_suit)]
                    c_v = p.choose_card(input_dat)
                    weights = [a + b for a, b in zip(cor, c_v)]
                    print(correct_suit)
                    print(c_v)
                    print(p.not_played)
                    print(weights)
                    table[p.place][turn] = p.hand[weights.index(max(weights))]
                    p.not_played[weights.index(max(weights))] = 0
                    print(table[p.place][turn].maa, table[p.place][turn].num)
        leading_card = 0
        for i in range(player_n):
            if table[i][turn].maa == suit and table[i][turn].num > leading_card:
                leading_card = table[i][turn].num
                leader = i

def test_input_dat(place, table, p=players ):
        dat = []
        for i in range(5):
            dat.append(p[place].hand[i].c_id)
        for j in range(player_n):
            for i in range(5):
                if table[j][i] == 0:
                    dat.append([0, 0, 0, 0, 0, 0])
                else:
                    dat.append(table[j][i].c_id)
        rer = np.array(dat)
        return rer.flatten()

def save_state(p, n):
    nw1, nw3, nb1, nb3 = "w1_"+str(n), "w3_"+str(n), "b1_"+str(n), "b3_"+str(n)
    np.save(nw1, p.w1)
    np.save(nw3, p.w3)
    np.save(nb1, p.b1)
    np.save(nb3, p.b3)
    
if __name__ == "__main__":
    cycle_wins = [0 for i in range(player_n)]
    cycle_number = 0
    random_wins = 0
    p = None
    t_start = time.time()
    if input("preload file? (y/n): ") == 'y':
        preload_p = True
    else:
        preload_p = False

    if preload_p:
        nm = input("File num: ")
        w1 = np.load("w1_"+nm+".npy")
        w3 = np.load("w3_"+nm+".npy")
        b1 = np.load("b1_"+nm+".npy")
        b3 = np.load("b3_"+nm+".npy")
        p = Player(0, in_neurons, hid_neurons, output_neurons, w1=w1, w3=w3, b1=b1, b3=b3)
    load_cards()
    if int(input("Training or vs (requires preload) (0/1): ")) and p is not None:
        human_vs_machine(p)
    else:
        while True:
            if p is not None:
                p, wins, t, confidence = training_cycle(cycle_number, p)
            else:
                p, wins, t, confidence = training_cycle(cycle_number)
            cycle_number += 1
            if cycle_number % 9000 == 0:
                print("State saved, time since beginning: " + str("{0:.1f}".format(((time.time()-t_start)/60)))
                      + " minutes")
                save_state(p, cycle_number)
            cycle_wins[p.place] += 1
            if p.place == 0:
                random_wins += 1
            print("Cycle "+ str(cycle_number) +" lasted: " + str("{0:.3f}".format(t)) + " seconds (" + str("{0:.1f}".format(((time.time()-t_start)/60))) + " min)")
            print("Wins: " + str(wins))
            print("Best player: " + str(p.place))
            print("Average confidence: " + str("{0:.7f}".format(confidence)))
            # print("Random wins: " + str(int(random_wins/cycle_number*1000)/10) + "%")
