#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate
from collections import deque


def main():

    ops = {
        "deck": 9 * [4] + [16],
        "blackjack": 21,
        "dealer_target": 17,
        "ace_plus": 10,
    }

    print_ops = {
        "totals": (0, 2),
        "dealer_stand_prob": (False, 21),
        "stand_win_prob": False,
        "optimal_policy": True,
        "floatfmt": ".4f",
        "plot": True,
    }

    solver = DP_solver(**ops)
    solver.valid_hands()
    solver.scores()
    solver.dealer_paths_to_hand()
    solver.dealer_stand_probability()
    solver.stand_value()
    solver.optimal_policy()

    log(solver, print_ops)
    return


class DP_solver:
    def __init__(self, deck, blackjack, dealer_target, ace_plus):
        self.deck = np.array(deck, dtype=np.uint8)
        self.N_card_types = len(self.deck)
        self.N_cards_in_deck = np.sum(self.deck)
        self.blackjack = blackjack
        self.dealer_target = dealer_target
        self.ace_plus = ace_plus
        for p in (
            "hands",
            "N_hands",
            "N_cards_in_hand",
            "T",
            "hand_scores",
            "N_paths_to_hand",
            "prob_dealer_stand_at",
            "prob_if_stand",
            "policy",
            "values",
        ):
            setattr(self, p, None)

    def valid_hands(self):
        T = dict()
        q = deque()

        empty_hand = np.zeros(self.N_card_types, dtype=np.uint8)
        q.append((empty_hand, 0))
        while q:
            hand, min_score = q.popleft()
            t_hand = tuple(hand)
            if t_hand in T:
                continue
            T[t_hand] = []
            card = 0
            min_score += 1
            while card < self.N_card_types and min_score <= self.blackjack:
                if hand[card] < self.deck[card]:
                    hand[card] += 1
                    T[t_hand].append((tuple(hand), card))
                    q.append((hand.copy(), min_score))
                    hand[card] -= 1
                card += 1
                min_score += 1
        hands = sorted(T.keys(), key=lambda x: sum(x))
        htoi = {h: i for i, h in enumerate(hands)}
        self.T = [
            [(htoi[dest], card) for dest, card in T[hand]] for hand in hands
        ]
        self.hands = np.array(hands, dtype=np.uint8)
        self.N_hands = len(self.hands)
        self.N_cards_in_hand = np.sum(self.hands, axis=-1, dtype=np.uint8)
        return

    def scores(self):
        card_vals = np.arange(1, self.N_card_types + 1, dtype=np.uint8)
        V = self.hands @ card_vals
        Vp = V + self.ace_plus
        self.hand_scores = np.where(
            np.logical_and(self.hands[:, 0], Vp <= self.blackjack), Vp, V
        )
        return

    def dealer_paths_to_hand(self):
        self.N_paths_to_hand = np.zeros(
            (self.N_hands, self.N_card_types), dtype=np.uint8
        )
        for nxt, card in self.T[0]:
            self.N_paths_to_hand[nxt, card] = 1

        for h in range(1, self.N_hands):
            if self.hand_scores[h] >= self.dealer_target:
                continue
            for nxt, _ in self.T[h]:
                self.N_paths_to_hand[nxt] += self.N_paths_to_hand[h]
        return

    def dealer_stand_probability(self):
        deck_left_for_dealer = self.deck - self.hands
        stand_scores = self.blackjack - self.dealer_target + 1
        self.prob_dealer_stand_at = np.zeros(
            (self.N_hands, self.N_card_types, stand_scores)
        )
        valid_dealer_hands = np.logical_and(
            np.any(self.N_paths_to_hand, axis=-1),
            self.hand_scores >= self.dealer_target,
        ).nonzero()[0]
        for h in valid_dealer_hands:
            N_cards_left_for_dealer = (
                self.N_cards_in_deck - self.N_cards_in_hand - 1
            )
            cards_to_draw = np.nonzero(self.hands[h])[0]
            denominator = np.where(
                np.all(deck_left_for_dealer >= self.hands[h], axis=-1),
                1
                / np.prod(
                    [
                        N_cards_left_for_dealer - i
                        for i in range(self.N_cards_in_hand[h] - 1)
                    ],
                    axis=0,
                    dtype=float,
                ),
                0,
            )
            prob = np.tile(
                self.N_paths_to_hand[h].astype(float),
                (self.N_hands, 1),
            )
            for fc in cards_to_draw:
                prob[:, fc] *= (
                    np.prod(
                        [
                            deck_left_for_dealer[:, c] - i
                            for c in cards_to_draw
                            for i in np.arange(c == fc, self.hands[h, c])
                        ],
                        axis=0,
                        dtype=float,
                    )
                    * denominator
                )
            self.prob_dealer_stand_at[
                :, :, self.hand_scores[h] - self.dealer_target
            ] += prob
        return

    def stand_value(self):
        self.prob_if_stand = np.zeros(
            (self.N_hands, self.N_card_types),
            dtype=[
                ("win", float),
                ("lose", float),
            ],
        )

        def get_total_prob(inds):
            v = np.full(self.prob_if_stand.shape, np.nan)
            for i in range(self.N_hands):
                for j in range(self.N_card_types):
                    if not any(np.isnan(self.prob_dealer_stand_at[i, j])):
                        k = self.prob_dealer_stand_at[i, j, inds[i] :]
                        v[i][j] = np.sum(k, axis=-1)
            return v

        val = self.hand_scores - self.dealer_target
        self.prob_if_stand["win"] = 1 - get_total_prob(
            np.where(self.hand_scores > self.dealer_target, val, 0)
        )
        self.prob_if_stand["lose"] = get_total_prob(
            np.where(self.hand_scores < self.dealer_target, 0, val + 1)
        )
        return

    def optimal_policy(self):
        self.policy = np.zeros((self.N_hands, self.N_card_types), dtype=bool)
        self.EV = np.zeros(self.policy.shape)
        for h in range(self.N_hands - 1, -1, -1):
            hand = self.hands[h]
            EV_stand = (
                self.prob_if_stand["win"][h] - self.prob_if_stand["lose"][h]
            )
            if not self.T[h]:
                self.EV[h] = EV_stand
                continue
            for dealer_card in range(self.N_card_types):
                cards_left = self.deck - hand
                if cards_left[dealer_card]:
                    cards_left[dealer_card] -= 1
                    num_cards = (
                        self.N_cards_in_deck - self.N_cards_in_hand[h] - 1
                    )
                    EV_hit = 0
                    total_prob = 0
                    for nxt, card_drawn in self.T[h]:
                        if cards_left[card_drawn]:
                            prob = cards_left[card_drawn] / num_cards
                            total_prob += prob
                            EV_hit += self.EV[nxt][dealer_card] * prob
                    EV_hit -= 1 - total_prob
                    if EV_hit > EV_stand[dealer_card]:
                        self.policy[h, dealer_card] = 1
                        self.EV[h, dealer_card] = EV_hit
                    else:
                        self.EV[h, dealer_card] = EV_stand[dealer_card]
                else:
                    self.EV[h, dealer_card] = np.nan
        return


def log(solver, ops):

    cond1 = np.logical_not(np.all(solver.policy, axis=-1))
    cond2 = False
    for t in ops["totals"]:
        cond2 = np.logical_or(cond2, solver.N_cards_in_hand == t)
    inds = np.nonzero(np.logical_and(cond1, cond2))

    charmap = {i: str(i + 1) for i in range(solver.N_card_types)}
    charmap[9] = "K"
    charmap[0] = "A"

    hands = solver.hands[inds]
    hands = [
        ["".join([charmap[i] * c for i, c in enumerate(hand) if c])]
        for hand in hands
    ]
    headers = ["hand"] + [charmap[i] for i in range(solver.N_card_types)]

    def Tabulate(arr):
        return tabulate(
            [hands[i] + k for i, k in enumerate(arr)],
            headers=headers,
            floatfmt=ops["floatfmt"],
        )

    if ops["dealer_stand_prob"][0]:
        target = ops["dealer_stand_prob"][1]
        prob_dealer = solver.prob_dealer_stand_at[inds]
        prob_dealer = (
            prob_dealer[:, :, target - solver.dealer_target]
            .astype(float)
            .tolist()
        )
        print(
            f"Dealer prob to stand at {target} (player hand vs. dealer card)"
        )
        print(Tabulate(prob_dealer))

    if ops["stand_win_prob"]:
        prob = solver.prob_if_stand["win"][inds].astype(float).tolist()
        print("\nWin Probability if stand (player hand vs. dealer card)")
        print(Tabulate(prob))

    if ops["optimal_policy"]:
        pol = solver.policy[inds].astype(int).tolist()
        print("\nOptimal Policy (player hand vs. dealer card)")
        print("1=Hit  0=Stand")
        print(Tabulate(pol))

    game_EV = (solver.EV[0] @ solver.deck) / solver.N_cards_in_deck
    print("\nGame EV:", game_EV)

    if ops["plot"]:
        strmap = {0: "S", 1: "H"}
        clrmap = {0: "black", 1: "white"}
        fig, ax = plt.subplots(1, 1, figsize=(4, 8))
        ax.imshow(solver.policy[inds], cmap="binary")

        ax.set_title("Optimal Policy (Hit or Stand?)")
        ax.set_xlabel("Dealer Card")
        ax.set_ylabel("Player Hand")
        ax.set_xticks(np.arange(solver.N_card_types))
        ax.set_yticks(np.arange(len(hands)))
        ax.set_xticklabels(headers[1:])
        ax.set_yticklabels([hand[0] for hand in hands])
        for i in range(solver.N_card_types):
            for j in range(len(hands)):
                ax.text(
                    i,
                    j,
                    strmap[solver.policy[inds][j, i]],
                    ha="center",
                    va="center",
                    color=clrmap[solver.policy[inds][j, i]],
                )

        fig.savefig("optimal_policy.png")

    return


main()
