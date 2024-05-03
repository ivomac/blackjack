#!/usr/bin/python

from collections import deque, namedtuple

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


def main():

    ops = {
        "deck": 9 * [4] + [16],
        "blackjack": 21,
        "dealer_target": 17,
        "ace_plus": 10,
        "surrender": True,
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
    solver.hands.get_scores(solver.blackjack, solver.ace_plus)
    solver.hands.get_dealer_paths(solver.dealer_target)
    solver.dealer_stand_probability()
    solver.stand_value()
    solver.optimal_policy()

    solver.log(print_ops)
    return


class DP_solver:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.deck = Deck(self.deck)
        for p in (
            "prob_dealer_stand_at",
            "prob_if_stand",
            "policy",
            "values",
        ):
            setattr(self, p, None)

    def valid_hands(self):
        futures = dict()
        q = deque()

        empty_hand = np.zeros(self.deck.len, dtype=np.uint8)
        q.append((empty_hand, 0))
        while q:
            hand, min_score = q.popleft()
            t_hand = tuple(hand)
            if t_hand in futures:
                continue
            futures[t_hand] = []
            card = 0
            min_score += 1
            while card < self.deck.len and min_score <= self.blackjack:
                if hand[card] < self.deck[card]:
                    hand[card] += 1
                    futures[t_hand].append((tuple(hand), card))
                    q.append((hand.copy(), min_score))
                    hand[card] -= 1
                card += 1
                min_score += 1
        self.hands = Hands(futures)
        return

    def dealer_stand_probability(self):
        deck_left_for_dealer = self.deck - self.hands
        stand_scores = self.blackjack - self.dealer_target + 1
        self.prob_dealer_stand_at = np.zeros(
            (self.hands.len, self.deck.len, stand_scores)
        )
        valid_dealer_hands = np.logical_and(
            np.any(self.hands.dealer_paths, axis=-1),
            self.hands.scores >= self.dealer_target,
        ).nonzero()[0]
        for h in valid_dealer_hands:
            N_cards_left_for_dealer = self.deck.sum - self.hands.sum - 1
            cards_to_draw = np.nonzero(self.hands[h])[0]
            denominator = np.where(
                np.all(deck_left_for_dealer >= self.hands[h], axis=-1),
                1
                / np.prod(
                    [
                        N_cards_left_for_dealer - i
                        for i in range(self.hands.sum[h] - 1)
                    ],
                    axis=0,
                    dtype=float,
                ),
                0,
            )
            prob = np.tile(
                self.hands.dealer_paths[h].astype(float),
                (self.hands.len, 1),
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
                :, :, self.hands.scores[h] - self.dealer_target
            ] += prob
        return

    def stand_value(self):
        self.prob_if_stand = np.zeros(
            (self.hands.len, self.deck.len),
            dtype=[
                ("win", float),
                ("lose", float),
            ],
        )

        def get_total_prob(inds):
            v = np.full(self.prob_if_stand.shape, np.nan)
            for i in range(self.hands.len):
                for j in range(self.deck.len):
                    if not any(np.isnan(self.prob_dealer_stand_at[i, j])):
                        k = self.prob_dealer_stand_at[i, j, inds[i] :]
                        v[i][j] = np.sum(k, axis=-1)
            return v

        val = self.hands.scores - self.dealer_target
        self.prob_if_stand["win"] = 1 - get_total_prob(
            np.where(self.hands.scores > self.dealer_target, val, 0)
        )
        self.prob_if_stand["lose"] = get_total_prob(
            np.where(self.hands.scores < self.dealer_target, 0, val + 1)
        )
        return

    def optimal_policy(self):
        self.policy = np.zeros((self.hands.len, self.deck.len), dtype=np.uint8)
        self.EV = np.zeros(self.policy.shape)
        for h in range(self.hands.len - 1, -1, -1):
            hand = self.hands[h]
            EV_stand = (
                self.prob_if_stand["win"][h] - self.prob_if_stand["lose"][h]
            )
            if not self.hands.futures[h]:
                self.EV[h] = EV_stand
                continue
            for dealer_card in range(self.deck.len):
                cards_left = self.deck - hand
                if cards_left[dealer_card]:
                    cards_left[dealer_card] -= 1
                    num_cards = self.deck.sum - self.hands.sum[h] - 1
                    EV_hit = 0
                    total_prob = 0
                    for nxt, card_drawn in self.hands.futures[h]:
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

                    # Surrender
                    if (
                        self.surrender
                        and self.hands.sum[h] == 2
                        and self.EV[h, dealer_card] < -0.5
                    ):
                        self.policy[h, dealer_card] = 2
                        self.EV[h, dealer_card] = -0.5
                else:
                    self.EV[h, dealer_card] = np.nan
        return

    def log(self, ops):

        cond1 = np.logical_not(np.all(self.policy, axis=-1))
        cond2 = False
        for t in ops["totals"]:
            cond2 = np.logical_or(cond2, self.hands.sum == t)
        inds = np.nonzero(np.logical_and(cond1, cond2))

        charmap = {i: str(i + 1) for i in range(self.deck.len)}
        charmap[9] = "K"
        charmap[0] = "A"

        hands = self.hands[inds]
        hands = [
            ["".join([charmap[i] * c for i, c in enumerate(hand) if c])]
            for hand in hands
        ]
        headers = ["hand"] + [charmap[i] for i in range(self.deck.len)]

        def Tabulate(arr):
            return tabulate(
                [hands[i] + k for i, k in enumerate(arr)],
                headers=headers,
                floatfmt=ops["floatfmt"],
            )

        if ops["dealer_stand_prob"][0]:
            target = ops["dealer_stand_prob"][1]
            prob_dealer = self.prob_dealer_stand_at[inds]
            prob_dealer = (
                prob_dealer[:, :, target - self.dealer_target]
                .astype(float)
                .tolist()
            )
            print(
                f"Dealer prob to stand at {target} (player hand vs. dealer card)"
            )
            print(Tabulate(prob_dealer))

        if ops["stand_win_prob"]:
            prob = self.prob_if_stand["win"][inds].astype(float).tolist()
            print("\nWin Probability if stand (player hand vs. dealer card)")
            print(Tabulate(prob))

        if ops["optimal_policy"]:
            pol = self.policy[inds].astype(int).tolist()
            print("\nOptimal Policy (player hand vs. dealer card)")
            print("1=Hit  0=Stand")
            print(Tabulate(pol))

        game_EV = (self.EV[0] @ self.deck) / self.deck.sum
        print("\nGame EV:", game_EV)

        if ops["plot"]:
            strmap = {0: "S", 1: "H", 2: "U"}
            clrmap = {0: "black", 1: "white", 2: "white"}
            fig, ax = plt.subplots(1, 1, figsize=(3.5, 8))
            ax.imshow(self.policy[inds], cmap="binary")

            ax.set_title("Optimal Policy")
            ax.set_xlabel("Dealer Card")
            ax.set_ylabel("Player Hand")
            ax.set_xticks(np.arange(self.deck.len))
            ax.set_yticks(np.arange(len(hands)))
            ax.set_xticklabels(headers[1:])
            ax.set_yticklabels([hand[0] for hand in hands])
            for i in range(self.deck.len):
                for j in range(len(hands)):
                    ax.text(
                        i,
                        j,
                        strmap[self.policy[inds][j, i]],
                        ha="center",
                        va="center",
                        color=clrmap[self.policy[inds][j, i]],
                    )

            fig.savefig("optimal_policy.png")

        return


class Deck(np.ndarray):
    def __new__(cls, deck):
        instance = np.array(deck, dtype=np.uint8).view(cls)
        instance.sum = np.sum(instance)
        instance.len = len(instance)
        return instance


Future = namedtuple("Future", ["hand_index", "card"])


class Hands(np.ndarray):

    def __new__(cls, futures):
        hands = sorted(futures.keys(), key=lambda x: sum(x))
        htoi = {h: i for i, h in enumerate(hands)}
        instance = np.array(hands, dtype=np.uint8).view(cls)
        instance.futures = [
            [Future(htoi[dest], card) for dest, card in futures[hand]]
            for hand in hands
        ]
        instance.len = len(instance)
        instance.sum = np.sum(instance, axis=-1, dtype=np.uint8)
        return instance

    def get_scores(self, blackjack, ace_plus):
        card_vals = np.arange(1, len(self[0]) + 1, dtype=np.uint8)
        V = self @ card_vals
        Vp = V + ace_plus
        self.scores = np.where(
            np.logical_and(self[:, 0], Vp <= blackjack), Vp, V
        )
        return

    def get_dealer_paths(self, dealer_target):
        self.dealer_paths = np.zeros_like(self, dtype=np.uint8)
        for nxt, card in self.futures[0]:
            self.dealer_paths[nxt, card] = 1

        for h in range(1, self.len):
            if self.scores[h] >= dealer_target:
                continue
            for nxt, _ in self.futures[h]:
                self.dealer_paths[nxt] += self.dealer_paths[h]
        return


main()
