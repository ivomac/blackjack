#!/usr/bin/python

from collections import deque, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import tabulate

BASEDECK = [4] * 9 + [16]
BLACKJACK = 21
ACE_PLUS = 10

PLAYER_OPTS = {
    "split": False,
    "double": False,
    "surrender": True,
}

DEALER_OPTS = {
    "target": 17,
    "peeks": False,  # NotImplemented
    "hit_soft_target": False,  # NotImplemented
}

LOG_OPTS = {
    "totals": (0, 2),
    "dealer_stand_prob": (False, 21),
    "stand_win_prob": False,
    "optimal_policy": True,
    "floatfmt": ".4f",
    "plot": True,
}


class Solver:
    """Optimal policy solver."""

    def __init__(self) -> None:
        self.policy = None
        self.values = None
        self.exp_val = None

    def optimal_policy(self):
        """Solve for optimal policy."""
        self.policy = np.zeros((HANDS.len, DECK.len), dtype=np.uint8)
        self.exp_val = np.zeros(self.policy.shape)
        for h in range(HANDS.len - 1, -1, -1):
            hand = HANDS[h]
            exp_val_stand = (
                PLAYER.prob_if_stand["win"][h] - PLAYER.prob_if_stand["lose"][h]
            )
            if not DRAWS[h]:
                self.exp_val[h] = exp_val_stand
                continue
            for dealer_card in range(DECK.len):
                cards_left = DECK - hand
                if cards_left[dealer_card]:
                    cards_left[dealer_card] -= 1
                    num_cards = DECK.sum - HANDS.sum[h] - 1
                    exp_val_hit = 0
                    total_prob = 0
                    for nxt, card_drawn in DRAWS[h]:
                        if cards_left[card_drawn]:
                            prob = cards_left[card_drawn] / num_cards
                            total_prob += prob
                            exp_val_hit += self.exp_val[nxt][dealer_card] * prob
                    exp_val_hit -= 1 - total_prob
                    if exp_val_hit > exp_val_stand[dealer_card]:
                        self.policy[h, dealer_card] = 1
                        self.exp_val[h, dealer_card] = exp_val_hit
                    else:
                        self.exp_val[h, dealer_card] = exp_val_stand[dealer_card]

                    if HANDS.sum[h] == 2:
                        # Surrender
                        if (
                            PLAYER.options.surrender
                            and self.exp_val[h, dealer_card] < -0.5
                        ):
                            self.policy[h, dealer_card] = 2
                            self.exp_val[h, dealer_card] = -0.5
                else:
                    self.exp_val[h, dealer_card] = np.nan


class Deck(np.ndarray):
    """Deck of cards"""

    def __new__(cls):
        return np.array(BASEDECK, dtype=np.uint8).view(cls)

    def __init__(self):
        self.htoi = None
        self.len = len(self)
        self.sum = np.sum(self)

    def get_hands_and_draws(self):
        """Get all possible hands and draws"""
        draws = dict()
        q = deque()

        empty_hand = np.zeros(self.len, dtype=np.uint8)
        q.append((empty_hand, 0))
        while q:
            hand, min_score = q.popleft()
            t_hand = tuple(hand)
            if t_hand in draws:
                continue
            draws[t_hand] = []
            card = 0
            min_score += 1
            while card < self.len and min_score <= BLACKJACK:
                if hand[card] < self[card]:
                    hand[card] += 1
                    draws[t_hand].append((tuple(hand), card))
                    q.append((hand.copy(), min_score))
                    hand[card] -= 1
                card += 1
                min_score += 1

        hands = Hands(*sorted_hands(draws.keys()))
        draws = [
            [(hands.htoi[dest], card) for dest, card in draws[tuple(hand)]]
            for hand in hands
        ]
        return hands, draws


def sorted_hands(hands):
    hands = sorted(hands, key=lambda x: sum(x))
    htoi = {h: i for i, h in enumerate(hands)}
    return hands, htoi


class Hands(np.ndarray):
    def __new__(cls, hands, _):
        return np.array(hands, dtype=np.uint8).view(cls)

    def __init__(self, _, htoi):
        self.htoi = htoi
        self.len = len(self)
        self.sum = np.sum(self, axis=-1, dtype=np.uint8)

    def get_scores(self):
        card_vals = np.arange(1, len(self[0]) + 1, dtype=np.uint8)
        scores = self @ card_vals
        scores_plus = scores + ACE_PLUS
        self.scores = np.where(
            np.logical_and(self[:, 0], scores_plus <= BLACKJACK),
            scores_plus,
            scores,
        )


PlayerOptions = namedtuple("PlayerOptions", ["split", "double", "surrender"])


class Player:
    def __init__(self, **options):
        self.options = PlayerOptions(**options)
        self.prob_if_stand = None

    def get_stand_value(self):
        self.prob_if_stand = np.zeros(
            (HANDS.len, DECK.len),
            dtype=[
                ("win", float),
                ("lose", float),
            ],
        )

        def get_total_prob(inds):
            v = np.full(self.prob_if_stand.shape, np.nan)
            for i in range(HANDS.len):
                for j in range(DECK.len):
                    if not any(np.isnan(DEALER.prob_stand[i, j])):
                        k = DEALER.prob_stand[i, j, inds[i] :]
                        v[i][j] = np.sum(k, axis=-1)
            return v

        val = HANDS.scores - DEALER.target
        self.prob_if_stand["win"] = 1 - get_total_prob(
            np.where(HANDS.scores > DEALER.target, val, 0)
        )
        self.prob_if_stand["lose"] = get_total_prob(
            np.where(HANDS.scores < DEALER.target, 0, val + 1)
        )


class Dealer:
    def __init__(self, target, peeks, hit_soft_target):
        self.target = target
        self.peeks = peeks
        self.hit_soft = hit_soft_target

        self.paths = None
        self.inds_stand_hands = None
        self.prob_stand = None

    def get_paths(self):
        self.paths = np.zeros_like(HANDS, dtype=np.uint8)
        for nxt, card in DRAWS[0]:
            self.paths[nxt, card] = 1

        for h in range(1, HANDS.len):
            if HANDS.scores[h] >= self.target:
                continue
            for nxt, card in DRAWS[h]:
                self.paths[nxt] += self.paths[h]

    def get_stand_hands(self):
        self.inds_stand_hands = np.logical_and(
            np.any(self.paths, axis=-1),
            HANDS.scores >= self.target,
        ).nonzero()[0]

    def get_stand_probability(self):
        deck_left = DECK - HANDS
        stand_scores = BLACKJACK - self.target + 1
        self.prob_stand = np.zeros((HANDS.len, DECK.len, stand_scores))
        for h in self.inds_stand_hands:
            num_cards_left = DECK.sum - HANDS.sum - 1
            cards_to_draw = np.nonzero(HANDS[h])[0]
            denominator = np.where(
                np.all(deck_left >= HANDS[h], axis=-1),
                1
                / np.prod(
                    [num_cards_left - i for i in range(HANDS.sum[h] - 1)],
                    axis=0,
                    dtype=float,
                ),
                0,
            )
            prob = np.tile(
                self.paths[h].astype(float),
                (HANDS.len, 1),
            )
            for fc in cards_to_draw:
                prob[:, fc] *= (
                    np.prod(
                        [
                            deck_left[:, c] - i
                            for c in cards_to_draw
                            for i in np.arange(c == fc, HANDS[h, c])
                        ],
                        axis=0,
                        dtype=float,
                    )
                    * denominator
                )
            self.prob_stand[:, :, HANDS.scores[h] - self.target] += prob


def log():
    ops = LOG_OPTS

    cond1 = np.logical_not(np.all(SOLVER.policy, axis=-1))
    cond2 = False
    for t in ops["totals"]:
        cond2 = np.logical_or(cond2, HANDS.sum == t)
    inds = np.nonzero(np.logical_and(cond1, cond2))

    charmap = {i: str(i + 1) for i in range(DECK.len)}
    charmap[9] = "K"
    charmap[0] = "A"

    hands = HANDS[inds]
    hands = [
        ["".join([charmap[i] * c for i, c in enumerate(hand) if c])] for hand in hands
    ]
    headers = ["hand"] + [charmap[i] for i in range(DECK.len)]

    def Tabulate(arr):
        return tabulate.tabulate(
            [hands[i] + k for i, k in enumerate(arr)],
            headers=headers,
            floatfmt=ops["floatfmt"],
        )

    if ops["dealer_stand_prob"][0]:
        target = ops["dealer_stand_prob"][1]
        prob_dealer = DEALER.prob_stand[inds]
        prob_dealer = prob_dealer[:, :, target - DEALER.target].astype(float).tolist()
        print(f"Dealer prob to stand at {target} (PLAYER hand vs. dealer card)")
        print(Tabulate(prob_dealer))

    if ops["stand_win_prob"]:
        prob = PLAYER.prob_if_stand["win"][inds].astype(float).tolist()
        print("\nWin Probability if stand (PLAYER hand vs. dealer card)")
        print(Tabulate(prob))

    if ops["optimal_policy"]:
        pol = SOLVER.policy[inds].astype(int).tolist()
        print("\nOptimal Policy (PLAYER hand vs. dealer card)")
        print("1=Hit  0=Stand")
        print(Tabulate(pol))

    game_exp_val = (SOLVER.exp_val[0] @ DECK) / DECK.sum
    print("\nGame EV:", game_exp_val)

    if ops["plot"]:
        strmap = {0: "S", 1: "H", 2: "U"}
        clrmap = {0: "black", 1: "white", 2: "white"}
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 8))
        ax.imshow(SOLVER.policy[inds], cmap="binary")

        ax.set_title("Optimal Policy")
        ax.set_xlabel("Dealer Card")
        ax.set_ylabel("Player Hand")
        ax.set_xticks(np.arange(DECK.len))
        ax.set_yticks(np.arange(len(hands)))
        ax.set_xticklabels(headers[1:])
        ax.set_yticklabels([hand[0] for hand in hands])
        for i in range(DECK.len):
            for j in range(len(hands)):
                ax.text(
                    i,
                    j,
                    strmap[SOLVER.policy[inds][j, i]],
                    ha="center",
                    va="center",
                    color=clrmap[SOLVER.policy[inds][j, i]],
                )

        fig.savefig("optimal_policy.png")


SOLVER = Solver()
PLAYER = Player(**PLAYER_OPTS)
DEALER = Dealer(**DEALER_OPTS)
DECK = Deck()
HANDS, DRAWS = DECK.get_hands_and_draws()

HANDS.get_scores()
DEALER.get_paths()
DEALER.get_stand_hands()
DEALER.get_stand_probability()
PLAYER.get_stand_value()
SOLVER.optimal_policy()
log()
