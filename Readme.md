
A Blackjack solver. The player's hand composition is considered, not only the total sum.

### Parameters

- Deck (default: standard single deck)
- Blackjack amount (default: $B=21$)
- Dealer's target (or lowest amount to stand on) (default: $T=17$)
- Ace alternate extra value. Base value is 1. (default: 10)

The dealer always stands on reaching soft score $T$.

### Todo

- Add double down
- Add surrender
- Add split
- Add option for dealer to peek for blackjack
- Add option for dealer to hit at soft 17

### Blackjack solver - rough outline

The steps to determine an optimal policy using dynamic programming:

1. List all valid hands (all hands that have not busted), including empty and 1-card hands.

2. Find the probability $D_s(h, c, t)$ that the dealer will stand at $t = T,...,B$, for each player hand $h$ and each revealed dealer card $c$.

3. The EV (expected value) of standing at state $(h, d)$ is

$$
EV_s(h, d) = P(\text{dealer busts}) + P(t < S(h)) - P(t > S(h))
$$
$$
= \sum_{t \geq S(h)} D_s(h, d, t) - \sum_{t > S(h)} D_s(h, d, t)
$$

where $S(h)$ is the score or sum value of hand $h$.

4. The optimal policy/value for each state $(h, c)$ can be determined in one sweep of all states going backwards from the largest hand size states to the lowest.


### Solutions

Base parameters: $EV = -0.04246$. Hit on all two-card hands not shown.

<img src="./optimal_policy.png" width="350">

