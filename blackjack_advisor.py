"""Blackjack advisor.

Enter your hand and the dealer's up card; the program computes the exact
win / push / lose probabilities and expected value (EV) of every legal move
(stand, hit, double, split, insurance) and recommends the best one.

Assumptions (configurable with command-line flags):
  * Shoe of N decks (default 1); the cards on the table are removed from it.
  * Dealer stands on soft 17 (use --h17 for hit soft 17).
  * Dealer peeks for blackjack, so move EVs assume the dealer does NOT have
    blackjack (insurance is evaluated before the peek).
  * Double on any first two cards, double after split allowed, no re-split,
    split aces receive one card each.
  * Blackjack pays 3:2.
Card probabilities are held fixed after removing the visible cards, which is
very accurate for multi-deck shoes.
"""

import argparse
import re

ACE = 11
CARD_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, ACE]
CARD_NAMES = {ACE: "A", 10: "10"}


def parse_cards(text):
    """Turn input like 'A 7', 'K,5' or 'a7' into card values."""
    tokens = re.findall(r"10|[2-9ajqkt]", text.lower())
    leftover = re.sub(r"10|[2-9ajqkt]|[\s,]", "", text.lower())
    if leftover:
        raise ValueError(f"Unrecognized card(s): {leftover!r}")
    cards = []
    for t in tokens:
        if t == "a":
            cards.append(ACE)
        elif t in ("10", "t", "j", "q", "k"):
            cards.append(10)
        else:
            cards.append(int(t))
    return cards


def add_card(total, soft, card):
    """Add a card to a hand described by (total, soft)."""
    if card == ACE:
        if soft:
            total += 1
        else:
            total += 11
            soft = True
    else:
        total += card
    if total > 21 and soft:
        total -= 10
        soft = False
    return total, soft


def hand_value(cards):
    total, soft = 0, False
    for c in cards:
        total, soft = add_card(total, soft, c)
    return total, soft


def card_name(c):
    return CARD_NAMES.get(c, str(c))


class Result:
    """Win/push/lose probabilities and EV (in units of the original bet)."""

    def __init__(self, win=0.0, push=0.0, lose=0.0, ev=0.0):
        self.win, self.push, self.lose, self.ev = win, push, lose, ev

    def add(self, other, weight):
        self.win += other.win * weight
        self.push += other.push * weight
        self.lose += other.lose * weight
        self.ev += other.ev * weight


BUST = Result(lose=1.0, ev=-1.0)


class Advisor:
    def __init__(self, player_cards, dealer_card, decks=1, h17=False):
        self.dealer_card = dealer_card
        self.h17 = h17

        counts = {v: 4 * decks for v in CARD_VALUES}
        counts[10] = 16 * decks
        for c in player_cards + [dealer_card]:
            if counts[c] == 0:
                raise ValueError(f"Too many {card_name(c)}s for a {decks}-deck shoe")
            counts[c] -= 1
        n = sum(counts.values())
        self.probs = {v: counts[v] / n for v in CARD_VALUES}

        self.dealer_dist = self._dealer_distribution()
        self._stand_cache = {}
        self._play_cache = {}

    # ---------- dealer ----------
    def _dealer_from(self, total, soft, cache):
        key = (total, soft)
        if key in cache:
            return cache[key]
        if total > 21:
            dist = {"bust": 1.0}
        elif total >= 18 or (total == 17 and not (soft and self.h17)):
            dist = {total: 1.0}
        else:
            dist = {}
            for c, p in self.probs.items():
                sub = self._dealer_from(*add_card(total, soft, c), cache)
                for k, v in sub.items():
                    dist[k] = dist.get(k, 0.0) + p * v
        cache[key] = dist
        return dist

    def _dealer_distribution(self):
        """Dealer final-total distribution, given the dealer has no blackjack."""
        up = self.dealer_card
        hole_probs = dict(self.probs)
        if up == ACE:
            hole_probs[10] = 0.0
        elif up == 10:
            hole_probs[ACE] = 0.0
        norm = sum(hole_probs.values())

        cache, dist = {}, {}
        start = add_card(0, False, up)
        for c, p in hole_probs.items():
            if p == 0:
                continue
            sub = self._dealer_from(*add_card(*start, c), cache)
            for k, v in sub.items():
                dist[k] = dist.get(k, 0.0) + (p / norm) * v
        return dist

    # ---------- player ----------
    def stand(self, total):
        if total in self._stand_cache:
            return self._stand_cache[total]
        win = push = lose = 0.0
        for outcome, p in self.dealer_dist.items():
            if outcome == "bust" or outcome < total:
                win += p
            elif outcome == total:
                push += p
            else:
                lose += p
        r = Result(win, push, lose, win - lose)
        self._stand_cache[total] = r
        return r

    def hit(self, total, soft):
        """Take one card, then continue playing optimally (hit/stand)."""
        r = Result()
        for c, p in self.probs.items():
            t, s = add_card(total, soft, c)
            r.add(BUST if t > 21 else self.best_play(t, s), p)
        return r

    def best_play(self, total, soft):
        """Best of stand/hit (no doubling) from this state."""
        key = (total, soft)
        if key not in self._play_cache:
            best = self.stand(total)
            if total < 21:
                h = self.hit(total, soft)
                if h.ev > best.ev:
                    best = h
            self._play_cache[key] = best
        return self._play_cache[key]

    def double(self, total, soft):
        r = Result()
        for c, p in self.probs.items():
            t, _ = add_card(total, soft, c)
            r.add(BUST if t > 21 else self.stand(t), p)
        r.ev *= 2
        return r

    def split_hand(self, card):
        """Result for ONE of the two split hands (EV per hand)."""
        r = Result()
        start = add_card(0, False, card)
        for c, p in self.probs.items():
            t, s = add_card(*start, c)
            if card == ACE:
                r.add(self.stand(t), p)  # split aces get one card only
            else:
                options = [self.best_play(t, s), self.double(t, s)]
                r.add(max(options, key=lambda x: x.ev), p)
        return r

    def insurance(self):
        """EV of insurance per original bet (insurance costs half the bet)."""
        p_bj = self.probs[10]
        return p_bj, 0.5 * (2 * p_bj - (1 - p_bj))


def fmt_pct(x):
    return f"{x * 100:6.1f}%"


def analyze(player_cards, dealer_card, decks, h17):
    total, soft = hand_value(player_cards)
    hand_str = " ".join(card_name(c) for c in player_cards)
    kind = "soft" if soft else "hard"
    print(f"\nYour hand: {hand_str}  ({kind} {total})    Dealer shows: {card_name(dealer_card)}")

    if total > 21:
        print("You're already bust.")
        return

    adv = Advisor(player_cards, dealer_card, decks, h17)
    two_cards = len(player_cards) == 2

    if two_cards and total == 21:
        print("BLACKJACK! Stand – it pays 3:2.")
        if dealer_card == ACE:
            p_bj, _ = adv.insurance()
            print(f"Dealer has blackjack {fmt_pct(p_bj).strip()} of the time; taking even money "
                  "guarantees +1.0 vs. an average of "
                  f"{1.5 * (1 - p_bj):+.3f} if you decline.")
        return

    results = {"Stand": adv.stand(total)}
    if total < 21:
        results["Hit"] = adv.hit(total, soft)
    if two_cards:
        results["Double"] = adv.double(total, soft)
    is_pair = two_cards and player_cards[0] == player_cards[1]
    if is_pair:
        per_hand = adv.split_hand(player_cards[0])
        results["Split"] = Result(per_hand.win, per_hand.push, per_hand.lose, 2 * per_hand.ev)

    print("\n(Assuming dealer does not have blackjack)")
    print(f"{'Move':<10}{'Win':>8}{'Push':>8}{'Lose':>8}{'EV':>10}")
    print("-" * 44)
    best_name = max(results, key=lambda k: results[k].ev)
    for name, r in results.items():
        marker = "  <-- BEST" if name == best_name else ""
        print(f"{name:<10}{fmt_pct(r.win):>8}{fmt_pct(r.push):>8}{fmt_pct(r.lose):>8}{r.ev:>+10.3f}{marker}")

    if "Hit" in results:
        bust_next = sum(p for c, p in adv.probs.items() if add_card(total, soft, c)[0] > 21)
        print(f"\nChance of busting on the next card: {fmt_pct(bust_next).strip()}")
    if "Split" in results:
        print("Split win/push/lose are per hand; EV is for both hands combined.")
    print("EV is in units of your original bet (e.g. -0.10 = lose 10% of your bet on average).")

    if dealer_card == ACE:
        p_bj, ins_ev = adv.insurance()
        verdict = "TAKE insurance" if ins_ev > 0 else "Do NOT take insurance"
        print(f"\nInsurance: dealer blackjack chance {fmt_pct(p_bj).strip()}, "
              f"EV {ins_ev:+.3f} per original bet -> {verdict}")
    else:
        print("\nInsurance: not offered (dealer isn't showing an Ace)")

    print(f"\n>>> Best move: {best_name.upper()}")


def main():
    parser = argparse.ArgumentParser(description="Blackjack best-move advisor")
    parser.add_argument("--decks", type=int, default=1, help="number of decks in the shoe (default 1)")
    parser.add_argument("--h17", action="store_true", help="dealer hits soft 17 (default: stands)")
    args = parser.parse_args()

    rules = "hits" if args.h17 else "stands on"
    print(f"Blackjack Advisor – {args.decks} deck(s), dealer {rules} soft 17")
    print("Enter cards like 'A 7', 'K,5' or '10 6'. Type 'quit' or 'exit' to quit.")

    while True:
        try:
            raw = input("\nYour hand: ").strip()
            if raw.lower() in ("quit", "exit"):
                break
            player = parse_cards(raw)
            if len(player) < 2:
                print("Please enter at least two cards.")
                continue
            raw = input("Dealer up card: ").strip()
            if raw.lower() in ("quit", "exit"):
                break
            dealer = parse_cards(raw)
            if len(dealer) != 1:
                print("Please enter exactly one dealer card.")
                continue
            analyze(player, dealer[0], args.decks, args.h17)
        except ValueError as e:
            print(f"Error: {e}")
        except (EOFError, KeyboardInterrupt):
            print()
            break


if __name__ == "__main__":
    main()
