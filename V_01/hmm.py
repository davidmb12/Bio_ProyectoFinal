#!/usr/bin/env python
from collections import Counter
import itertools
import math
import numpy as np
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
import unittest

# TODO: consier https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing

class AdditiveSmoothingBigramEstimator:
    """
    BigramEstimator uses sample data to build probabilities of Unigram and Bigrams.
    - https://en.wikipedia.org/wiki/N-gram
    - Uses Additive smoothing (https://en.wikipedia.org/wiki/Additive_smoothing) to
      account for non-observed elements.
    """

    def __init__(
        self, samples: Iterable[List[str]], uniqElements: Optional[int] = None
    ) -> None:
        """
        uniqElements may be provided if known, otherwise its estimated.
        """
        self.alpha: float = 1.0e-2
        self.pUnigram: Dict[str, float]
        self.pUnigramDefault: float
        self.unigramCounts: Counter
        self.uniques: int
        (
            self.pUnigram,
            self.pUnigramDefault,
            self.unigramCounts,
            self.uniques,
        ) = self._unigrams(samples, self.alpha, uniqElements)
        self.pBigram = self._bigrams(
            samples, self.unigramCounts, self.alpha, self.uniques
        )

    @staticmethod
    def _unigrams(
        samples: Iterable[List[str]], alpha: float, uniqElements: Optional[int] = None
    ) -> Tuple[Dict[str, float], float, Counter, int]:
        """
        Build unigram probabilities from sample data.
        Use additive smoothing to assign some probability to unseen elements.
        """
        unigrams: Counter = Counter(itertools.chain(*samples))
        # Additive smoothing: Assume we've seen all elements at least once
        # (times alpha) to assign non-0 probability to unseen elements.
        uniques: int = uniqElements if uniqElements else len(unigrams)
        totalCount: float = sum(unigrams.values()) + alpha * uniques
        # Estimate unigram probabilities
        pUnigram: Dict[str, float] = {
            i: (unigrams[i] + alpha) / totalCount for i in unigrams.elements()
        }
        pUnigramDefault: float = alpha / totalCount
        return pUnigram, pUnigramDefault, unigrams, uniques

    @staticmethod
    def _bigrams(
        samples: Iterable[List[str]],
        unigramCounts: Counter,
        alpha: float,
        uniques: int,
    ) -> Dict[Tuple[str, str], float]:
        """
        Build bigram probabilities from sample data.
        Use additive smoothing to assign some probability to unseen elements.
        """
        bigrams: Counter = Counter(itertools.chain(*[zip(s, s[1:]) for s in samples]))
        pBigram: Dict[Tuple[str, str], float] = {
            (i, j): (bigrams[(i, j)] + alpha) / (unigramCounts[i] + alpha * uniques)
            for i, j in bigrams.elements()
        }
        return pBigram

    def pX(self, x: str) -> float:
        """
        Return p(x) from p(x)/total
        """
        return self.pUnigram.get(x, self.pUnigramDefault)

    def pXY(self, x: str, y: str) -> float:
        """
        Return p(y|x) from p(x,y)/p(x)
        """
        return self.pBigram.get(
            (x, y),
            self.alpha
            / (self.pUnigram.get(x, self.pUnigramDefault) + self.alpha * self.uniques),
        )


class TuringGoodBigramEstimator:
    """
    BigramEstimator uses sample data to build probabilities of Unigram and Bigrams.
    """

    def __init__(self, samples: Iterable[List[str]]) -> None:
        # Get count of unigrams
        unigrams: Counter = Counter(itertools.chain(*samples))
        uniFreqFreq: Counter = Counter(unigrams.values())
        self.totalElems: int = sum(unigrams.values())
        # Make sure we've got aligned data for log fitting
        _ux, _uy = list(zip(*uniFreqFreq.items()))
        uniFFLogFit: Callable[[float], float] = self.logFit(_ux, _uy)
        # Good-Turing redistribution of mass prob to allow for unseen events
        self.adjUnigrams: Dict[str, float] = {
            unigram: (count + 1)
            * uniFreqFreq.get(count + 1, uniFFLogFit(count + 1))
            / uniFreqFreq.get(count, uniFFLogFit(count))
            for unigram, count in unigrams.items()
        }
        assert (
            uniFreqFreq[1] != 0
        ), "Need to have seen some elements only once to estimate unseen"
        self.unseenUnigrams = uniFreqFreq.get(1, uniFFLogFit(1)) / self.totalElems

        # Get count for bigrams
        bigrams: Counter = Counter(itertools.chain(*[zip(s, s[1:]) for s in samples]))
        # Counts of all bigrams we've seen x times.
        # Eg: There's 1k different bigrams we've seen x times.
        biFreqFreq: Counter = Counter(bigrams.values())
        _bx, _by = list(zip(*biFreqFreq.items()))
        biFFLogFit: Callable[[float], float] = self.logFit(_bx, _by)
        # Good-Turing redistribution of mass prob to allow for unseen events
        self.adjBigrams: Dict[Tuple[str, str], float] = {
            bigram: (count + 1)
            * biFreqFreq.get(count + 1, biFFLogFit(count + 1))
            / biFreqFreq.get(count, biFFLogFit(count))
            for bigram, count in bigrams.items()
        }
        # Estimate unseen bigram count as (0+1) * N1/N
        assert (
            biFreqFreq[1] != 0
        ), "Need to have seen some elements only once to estimate unseen"
        self.unseenBigrams = biFreqFreq.get(1, biFFLogFit(1)) / sum(bigrams.values())

    @staticmethod
    def logFit(
        xValues: Iterable[float], yValues: Iterable[float]
    ) -> Callable[[float], float]:
        [fit_a, fit_b] = np.polyfit(np.log(list(xValues)), list(yValues), 1)
        # Build a function to map points to fit curve
        return lambda x: fit_a + fit_b * math.log(x)

    def pXY(self, x: str, y: str) -> float:
        """ Return p(y|x) from p(x,y)/p(x) adjusted by Good-Turing"""
        return self.adjBigrams.get((x, y), self.unseenBigrams) / self.adjUnigrams.get(
            x, self.unseenUnigrams
        )

    def pX(self, x: str) -> float:
        return self.adjUnigrams.get(x, self.unseenUnigrams) / self.totalElems


class MarkovChain:
    "1st Order Markov Chain"

    def __init__(
        self,
        pEstimator: Union[AdditiveSmoothingBigramEstimator, TuringGoodBigramEstimator],
    ) -> None:
        self.pEstimator = pEstimator

    def pSequence(self, sequence: List[str]) -> float:
        "Calculate the probability of this sequence of events"
        if len(sequence) == 0:
            return 0.0
        logProb: float = math.log(self.pEstimator.pX(sequence[0])) + sum(
            math.log(self.pEstimator.pXY(x, y)) for x, y in zip(sequence, sequence[1:])
        )
        return math.exp(logProb)


class MarkovChainTest(unittest.TestCase):
    def test_pSequence(self):
        # Calculate some initial state probabilities (probability of any state)
        samples: List[List[str]] = [
            list("accgcgctta"),
            list("gcttagtgac"),
            list("tagccgttac"),
        ]
        # Now that we've got same fake data build a MarkovChain
        mc: MarkovChain = MarkovChain(AdditiveSmoothingBigramEstimator(samples))
        print(f'pSequence("cggt") = {mc.pSequence(list("cggt"))}')
        print(f'pSequence("gctt") = {mc.pSequence(list("gctt"))}')
        print(f'pSequence("ccgt") = {mc.pSequence(list("ccgt"))}')
        print(f'pSequence("qact") = {mc.pSequence(list("qact"))}')
        print(f'pSequence("tagt") = {mc.pSequence(list("tagt"))}')
        print(f'pSequence("ta") = {mc.pSequence(list("ta"))}')
        self.assertTrue(True)

    def test_pSequenceGoodTuring(self):
        # Calculate some initial state probabilities (probability of any state)
        samples: List[List[str]] = [
            list("accgcgctta"),
            list("gcttagtgac"),
            list("tagccgttac"),
            list("q"),  # need some odd unigram only seen once
        ]
        # Now that we've got same fake data build a MarkovChain
        mc: MarkovChain = MarkovChain(TuringGoodBigramEstimator(samples))
        print(f'pSequence("cggt") = {mc.pSequence(list("cggt"))}')
        print(f'pSequence("gctt") = {mc.pSequence(list("gctt"))}')
        print(f'pSequence("ccgt") = {mc.pSequence(list("ccgt"))}')
        print(f'pSequence("qact") = {mc.pSequence(list("qact"))}')
        print(f'pSequence("tagt") = {mc.pSequence(list("tagt"))}')
        print(f'pSequence("ta") = {mc.pSequence(list("ta"))}')
        self.assertTrue(True)


class Hmm:
    """
    Hidden Markov Model: https://en.wikipedia.org/wiki/Hidden_Markov_model
    - Unobservable states with known transitions between them.
    - Observable events (with known prob) influenced by being in a hidden state.
    """
    def __init__(
        self,
        startProb: Dict[str, float],
        stateTransitionProb: Dict[str, Dict[str, float]],
        eventEmissionProb: Dict[str, Dict[str, float]],
    ) -> None:
        self.startProb = startProb
        self.stateTransitionProb = stateTransitionProb
        self.eventEmissionProb = eventEmissionProb
        self.states = sorted(startProb)

    def viterbiDecode(self, observedEvents: List[str]):
        """
        Deduce what where the most likely states the HMM
        walked through to produce the observed events.
        - Given state-i only depends on state-i-1 and
          using the Chain (rule.https://en.wikipedia.org/wiki/Chain_rule_(probability))
            Pr(s0, s1, ..., sn) = Pr(s0) * Pr(s1 | s0) * ... * Pr(sn | sn-1)
        """
        trellis: Dict[Tuple[str, int], Dict[str, Any]] = {}
        # Set prob of each starting state given first observation/event.
        for s in self.states:
            trellis[(s, 0)] = {
                "logp": (
                    math.log(self.startProb[s]) +
                    math.log(self.eventEmissionProb[s][observedEvents[0]])
                ),
                "bp": None,
            }
        # Fill in the trellis by walking over all remaining events.
        for ev_idx, ev in enumerate(observedEvents[1:], start=1):
            for s in self.states:
                # Argmax. Find most likely transition from 'k' to 's' given chained observations.
                most_likely_path: Optional[Dict[str, Any]] = None
                for k in self.states:
                    p: float = (
                        trellis[(k, ev_idx - 1)]["logp"] +
                        math.log(self.stateTransitionProb[k][s]) +
                        math.log(self.eventEmissionProb[s][ev])
                    )
                    if most_likely_path is None or p > most_likely_path["logp"]:
                        most_likely_path = {"logp": p, "bp": k}
                assert most_likely_path is not None
                trellis[(s, ev_idx)] = most_likely_path
        # Trace back the most likely path.
        last_idx: int = len(observedEvents) - 1
        most_likely_last_state: Optional[str] = None
        for s in self.states:
            p: float = trellis[(s, last_idx)]["logp"]
            if (most_likely_last_state is None
                    or p > trellis[(most_likely_last_state, last_idx)]["logp"]):
                most_likely_last_state = s

        path: List[str] = []
        for idx in reversed(range(len(observedEvents))):
            assert most_likely_last_state is not None
            path.append(most_likely_last_state)
            most_likely_last_state = trellis[(most_likely_last_state, idx)]["bp"]

        return list(reversed(path))




class HmmTest(unittest.TestCase):
    def test_sick_patient(self):
        """
        https://youtu.be/uAT3iJpQwJ0
        https://en.wikipedia.org/wiki/Viterbi_algorithm
        """
        start_p = {"Healthy": 0.6, "Fever": 0.4}
        hidden_state_p = {
            "Healthy": {"Healthy": 0.7, "Fever": 0.3},
            "Fever": {"Healthy": 0.4, "Fever": 0.6},
        }
        event_p = {
            "Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
            "Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6},
        }
        hmm = Hmm(start_p, hidden_state_p, event_p)

        path = hmm.viterbiDecode(["normal", "cold", "dizzy"])
        print(f'viterbiDecode(["normal", "cold", "dizzy"]) = {path}')
        self.assertEqual(path, ["Healthy", "Healthy", "Fever"])

    def test_weather(self):
        """
        https://en.wikipedia.org/wiki/Hidden_Markov_model
        """
        start_p = {"Rainy": 0.6, "Sunny": 0.4}
        hidden_state_p = {
            "Rainy": {"Rainy": 0.7, "Sunny": 0.3},
            "Sunny": {"Rainy": 0.4, "Sunny": 0.6},
        }
        event_p = {
            "Rainy": {"walk": 0.1, "shop": 0.4, "clean": 0.5},
            "Sunny": {"walk": 0.6, "shop": 0.3, "clean": 0.1},
        }
        hmm = Hmm(start_p, hidden_state_p, event_p)

        path = hmm.viterbiDecode(["walk", "shop", "walk"])
        print(f'viterbiDecode(["walk", "shop", "walk"]) = {path}')
        self.assertEqual(path, ["Sunny", "Sunny", "Sunny"])

    def test_gataca(self):
        """
        https://www.cis.upenn.edu/~cis262/notes/Example-Viterbi-DNA.pdf
        """
        start_p = {"H": 0.5, "L": 0.5}
        hidden_state_p = {
            "H": {"H": 0.5, "L": 0.5},
            "L": {"H": 0.4, "L": 0.6},
        }
        event_p = {
            "H": {"A": 0.2, "C": 0.3, "G": 0.3, "T": 0.2},
            "L": {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3},
        }
        hmm = Hmm(start_p, hidden_state_p, event_p)

        path = hmm.viterbiDecode(list("GGCACTGAA"))
        print(f'viterbiDecode("GGCACTGAA") = {path}')
        self.assertEqual(path, list("HHHLLLLLL"))
    def test_proteins(self):
        start_p ={""}

if __name__ == "__main__":
    unittest.main()

# Refs
# - http://pages.cs.wisc.edu/~molla/summer_research_program/lecture5.1.pdf
# - https://web.stanford.edu/~jurafsky/slp3/A.pdf
# - https://tscheffler.github.io/teaching/2016advancednlp/slides/04-smoothing.pdf