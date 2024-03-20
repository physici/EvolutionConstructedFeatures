# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:57:12 2023

@author: rainer.jacob

Main library for the ECO feature algorithm according to
http://dx.doi.org/10.1016/j.patcog.2013.06.002
"""
import numpy as np

from . import image_operators as image_operators

# import image_operators as image_operators
from typing import List, Any
from random import randint, choice, random, choices
import inspect
import copy
from numpy.typing import NDArray
from scipy.stats import qmc
import statistics

# get a list of possible operators as functions
# there respective initializers are called later
genes = inspect.getmembers(image_operators, inspect.isclass)
genes = [x for x in genes if x[0] != "NDArray"]

# max image dimensions
max_width: int = 866
max_height: int = 1155

# limit the length of the genom for each creature
max_genom_length = 8

fixed_rois = False


def create_creatures(
    population_size: int, fixed_size: bool = False
) -> List[List[Any]]:
    """
    Create inital creatures by randomly selecting image operators

    Parameters
    ----------
    population_size : int
        Number of desired creatures..
    fixed_size : bool
        Switch to constrain the AOI to the full image size

    Returns
    -------
    List[List[Any]]
        List of creatures.

    """
    global fixed_rois
    creatures: List[List[Any]] = []
    sampler = qmc.LatinHypercube(d=len(genes))
    sample = sampler.integers(
        l_bounds=0, u_bounds=len(genes) - 1, n=population_size
    )
    sample = sample.tolist()
    idx = 0
    while len(creatures) < population_size:
        genom_length = randint(1, max_genom_length)
        genom = []
        if not fixed_size:
            x1 = randint(0, max_width - 30)
            x2 = randint(x1 + 10, max_width - 1)
            y1 = randint(0, max_height - 30)
            y2 = randint(y1 + 10, max_height - 1)
            genom.append(x1)
            genom.append(x2)
            genom.append(y1)
            genom.append(y2)
        else:
            genom.append(0)
            genom.append(max_width - 1)
            genom.append(0)
            genom.append(max_height - 1)
        for idy in range(genom_length):
            # obj = genes[sample[idx][idy]]
            obj = choice(genes)
            my_class = obj[1]()
            genom.append(my_class)

        creatures.append(genom)
        idx += 1

    fixed_rois = fixed_size

    return creatures


def mutate_creatures(
    creatures: List[List[Any]],
    mutation_probability: float = 0.3,
    fixed_size: bool = False,
) -> List[List[Any]]:
    """
    Randomly mutate the parameters of the image operators in the creatures.

    Parameters
    ----------
    creatures : List[List[Any]]
        List of creatures.
    mutation_probability: float
        Chance that a mutation occurs.
    fixed_size: bool
        Disable mutation of the AOI.

    Returns
    -------
    List[List[Any]]
        List of mutated creatures.

    """
    for genom in creatures:
        if not fixed_size:
            if random() < mutation_probability:
                x1 = randint(0, max_width - 40)
                genom[0] = x1
                x1 = randint(x1 + 10, max_width - 30)
                genom[1] = x1
            if random() < mutation_probability:
                x1 = randint(0, max_height - 40)
                genom[2] = x1
                x1 = randint(x1 + 10, max_height - 30)
                genom[3] = x1
        for gene in genom[4:]:
            gene.mutate(mutation_probability)

    return creatures


def crossover_creatures(creatures: List[List[Any]]) -> List[List[Any]]:
    """
    Randomly cross-over creature genomes.

    Parameters
    ----------
    creatures : List[List[Any]]
        List of creatures.

    Returns
    -------
    List[Any]
        A new creature.

    """
    male = choice(creatures)
    female = choice(creatures)

    split_male = randint(4, len(male) - 1)
    split_female = randint(4, len(female) - 1)

    child = male[:split_male] + female[split_female:]
    child = copy.deepcopy(child)

    return child


def apply_creature(
    creatures: List[Any], img: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """
    Apply creature to an image and obtaine the transformed image.

    Parameters
    ----------
    creature : List[Any]
        Creature that should be applied to the image.
    img : NDArray[np.float16]
        Gray-image, input.

    Returns
    -------
    result : NDArray[np.uint8]
        The transformed image.

    """
    tmp = img.copy()
    tmp = tmp[creatures[2] : creatures[3], creatures[0] : creatures[1]]
    for gene in creatures[4:]:
        try:
            tmp = gene.apply(tmp)
        except:
            return tmp

    return tmp


def tournament_selection(
    fitness_scores: List[float], tournament_size: int = 2
) -> int:
    """
    Tournament selection algorithm to select best creatures from the population
    by small tournaments.

    Parameters
    ----------
    fitness_scores : List[float]
        Fitness scores of the individual creatures.
    tournament_size : int, optional
        Size of the tournament. The default is 2.

    Returns
    -------
    int
        Index of the winner of the tournament in the population.

    """
    population = len(fitness_scores) - 1
    best = None
    for tournament in range(tournament_size):
        idx = randint(0, population)
        if best is None or fitness_scores[idx] > fitness_scores[best]:
            best = idx
    return best


def calculate_similarity(
    population: List[List[Any]], tournament_size: int = 3
) -> float:
    """
    Calculate the similarity of a population based on a tournament selection
    to prevent having to calculate all possible combinations.

    Parameters
    ----------
    population : List[List[Any]]
        The creaturese to check.
    tournament_size : int, optional
        Size of the tournmanet. The default is 3.

    Returns
    -------
    float
        The similarity score of the current population.

    """
    similarity_scores = []

    # Perform tournament selection to compare a subset of chromosomes
    for idx, creature in enumerate(population):
        tournament = choices(range(len(population)), k=tournament_size)
        size = len(tournament)
        # Compare each creature in the tournament with the current creature
        # index based as competitors can be at the same memory position
        # when created during cross-over
        similarity_score: float = 0
        for idy in tournament:
            if not idx == idy:
                competitor = population[idy]
                # Calculate the similarity
                similarity_score += _creature_similarity(creature, competitor)
            else:
                size -= 1

        # Normalize the similarity score by dividing it by the tournament size
        if size == 0:
            size = 1
        normalized_score = similarity_score / size
        similarity_scores.append(normalized_score)

    return statistics.mean(similarity_scores)


def _creature_similarity(creature1: List[Any], creature2: List[Any]) -> float:
    """
    Calculate the similarity between two creatures.

    Parameters
    ----------
    creature1 : List[Any]
        First creature.
    creature2 : List[Any]
        Second creature.

    Returns
    -------
    float
        Similarity of both creatures.

    """
    global fixed_rois
    similarity: float = 0

    # length
    if len(creature1) == len(creature2):
        similarity += 1

    cnt = min(len(creature1), len(creature2))

    # similar image sections
    if not fixed_rois:
        for idx in range(4):
            if creature1[idx] == creature2[idx]:
                similarity += 1

    for idx in range(4, cnt):
        if type(creature1[idx]) == type(creature2[idx]):
            similarity += 1

    cnt += 1
    if fixed_rois:
        cnt -= 4
        similarity /= cnt
    else:
        similarity /= cnt

    return similarity
