"""
Created on Thu Jun  1 13:43:31 2023

@author: rainer.jacob

Main script for training of the features with the help of test images.
"""

import multiprocessing as mp
import argparse
import pickle
import time
import matplotlib.pyplot as plt
from evo_features import eco_feature as eco
from typing import List, Any, Union, Tuple
from numpy.typing import NDArray
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

from random import choice
import numpy as np
import pandas as pd

from skimage import io
from skimage.transform import rescale


milestones = 5
verbose = True


class EvoFeatures:
    "Base class for training a segmentation model based on genetic algorithms"

    def __init__(
        self,
        initial_creature_pool: int = 200,
        crossover_rate: float = 0.6,
        tournament_size: int = 2,
        mutation_rate: float = 0.05,
        evolutions: int = 10,
        numproc: int = 4,
        minimum_fitness: float = 0.1,
        target_fitness: float = 0.8,
        diversity_limit: float = 0.3,
        image_height: int = 448,
        image_width: int = 448,
        max_creature_length: int = 8,
        max_depth: int = 8,
    ):
        """
        Class constructor for EvoFeatures-Class

        initial_creature_pool : int, optional
            Initial number of creatures in the gene pool. The default is 200.
        crossover_rate : float, optional
            Fraction that the tournament has in the new generation.
            The default is 0.4.
        tournament_size : int, optional
            Tournament size. The default is 3.
        mutation_rate : float, optional
            Mutation probability. The default is 0.3.
        evolutions : int, optional
            Number of generations to train. The default is 200.
        numproc : int, optional
            Number of CPUs to use. The default is 8.
        minimum_fitness : float, optional
            Minimum creature fitness to use in generations. The default is 600.
        target_fitness : float, optional
            Target fitness of a population to trigger early stop.
            The default is 1.
        diversity_limit : float, optional
            Lower limit of diversity to trigger early stop. The default is 0.2.
            Setting limit to 0 deactivates the check.
        image_height : int, optional
            Height of the input image. Default is 100
        image_width : int, optional
            Width of the input image. Default is 100
        max_creature_length : int, optional
            Maximum length of the creatures. Default is 8.
        """

        self._initial_creature_pool = initial_creature_pool
        self._crossover_rate = crossover_rate
        self._tournament_size = tournament_size
        self._mutation_rate = mutation_rate
        self._evolutions = evolutions
        self._numproc = numproc
        self._minimum_fitness = minimum_fitness
        self._target_fitness = target_fitness
        self._diversity_limit = diversity_limit
        self._max_creature_length = max_creature_length
        self._image_height = image_height
        self._image_width = image_width
        self._max_depth = max_depth

    def _split(
        self, a: List[NDArray[np.uint8]], n: int = 4
    ) -> List[List[NDArray[np.uint8]]]:
        """
        Split a given list a into n parts of approx. equal size

        Parameters
        ----------
        a : List[NDArray[np.uint8]]
            List of images to split.
        n : int, optional
            How many splits desired. The default is 4.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        k, m = divmod(len(a), n)
        val: List[List[NDArray[np.uint8]]] = [
            a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(n)
        ]
        return val

    def _read_image(self, path: Path) -> NDArray[np.uint8]:
        """
        Reads an image from disk and converts to 255 grayscale.

        Parameters
        ----------
        path : Path
            Path to image.

        Returns
        -------
        image : np.array
            The image..

        """
        img: NDArray[np.float16] = io.imread(path, as_gray=True)
        img = rescale(img, 0.25)
        if img.max() <= 1:
            image: NDArray[np.uint8] = (255 * img).astype(np.uint8)
        else:
            image: NDArray[np.uint8] = img.astype(np.uint8)

        return image

    def load_data(
        self,
        path_to_data: Path,
        number_images: int = 1000,
        train_test_fraction: float = 0.25,
        random_state: int = 42,
    ) -> None:
        """
        Load the images into memory, split into train and test dataset
        and saves the train/test sets as pickle files.

        Parameters
        ----------
        number_images : int, optional
            Number of images loaded from the classes. The default is 1000.
        train_test_fraction : float, optional
            Fraction that should be used as test set. The default is 0.25.

        Returns
        -------
        None
        """
        # load into pandas
        df = pd.read_csv(path_to_data / "classification.csv")

        classes = df["label"]
        encoder = LabelEncoder()
        encoder.fit(classes)
        classes = encoder.transform(classes)

        with open("./LabelEncoder.pkl", "wb") as file:
            pickle.dump(encoder, file)

        images = [
            self._read_image(path_to_data / file) for file in df["image_url"]
        ]

        xtrain, xtest, ytrain, ytest = train_test_split(
            images,
            classes,
            test_size=int(number_images * train_test_fraction),
            train_size=number_images,
            random_state=random_state,
        )

        with open("./train.pkl", "wb") as file:
            pickle.dump((xtrain, ytrain), file)
        with open("./test.pkl", "wb") as file:
            pickle.dump((xtest, ytest), file)

        return

    def _calc_fitness(
        self, prediction: List[int], true_vals: List[int]
    ) -> float:
        """
        Calculate the fitness of a creature by providing the prediction.
        https://stackoverflow.com/a/50671617/4141279

        Parameters
        ----------
        prediction : List[int]
            Predicted classes.
        true_vals : List[int]
            True classes.

        Returns
        -------
        float
            The fitness value.

        """
        val: float = matthews_corrcoef(true_vals, prediction)

        return val

    def _train_perceptron(
        self,
        xtraining: List[NDArray[np.uint8]],
        xtesting: List[NDArray[np.uint8]],
        ytraining: List[int],
        ytesting: List[int],
    ) -> Tuple[float, RandomForestClassifier]:
        """
        Actual function for training the perceptron (random forest classifier)
        on the given images.

        Parameters
        ----------
        xtraining : List[NDArray[np.uint8]]
            List of training images.
        xtesting : List[NDArray[np.uint8]]
            List of test images.
        ytraining : List[int]
            List of training classes.
        ytesting : List[int]
            List of test classes.

        Returns
        -------
        tuple[float, RandomForestClassifier]
            fitness, classifier

        """

        clf = RandomForestClassifier(
            random_state=0,
            class_weight="balanced",
            max_depth=self._max_depth,
            n_jobs=-1,
        )
        try:
            clf.fit(xtraining, ytraining)
        except ValueError:
            return (-10, clf)

        predict = clf.predict(xtesting)
        fitness = self._calc_fitness(predict, ytesting)

        return (fitness, clf)

    def _process_images(
        self, creature: List[Any], image: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        """
        Actual process of applying a creature to an images.

        Parameters
        ----------
        creature : List[Any]
            Creature to modify to the images.
        image : NDArray[np.uint8]
            the image.

        Returns
        -------
        processed_image : NDArray[np.uint8]
            processed image.

        """
        processed_image = eco.apply_creature(creature, image)

        return processed_image.ravel()

    def _image_preparation_worker(
        self,
        manager: "mp.Manager.Namespace",
        train_indices: List[int],
        test_indices: List[int],
        creature_queue: "mp.Queue[List[NDArray[np.uint8]]]",
        result_queue: "mp.Queue[bool]",
    ) -> None:
        """
        Background worker to appliying the creature to the images.

        Parameters
        ----------
        Manager : mp.Manager
            Multiprocessing manager holding the images.
        train_indices : List[int]
            Indices of train images that should be used in this worker
        test_indices : List[int]
            Indices of test images that should be used in this worker
        creature_queue : "mp.Queue[List[NDArray[np.uint8]]]"
            Queue holding the creature to be used.
        status_queue : "mp.Queue[bool]"
            Queue for communicating when task is done.

        Returns
        -------
        None
            DESCRIPTION.

        """
        for creature in iter(creature_queue.get, "STOP"):
            for idx in train_indices:
                img = self._process_images(creature, manager.train[idx])
                result_queue.put((1, idx, img))
            for idx in test_indices:
                img = self._process_images(creature, manager.test[idx])
                result_queue.put((2, idx, img))

    def _train_population(
        self,
        path_train_data: str,
    ) -> Path:
        """
        Wrapper to start training of a population

        Parameters
        ----------
        path_train_data: str
            Path to the training data

        Returns
        -------
        milestone_path : pathlib.Path
            Path to the results

        """
        tick = time.time()
        milestone_path = Path(".")

        train_data = Path(path_train_data) / "train.pkl"
        test_data = Path(path_train_data) / "test.pkl"

        print("loading data and creating initial creatures")
        with open(train_data, "rb") as file:
            xtr, ytr = pickle.load(file)
        with open(test_data, "rb") as file:
            xte, yte = pickle.load(file)

        manager = mp.Manager()
        manager_namespace = manager.Namespace()

        manager_namespace.train = xtr
        manager_namespace.test = xte

        # indices of images for processing
        test_indices = self._split(range(len(yte)), self._numproc)
        train_indices = self._split(range(len(ytr)), self._numproc)

        number_total_images = len(ytr) + len(yte)

        # max image dimensions
        eco.max_width = self._image_height
        eco.max_height = self._image_width

        # limit the length of the genom for each creature
        eco.max_genom_length = self._max_creature_length
        print(self._max_creature_length, eco.max_genom_length)

        first_generation = eco.create_creatures(
            self._initial_creature_pool, fixed_size=True
        )

        creatures = first_generation.copy()

        print("Creating workers")
        creature_tasks: mp.Queue[Union[List[Any], str]] = mp.Queue()
        result_queue: mp.Queue[List[NDArray[np.uint8]]] = mp.Queue()

        pool = []
        for idx in range(self._numproc):
            pool.append(
                mp.Process(
                    target=self._image_preparation_worker,
                    args=(
                        manager_namespace,
                        train_indices[idx],
                        test_indices[idx],
                        creature_tasks,
                        result_queue,
                    ),
                    daemon=True,
                )
            )

        for element in pool:
            element.start()

        print("Start Evolution")
        cnt = 0
        mean: float = 0
        means = []
        best_mean: float = 0

        for evolution in range(self._evolutions):
            # train perceptrons
            fitness = []
            fitness_landscape = []
            mean = 0
            train_images = [np.ones((1, 1), "uint8")] * len(ytr)
            test_images = [np.ones((1, 1), "uint8")] * len(yte)
            for idx, creature in enumerate(creatures):
                # if verbose:
                #     print(creature)
                for element in pool:
                    creature_tasks.put(creature)
                for number in range(number_total_images):
                    res = result_queue.get(block=True)
                    if res[0] == 1:
                        train_images[res[1]] = res[2]
                    else:
                        test_images[res[1]] = res[2]
                result = self._train_perceptron(
                    train_images,
                    test_images,
                    ytr,
                    yte,
                )

                result = list(result)
                if verbose:
                    print(
                        f"Evolution {evolution}, creature {idx}, fitness {result[0]}",
                        flush=True,
                    )

                if result[0] > self._minimum_fitness:
                    result.insert(0, idx)
                    result.append(creature)
                    fitness.append(result)
                    mean += result[1]
                    fitness_landscape.append(result[1])

            if len(fitness) == 0:
                print("Generation died out")
                break
            mean = mean / len(fitness)
            means.append(mean)

            if mean > best_mean:
                best_mean = mean
                path = milestone_path / "Evolution_best.pkl"
                data = fitness.copy()
                with open(path, "wb") as f:
                    pickle.dump(data, f)

            if mean >= self._target_fitness:
                print("Stop due to fitness goal reached")
                break

            diversity = 1 - eco.calculate_similarity(creatures)
            if diversity < self._diversity_limit and self._diversity_limit > 0:
                print("Stop due to lack of diversity")
                break

            if verbose:
                print(f"Evolution {evolution} book keeping", flush=True)

            # save intermediate results
            if cnt == milestones:
                path = (
                    milestone_path
                    / f"Evolution_{evolution}_{int(mean*1000)}.pkl"
                )
                cnt = 0
                data = fitness.copy()
                with open(path, "wb") as f:
                    pickle.dump(data, f)

            if verbose:
                print(f"Evolution {evolution} tournament", flush=True)

            # tournament
            new_generation: List[List[Any]] = []
            number = int(
                self._initial_creature_pool * (1 - self._crossover_rate)
            )
            for idx in range(number):
                indices = []
                competitors = []
                for i in range(self._tournament_size):
                    (
                        number,
                        score,
                        _,
                        _,
                    ) = choice(fitness)
                    indices.append(number)
                    competitors.append(score)

                winner = eco.tournament_selection(
                    competitors, tournament_size=self._tournament_size
                )
                winner_index = indices[winner]
                new_generation.append(creatures[winner_index])
            if verbose:
                print(f"Evolution {evolution} cross over", flush=True)

            # create the rest of the new population by cross-over of already
            # selected members
            number = self._initial_creature_pool - len(new_generation)
            childs: List[List[Any]] = []
            for idx in range(number):
                child = eco.crossover_creatures(new_generation)
                if not isinstance(child[3], int):
                    idx -= 1
                    continue
                childs.append(child)

            # combine list to greate new population
            new_generation += childs

            if verbose:
                print(f"Evolution {evolution} mutation", flush=True)

            # allow some mutation, depending on fitness spread in the population
            fitness_spread = np.array(fitness_landscape).std()
            if fitness_spread < 0.01:
                print(
                    f"Low fitness spread, increasing mutation rate", flush=True
                )
                tmp_rate = self._mutation_rate * 5
                if tmp_rate > 1:
                    tmp_rate = 1
            else:
                tmp_rate = self._mutation_rate

            new_generation = eco.mutate_creatures(
                new_generation,
                mutation_probability=tmp_rate,
                fixed_size=True,
            )

            cnt += 1
            creatures = new_generation.copy()
            print(
                f"Evolution {evolution} - mean fitness {mean} - diversity {diversity}",
                flush=True,
            )

        print("Cleaning up")
        for element in pool:
            creature_tasks.put("STOP")

        path = milestone_path / "Evolution_best.pkl"
        path_new = milestone_path / f"Evolution_best_{int(best_mean*1000)}.pkl"
        path.rename(path_new)

        for element in pool:
            element.join()

        path = milestone_path / "Training.png"
        plt.figure()
        plt.plot(range(len(means)), means)
        plt.savefig(path)
        plt.close()

        tock = time.time()
        print(tock - tick)

        return path_new

    def _fit_adaboost(
        self,
        path_train_data: str,
        path_to_population: Path,
        desired_pool_size: int = 10,
        verbose: bool = False,
        num_proc: int = 8,
    ) -> "tuple[List[Any], List[RandomForestClassifier], List[float]]":
        """
        Fit the Adaboost ensemble classifier weights.

        Parameters
        ----------
        path_to_population : Path
            Path to the pickle file containing the creatures and classifiers.
        desired_pool_size : int, optional
            Desired number of classifiers in the ensemble. The default is 10.

        Returns
        -------
        (tuple[List[Any], List[RandomForestClassifier], List[float]])
            Selected creatures, corresponding classifier and weights.

        """
        # load images
        if verbose:
            print("Loading images", flush=True)
        with open(Path(path_train_data) / "test.pkl", "rb") as file:
            xte, yte = pickle.load(file)

        if verbose:
            print("Loading trained features", flush=True)
        # load creatures and classifiers
        with open(path_to_population, "rb") as file:
            results = pickle.load(file)

        classifiers = [a[2] for a in results]
        creatures = [a[3] for a in results]
        # creatures = [creatures[i] for i in creatures_indices]

        weights = np.ones((len(xte),))
        selected_classifier_weights = []
        selected_classifier = []
        selected_creatures = []

        if verbose:
            print("Creating workers", flush=True)
        pool = mp.Pool(processes=num_proc)

        if verbose:
            print("Training Adaboost", flush=True)
        for x in range(desired_pool_size):
            # normalize the image weights, wrongly classified have heigher weight
            weights /= weights.sum()
            # some list to keep track of the classifiers performance
            error_rate = []
            example_weights = []
            for idx, clf in enumerate(classifiers):
                # print(f"{idx} out of {len(creatures)}", end="\r")
                creature = creatures[idx]
                creature = [creature] * len(xte)
                train = pool.starmap(eco.apply_creature, zip(creature, xte))
                train = [x.ravel() for x in train]
                predict = clf.predict(train)

                # get the number of wrongly classified images and
                # sum the weights
                m = (yte == predict) * 0 + (yte != predict) * 1
                m = m * weights
                m = m.sum() / len(train)
                error_rate.append(m)

                # after selecting this classifier the weights of the images
                # would have to been changed to favor wrongly detected images in
                # the next loop
                example_weight = (yte == predict) * 1 + (yte != predict) * -1
                example_weights.append(example_weight)

            # select the classifier with the lowest error rate
            errors = np.array(error_rate)
            selection_index = np.where(errors == errors.min())[0][0]
            selection = errors[selection_index]
            if selection > 0.5:
                break

            # calculate the classifier (trust) weight
            coeff = 0.5 * np.log((1 - selection) / selection)

            # re-calculate the weights of the images
            c = example_weights[selection_index]
            weights *= np.exp(-coeff * c)

            selected_creatures.append(creatures[selection_index])
            selected_classifier.append(classifiers[selection_index])
            selected_classifier_weights.append(coeff)

            if verbose:
                print(f"Selected creature {selection_index}", flush=True)

        pool.close()
        out_path = Path(".")
        with open(out_path / "adaboost.pkl", "wb") as file:
            pickle.dump(
                (
                    selected_creatures,
                    selected_classifier,
                    selected_classifier_weights,
                ),
                file,
            )

        return (
            selected_creatures,
            selected_classifier,
            selected_classifier_weights,
        )

    def _predict_multiclass_adaboost(
        self,
        images: NDArray[np.uint8],
        creatures: List[List[Any]],
        classifiers: List[RandomForestClassifier],
        weights: List[float],
    ) -> NDArray[np.float_]:
        """
        Make an ensemble prediciton for the provided images, using a set of
        classifieres and their respective adaboost weight.

        Parameters
        ----------
        images : NDArray(np.uint8)
            Images to classify.
        creatures : List[Any]
            List of creatures to apply to the images
        classifiers: List[RandomForestClassifier]
            List of classifier corresponding to the creatures.
        weights: List[float]
            List of weights for the classifiers.

        Returns
        -------
        NDArray(float)
            By the ensemble predicted classes. A threshold needs to be applied
            to obtain 0/1 notification.

        """

        n_samples = len(images)
        classes = np.zeros((n_samples, len(classifiers)))

        pool = mp.Pool(processes=8)

        # weighted predictions
        for idx, clf in enumerate(classifiers):
            # print(f"{idx} out of {len(creatures)}", end="\r")
            creature = creatures[idx]
            creature = [creature] * len(images)
            train = pool.starmap(eco.apply_creature, zip(creature, images))
            train = [x.ravel() for x in train]

            classes[:, idx] = clf.predict(train)

        classes = classes.astype(int)
        y_pred = np.zeros(n_samples)

        for i, sample in enumerate(classes):
            y_pred[i] = np.argmax(np.bincount(sample, weights=weights))

        pool.close()
        return y_pred

    def fit(self, path_train_data: str):
        path = self._train_population(path_train_data)

        (
            selected_creatures,
            selected_classifier,
            selected_classifier_weights,
        ) = self._fit_adaboost(
            path_train_data=".",
            path_to_population=path,
            desired_pool_size=10,
            num_proc=8,
        )

        with open(f"{path_train_data}/test.pkl", "rb") as file:
            xte, yte = pickle.load(file)

        with open("./LabelEncoder.pkl", "rb") as file:
            encoder = pickle.load(file)

        pred = self._predict_multiclass_adaboost(
            xte,
            selected_creatures,
            selected_classifier,
            selected_classifier_weights,
        )
        pred = [int(x) for x in pred]

        results = matthews_corrcoef(yte, pred)

        yte = encoder.inverse_transform(yte)
        pred = encoder.inverse_transform(pred)

        cnf_disp = ConfusionMatrixDisplay.from_predictions(yte, pred)
        cnf_disp.figure_.tight_layout()


class EnsembleClassifier:

    def __init__(self, path_to_adaboost: Path) -> None:

        with open(path_to_adaboost, "rb") as file:
            res = pickle.load(file)

        self._creatures: List[List[Any]] = res[0]
        self._classifiers: List[RandomForestClassifier] = res[1]
        self._weights: List[float] = res[2]

    def predict(
        self,
        images: NDArray[np.uint8],
    ) -> NDArray[np.float_]:
        """
        Make an ensemble prediciton for the provided images, using a set of
        classifieres and their respective adaboost weight.

        Parameters
        ----------
        images : NDArray(np.uint8)
            Images to classify.

        Returns
        -------
        NDArray(float)
            By the ensemble predicted classes. A threshold needs to be applied
            to obtain 0/1 notification.

        """

        n_samples = len(images)
        classes = np.zeros((n_samples, len(self._classifiers)))

        pool = mp.Pool(processes=8)

        # weighted predictions
        for idx, clf in enumerate(self._classifiers):
            # print(f"{idx} out of {len(creatures)}", end="\r")
            creature = self._creatures[idx]
            creature = [creature] * len(images)
            train = pool.starmap(eco.apply_creature, zip(creature, images))
            train = [x.ravel() for x in train]

            classes[:, idx] = clf.predict(train)

        classes = classes.astype(int)
        y_pred = np.zeros(n_samples)

        for i, sample in enumerate(classes):
            y_pred[i] = np.argmax(np.bincount(sample, weights=self._weights))

        pool.close()
        return y_pred


if __name__ == "__main__":

    mymodel = EvoFeatures(
        initial_creature_pool=10,
        crossover_rate=0.6,
        tournament_size=3,
        mutation_rate=0.05,
        evolutions=20,
        numproc=4,
        minimum_fitness=0.1,
        target_fitness=0.9,
        max_creature_length=4,
        diversity_limit=0.4,
        image_height=512,
        image_width=470,
        max_depth=8,
    )

    mymodel.load_data(
        path_to_data=Path(r"."),
        number_images=40,
        train_test_fraction=0.2,
    )

    mymodel.fit(".")
