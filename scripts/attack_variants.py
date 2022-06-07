from textattack.shared import Attack
from textattack.constraints.overlap import (
    LevenshteinEditDistance,
    MaxWordsPerturbed
)
from textattack.constraints.pre_transformation import (
    MinWordLength,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR, GreedySearch
from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution
)

from textattack.attack_recipes import AttackRecipe


# ========== TEXTBUGGER VARIANTS ==========
class TextBuggerLi2018V1(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        goal_function = UntargetedClassification(model_wrapper)

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(UniversalSentenceEncoder(threshold=0.8))

        transformation = CompositeTransformation(
            [
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert="-",  # changed space to hyphen
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                # removed random character deletion transformation
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                WordSwapHomoglyphSwap(),
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)


class TextBuggerLi2018V2(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        goal_function = UntargetedClassification(model_wrapper)

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(UniversalSentenceEncoder(threshold=0.7))  # changed threshold from .8 to .7

        transformation = CompositeTransformation(
            [
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                WordSwapHomoglyphSwap(),
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)


class TextBuggerLi2018V3(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        goal_function = UntargetedClassification(model_wrapper)

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(UniversalSentenceEncoder(threshold=0.6))  # changed threshold from .8 to .6

        transformation = CompositeTransformation(
            [
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                WordSwapHomoglyphSwap(),
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)


class TextBuggerLi2018V4(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        goal_function = UntargetedClassification(model_wrapper)

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(UniversalSentenceEncoder(threshold=0.6))  # changed threshold from .8 to .6

        transformation = CompositeTransformation(
            [
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert="-",  # changed space to hyphen
                    skip_first_char=True,
                    skip_last_char=False,   # changed skip_last_char to False
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=False  # changed skip_last_char to False
                ),
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=False  # changed skip_last_char to False
                ),
                WordSwapHomoglyphSwap(),
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)


# ========== PRUTHI VARIANTS ==========
class Pruthi2019V1(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        goal_function = UntargetedClassification(model_wrapper)

        constraints = [
            MinWordLength(min_length=4),
            StopwordModification(),
            MaxWordsPerturbed(max_num_words=2),  # changed maximum number of words that can be perturbed from 1 to 2
            RepeatModification(),
        ]

        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterInsertion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapQWERTY(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
            ]
        )

        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)


class Pruthi2019V2(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        goal_function = UntargetedClassification(model_wrapper)

        constraints = [
            MinWordLength(min_length=4),
            StopwordModification(),
            MaxWordsPerturbed(max_num_words=1),
            RepeatModification(),
        ]

        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterInsertion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                )
                # removed QWERTY character swaps
            ]
        )

        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)


class Pruthi2019V3(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        goal_function = UntargetedClassification(model_wrapper)

        constraints = [
            MinWordLength(min_length=4),
            StopwordModification(),
            MaxWordsPerturbed(max_num_words=1),
            RepeatModification(),
        ]

        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterInsertion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapQWERTY(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
            ]
        )

        search_method = GreedyWordSwapWIR(wir_method="delete")  # changed from greedy search

        return Attack(goal_function, constraints, transformation, search_method)


class Pruthi2019V4(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        goal_function = UntargetedClassification(model_wrapper)

        constraints = [
            MinWordLength(min_length=1),  # changed from 4 to 1
            # removed stop word modification constraint
            MaxWordsPerturbed(max_num_words=2),  # changed from 1 to 2
            RepeatModification(),
        ]

        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterInsertion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                # removed QWERTY character swaps
            ]
        )

        search_method = GreedyWordSwapWIR(wir_method="delete")  # changed from greedy search

        return Attack(goal_function, constraints, transformation, search_method)


# ========== DEEPWORDBUG VARIANTS ==========
class DeepWordBugGao2018V1(AttackRecipe):

    @staticmethod
    def build(model_wrapper, use_all_transformations=True):

        goal_function = UntargetedClassification(model_wrapper)

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(LevenshteinEditDistance(30))

        if use_all_transformations:
            transformation = CompositeTransformation(
                [
                    WordSwapNeighboringCharacterSwap(),
                    WordSwapRandomCharacterSubstitution(),
                    WordSwapRandomCharacterDeletion(),
                    # removed random character insertions
                ]
            )
        else:
            transformation = WordSwapRandomCharacterSubstitution()

        search_method = GreedyWordSwapWIR()

        return Attack(goal_function, constraints, transformation, search_method)


class DeepWordBugGao2018V2(AttackRecipe):

    @staticmethod
    def build(model_wrapper, use_all_transformations=True):

        goal_function = UntargetedClassification(model_wrapper)

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(LevenshteinEditDistance(15))  # changed from 30 to 15

        if use_all_transformations:
            transformation = CompositeTransformation(
                [
                    WordSwapNeighboringCharacterSwap(),
                    WordSwapRandomCharacterSubstitution(),
                    WordSwapRandomCharacterDeletion(),
                    WordSwapRandomCharacterInsertion(),
                ]
            )
        else:
            transformation = WordSwapRandomCharacterSubstitution()

        search_method = GreedyWordSwapWIR()

        return Attack(goal_function, constraints, transformation, search_method)


class DeepWordBugGao2018V3(AttackRecipe):

    @staticmethod
    def build(model_wrapper, use_all_transformations=False):  # changed to False to use only the one transformation

        goal_function = UntargetedClassification(model_wrapper)

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(LevenshteinEditDistance(30))

        if use_all_transformations:
            transformation = CompositeTransformation(
                [
                    WordSwapNeighboringCharacterSwap(),
                    WordSwapRandomCharacterSubstitution(),
                    WordSwapRandomCharacterDeletion(),
                    WordSwapRandomCharacterInsertion(),
                ]
            )
        else:
            transformation = WordSwapRandomCharacterSubstitution()

        search_method = GreedyWordSwapWIR()

        return Attack(goal_function, constraints, transformation, search_method)


class DeepWordBugGao2018V4(AttackRecipe):

    @staticmethod
    def build(model_wrapper, use_all_transformations=True):

        goal_function = UntargetedClassification(model_wrapper)

        constraints = [StopwordModification()]  # removed repeat modification constraint
        constraints.append(LevenshteinEditDistance(60))  # changed from 30 to 60

        if use_all_transformations:
            transformation = CompositeTransformation(
                [
                    WordSwapNeighboringCharacterSwap(),
                    WordSwapRandomCharacterSubstitution(),
                    WordSwapRandomCharacterDeletion(),
                    # remove random character insertion transformation
                ]
            )
        else:
            transformation = WordSwapRandomCharacterSubstitution()

        search_method = GreedyWordSwapWIR()

        return Attack(goal_function, constraints, transformation, search_method)