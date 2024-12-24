from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # A is either a knight or a knave
    Or(AKnight, AKnave),
    # A cannot be both
    Not(And(AKnight, AKnave)),
    # A's statement matches reality if and only if A is a knight
    Biconditional(AKnight, And(AKnight, AKnave))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # A is either a knight or a knave
    Or(AKnight, AKnave),
    # B is either a knight or a knave
    Or(BKnight, BKnave),
    # A cannot be both
    Not(And(AKnight, AKnave)),
    # B cannot be both
    Not(And(BKnight, BKnave)),
    # A's statement matches reality if and only if A is a knight
    Biconditional(AKnight, And(AKnave, BKnave))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    # A is either a knight or a knave
    Or(AKnight, AKnave),
    # B is either a knight or a knave
    Or(BKnight, BKnave),
    # A cannot be both
    Not(And(AKnight, AKnave)),
    # B cannot be both
    Not(And(BKnight, BKnave)),
    # A's statement is true if and only if A is a knight
    Biconditional(AKnight, Biconditional(AKnight, BKnight)),
    # B's statement is true if and only if B is a knight
    Biconditional(BKnight, Not(Biconditional(AKnight, BKnight)))
)


# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # A is either a knight or a knave
    Or(AKnight, AKnave),
    # B is either a knight or a knave
    Or(BKnight, BKnave),
    # C is either a knight or a knave
    Or(CKnight, CKnave),
    # A cannot be both
    Not(And(AKnight, AKnave)),
    # B cannot be both
    Not(And(BKnight, BKnave)),
    # C cannot be both
    Not(And(CKnight, CKnave)),
    # A's statement matches reality
    Biconditional(AKnight, Or(AKnight, AKnave)),
    # B's statements match reality
    Biconditional(BKnight, And(Biconditional(AKnave, AKnave), CKnave)),
    # C's statement matches reality
    Biconditional(CKnight, AKnight)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
