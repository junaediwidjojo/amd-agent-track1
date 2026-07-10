"""Unit tests for deterministic solvers."""

from __future__ import annotations

from app.solvers.logic_solver import solve_logic
from app.solvers.math_solver import solve_math
from app.solvers.ner_solver import solve_ner
from app.solvers.sentiment_solver import solve_sentiment


class TestMathSolver:
    def test_simple_arithmetic(self) -> None:
        result = solve_math("What is 5 + 3?")
        assert result is not None
        assert result[0] == "8"
        assert result[1] == 1.0

    def test_percentage_of(self) -> None:
        result = solve_math("What is 25% of 80?")
        assert result is not None
        assert result[0] == "20"

    def test_percentage_word_problem(self) -> None:
        result = solve_math("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?")
        assert result is not None
        assert result[0] == "144"
        assert result[1] >= 0.9

    def test_sold_wording(self) -> None:
        result = solve_math("A bakery made 300 cookies. They sold 20% in the morning and 40 in the afternoon. How many are left?")
        assert result is not None
        assert result[0] == "200"

    def test_order_of_operations(self) -> None:
        result = solve_math("What is 15 + 27 * 3?")
        assert result is not None
        assert result[0] == "96"

    def test_no_numbers_returns_none(self) -> None:
        result = solve_math("What is the capital of France?")
        assert result is None


class TestLogicSolver:
    def test_three_friends_pets(self) -> None:
        result = solve_logic(
            "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. "
            "Sam does not own the bird. Jo owns the dog. Who owns the cat?"
        )
        assert result is not None
        assert result[0] == "Sam"
        assert result[1] == 1.0

    def test_colors(self) -> None:
        result = solve_logic(
            "Alice, Bob, and Carol each have a different favorite color: red, blue, green. "
            "Alice does not like green. Bob likes blue. Who likes red?"
        )
        assert result is not None
        assert result[0] == "Alice"

    def test_four_athletes(self) -> None:
        result = solve_logic(
            "Four athletes compete in track, swim, bike, and run. "
            "Tom does not do swim. Sara does track. Mike does bike. What does Alex do?"
        )
        assert result is not None
        assert result[0].lower() == "swim"

    def test_unsolvable_returns_none(self) -> None:
        result = solve_logic("What is the meaning of life?")
        assert result is None


class TestSentimentSolver:
    def test_positive(self) -> None:
        result = solve_sentiment("The movie was fantastic and the acting was superb.")
        assert result is not None
        assert result[0] == "Positive"

    def test_negative(self) -> None:
        result = solve_sentiment("The service was terrible and the food was cold.")
        assert result is not None
        assert result[0] == "Negative"

    def test_mixed(self) -> None:
        result = solve_sentiment("Great location but the rooms were noisy and small.")
        assert result is not None
        assert result[0] == "Mixed"

    def test_ambiguous_returns_none(self) -> None:
        result = solve_sentiment("The weather is cloudy today.")
        assert result is None


class TestNerSolver:
    def test_simple_entities(self) -> None:
        result = solve_ner(
            "Extract all named entities and their types from: "
            "Maria Sanchez joined Fireworks AI in Berlin last March."
        )
        assert result is not None
        entities = result[0]
        assert "Maria Sanchez" in entities
        assert "Person" in entities
        assert "Fireworks AI" in entities
        assert "Organization" in entities
        assert "Berlin" in entities
        assert "Location" in entities
        assert "last March" in entities
        assert "Date" in entities

    def test_extended_entities(self) -> None:
        result = solve_ner(
            "Extract entities: John Doe visited OpenAI in San Francisco last January."
        )
        assert result is not None
        entities = result[0]
        assert "John Doe" in entities
        assert "OpenAI" in entities
        assert "San Francisco" in entities
        assert "last January" in entities

    def test_no_entities_returns_none(self) -> None:
        result = solve_ner("What is the weather today?")
        assert result is None
