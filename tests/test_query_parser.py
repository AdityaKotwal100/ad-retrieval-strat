"""Unit tests for QueryParser."""

import pytest
from app.services.query_parser import QueryParser, ParsedQuery


class TestBrandDetection:
    def test_single_brand(self, parser: QueryParser):
        result = parser.parse("Nike running shoes")
        assert "nike" in result.brand_terms

    def test_multiple_brands(self, parser: QueryParser):
        result = parser.parse("Nike vs Adidas running shoes")
        assert "nike" in result.brand_terms
        assert "adidas" in result.brand_terms

    def test_no_brand(self, parser: QueryParser):
        result = parser.parse("best running shoes for marathon")
        assert result.brand_terms == []

    def test_brand_case_insensitive(self, parser: QueryParser):
        result = parser.parse("NIKE shoes")
        assert "nike" in result.brand_terms


class TestPriceExtraction:
    def test_under_price(self, parser: QueryParser):
        result = parser.parse("shoes under $150")
        assert result.price_max == 150.0

    def test_less_than(self, parser: QueryParser):
        result = parser.parse("laptop less than $1000")
        assert result.price_max == 1000.0

    def test_price_range(self, parser: QueryParser):
        result = parser.parse("shoes $50-$200")
        assert result.price_min == 50.0
        assert result.price_max == 200.0

    def test_over_price(self, parser: QueryParser):
        result = parser.parse("laptop over $500")
        assert result.price_min == 500.0

    def test_no_price(self, parser: QueryParser):
        result = parser.parse("best running shoes")
        assert result.price_max is None
        assert result.price_min is None

    def test_budget_intent(self, parser: QueryParser):
        result = parser.parse("budget laptop for students")
        assert result.attribute_intents.get("price_intent") == "budget"

    def test_premium_intent(self, parser: QueryParser):
        result = parser.parse("luxury watch collection")
        assert result.attribute_intents.get("price_intent") == "premium"


class TestAttributeExtraction:
    def test_waterproof(self, parser: QueryParser):
        result = parser.parse("waterproof hiking boots")
        assert result.attribute_intents.get("waterproof") == "true"

    def test_wireless(self, parser: QueryParser):
        result = parser.parse("wireless bluetooth headphones")
        assert result.attribute_intents.get("wireless") == "true"

    def test_vegan(self, parser: QueryParser):
        result = parser.parse("vegan meal delivery")
        assert result.attribute_intents.get("dietary") == "vegan"

    def test_trail(self, parser: QueryParser):
        result = parser.parse("trail running shoes")
        assert result.attribute_intents.get("terrain") == "trail"

    def test_electric(self, parser: QueryParser):
        result = parser.parse("electric SUV")
        assert result.attribute_intents.get("fuel_type") == "electric"


class TestNegativeTerms:
    def test_not_keyword(self, parser: QueryParser):
        result = parser.parse("shoes not leather")
        assert "leather" in result.negative_terms

    def test_without_keyword(self, parser: QueryParser):
        result = parser.parse("headphones without wires")
        assert "wires" in result.negative_terms

    def test_avoid_keyword(self, parser: QueryParser):
        result = parser.parse("avoid cheap laptops")
        assert any("cheap" in term for term in result.negative_terms)

    def test_no_negatives(self, parser: QueryParser):
        result = parser.parse("best running shoes for marathon")
        assert result.negative_terms == []


class TestUserContext:
    def test_location_parsing(self, parser: QueryParser):
        result = parser.parse("shoes", context={"location": "San Francisco, CA"})
        assert result.user_location == "US-CA"

    def test_location_already_formatted(self, parser: QueryParser):
        result = parser.parse("shoes", context={"location": "US-NY"})
        assert result.user_location == "US-NY"

    def test_age(self, parser: QueryParser):
        result = parser.parse("shoes", context={"age": 28})
        assert result.user_age == 28

    def test_gender(self, parser: QueryParser):
        result = parser.parse("shoes", context={"gender": "male"})
        assert result.user_gender == "male"

    def test_interests(self, parser: QueryParser):
        result = parser.parse("shoes", context={"interests": ["fitness", "running"]})
        assert result.user_interests == ["fitness", "running"]

    def test_no_context(self, parser: QueryParser):
        result = parser.parse("shoes")
        assert result.user_location is None
        assert result.user_age is None
        assert result.user_gender is None
        assert result.user_interests == []


class TestQueryTokens:
    def test_tokenization(self, parser: QueryParser):
        result = parser.parse("Best Running Shoes!")
        assert "best" in result.query_tokens
        assert "running" in result.query_tokens
        assert "shoes" in result.query_tokens
