"""
Test configuration for PyKara.
"""

import os
import sys

import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Test fixtures
@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return """This is a sample document for testing.

It contains multiple paragraphs and sections.
Each paragraph has multiple sentences.
Some sentences are longer than others.

This is another paragraph.
It also has multiple sentences.
The content is designed to test text splitting and chunking.

Final paragraph here.
With some concluding thoughts.
End of document."""


@pytest.fixture
def wikipedia_style_text() -> str:
    """Wikipedia-style text for testing."""
    return """
Machine learning (ML) is a type of artificial intelligence.

== History ==
Machine learning has its roots in statistics and computer science.
The term was coined in 1959 by Arthur Samuel.

== Types ==
There are three main types of machine learning:
* Supervised learning
* Unsupervised learning
* Reinforcement learning

=== Supervised Learning ===
Supervised learning uses labeled training data.
Common algorithms include linear regression and decision trees.

=== Unsupervised Learning ===
Unsupervised learning finds patterns in unlabeled data.
Clustering and dimensionality reduction are common techniques.

== Applications ==
Machine learning is used in many fields:
* Computer vision
* Natural language processing
* Recommendation systems
* Autonomous vehicles
"""


@pytest.fixture
def updated_wikipedia_text() -> str:
    """Updated version of Wikipedia-style text."""
    return """
Machine learning (ML) is a type of artificial intelligence (AI).

== History ==
Machine learning has its roots in statistics, computer science, and mathematics.
The term was coined in 1959 by Arthur Samuel at IBM.
Early developments included neural networks and decision trees.

== Types ==
There are four main types of machine learning:
* Supervised learning
* Unsupervised learning
* Reinforcement learning
* Semi-supervised learning

=== Supervised Learning ===
Supervised learning uses labeled training data to learn patterns.
Common algorithms include linear regression, decision trees, and neural networks.
It's widely used for classification and regression tasks.

=== Unsupervised Learning ===
Unsupervised learning finds patterns in unlabeled data.
Clustering and dimensionality reduction are common techniques.
Examples include k-means clustering and principal component analysis.

=== Reinforcement Learning ===
Reinforcement learning learns through interaction with an environment.
It uses rewards and penalties to improve decision-making.
Applications include game playing and robotics.

== Applications ==
Machine learning is used in many fields:
* Computer vision and image recognition
* Natural language processing and chatbots
* Recommendation systems for e-commerce
* Autonomous vehicles and robotics
* Healthcare diagnostics
* Financial fraud detection
"""
