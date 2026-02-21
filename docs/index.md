---
title: "RagaliQ — LLM & RAG Evaluation Testing Framework"
description: "Open-source Python framework for automated testing of RAG pipelines. Hallucination detection, faithfulness metrics, answer relevance scoring, and retrieval quality auditing with pytest integration."
---

# RagaliQ Documentation

**RagaliQ** is an open-source LLM & RAG evaluation testing framework for Python. It provides automated hallucination detection, faithfulness metrics, answer relevance scoring, context precision, and context recall evaluation — powered by an LLM-as-Judge architecture.

## Getting Started

- [Tutorial](TUTORIAL.md) — Full walkthrough from installation to CI/CD integration
- [Examples](https://github.com/dariero/RagaliQ/tree/main/examples) — Runnable scripts and pytest examples
- [API Reference](https://github.com/dariero/RagaliQ#readme) — Complete README with all features

## Core Concepts

### Evaluators

RagaliQ ships with five built-in evaluators for comprehensive RAG pipeline testing:

- **Faithfulness** — Verifies that responses are grounded only in provided context
- **Relevance** — Checks whether the response actually answers the user's query
- **Hallucination** — Detects claims not supported by retrieved documents
- **Context Precision** — Measures retrieval quality from your vector database
- **Context Recall** — Validates that retrieved context covers all expected facts

### LLM-as-Judge

RagaliQ uses Claude or OpenAI as a semantic judge to evaluate response quality. This approach captures nuanced errors that keyword-matching and embedding similarity approaches miss.

### Pytest Integration

RagaliQ integrates natively with pytest — RAG quality tests run alongside your existing unit tests with familiar fixtures and markers.

## Installation

```bash
pip install ragaliq
```

## Links

- [GitHub Repository](https://github.com/dariero/RagaliQ)
- [PyPI Package](https://pypi.org/project/ragaliq/)
- [Changelog](https://github.com/dariero/RagaliQ/blob/main/CHANGELOG.md)
- [Issue Tracker](https://github.com/dariero/RagaliQ/issues)

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "RagaliQ",
  "description": "Open-source LLM & RAG evaluation testing framework for Python. Automated hallucination detection, faithfulness metrics, answer relevance scoring, and retrieval pipeline testing with pytest integration.",
  "applicationCategory": "DeveloperApplication",
  "operatingSystem": "Cross-platform",
  "programmingLanguage": "Python",
  "softwareVersion": "0.1.0",
  "license": "https://opensource.org/licenses/MIT",
  "url": "https://github.com/dariero/RagaliQ",
  "downloadUrl": "https://pypi.org/project/ragaliq/",
  "author": {
    "@type": "Person",
    "name": "Darie Ro",
    "url": "https://github.com/dariero"
  },
  "codeRepository": "https://github.com/dariero/RagaliQ",
  "offers": {
    "@type": "Offer",
    "price": "0",
    "priceCurrency": "USD"
  }
}
</script>
