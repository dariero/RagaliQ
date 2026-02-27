# Security Policy

## Supported Versions

Security patches are applied to the latest release only.

| Version | Supported |
|---|---|
| 0.1.x (latest) | ✅ |
| < 0.1.0 | ❌ |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub Issues.**

RagaliQ uses GitHub's private security advisory system. To report a vulnerability:

1. Go to **[Security → Report a vulnerability](https://github.com/dariero/RagaliQ/security/advisories/new)**
2. Describe the issue: what it is, steps to reproduce, and the potential impact
3. You will receive an acknowledgment within **72 hours**

We ask that you do not disclose the vulnerability publicly until a patch has been released. We will credit reporters in the release notes unless you prefer to remain anonymous.

## Scope

**In scope:**
- Code injection or arbitrary code execution via the CLI or Python API
- Insecure handling of API keys or credentials passed to evaluators
- Prompt injection vulnerabilities in evaluator prompts that could exfiltrate data
- Dependency vulnerabilities with a known CVE affecting RagaliQ users

**Out of scope:**
- Rate limiting or availability issues on external LLM APIs (Anthropic, OpenAI) — these are outside our control
- Vulnerabilities in test datasets or examples provided by users
- Social engineering

## Preferred Languages

Reports may be submitted in English.
