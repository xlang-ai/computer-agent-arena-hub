# Contributing to Computer Agent Arena

As you can see, the Computer Agent Arena platform is for comparing different computer agents in real-world tasks. This repo is mainly for developers who want to introduce new agents to the arena. We provide a very friendly and use-to-use framework to plugin new agents. We value every contribution, whether it's adding new agents, fixing bugs, improving documentation, or suggesting new features.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Commit Guidelines](#commit-guidelines)
- [Development Setup](#development-setup)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [xlang.agentarena@gmail.com](mailto:xlang.agentarena@gmail.com).

## Getting Started

### Issues
- Search for existing issues before creating a new one
- Use issue templates when available
- Provide detailed information about bugs or feature requests

#### Bug Reports Should Include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)
- Environment details (OS, browser, etc.)
- Any relevant code snippets

#### Feature Requests Should Include:
- Clear description of the feature
- Use cases and benefits
- Potential implementation approach
- Any concerns or challenges

## Development Process

### Branching Strategy
- `main` - production-ready code
- `develop` - main development branch
- `feature/*` - new features
- `bugfix/*` - bug fixes
- `hotfix/*` - urgent production fixes
- `release/*` - release preparation

### Workflow
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

## Pull Request Guidelines

### PR Naming Convention
Format: `[type] Brief description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semi-colons, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### PR Process
1. Fill out the PR template completely
2. Link related issues
3. Update documentation if needed
4. Add/update tests as needed
5. Ensure CI passes
6. Request review from maintainers
7. Address review feedback

### PR Review Criteria
- Code follows project style guide
- Tests are included and passing
- Documentation is updated
- Commits are properly formatted
- No merge conflicts
- CI/CD pipelines pass

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Example:
```
feat(auth): add user authentication endpoint

- Implement JWT token generation
- Add password hashing
- Create user validation middleware

Closes #123
```

## Development Setup

1. Clone the repository
   ```bash
   git clone git@github.com:xlang-ai/Computer-Agent-Arena.git
   cd Computer-Agent-Arena
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up pre-commit hooks
   ```bash
   pre-commit install
   ```

5. Start development server
   ```bash
   python main.py
   ```

## Testing Guidelines

### Writing Tests
- Write tests for all new features
- Maintain or improve code coverage
- Follow the existing test patterns
- Include unit and integration tests

### Running Tests
```bash

```

## Documentation Guidelines

### Code Documentation
- Use clear variable and function names
- Add comments for complex logic
- Include JSDoc for public APIs
- Keep README up to date

### Technical Documentation
- Keep documentation in sync with code changes
- Use clear and concise language
- Include examples where appropriate
- Update API documentation

## Community

### Getting Help
- Check our [FAQ](link-to-faq)
- Join our [Discord/Slack] community 
- Stack Overflow tag: [project-tag]
- GitHub Discussions

### Communication Channels
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and discussions
- Slack/Discord: Real-time communication
- Email: [xlang.agentarena@gmail.com](mailto:xlang.agentarena@gmail.com) for sensitive issues

## License

    By contributing, you agree that your contributions will be licensed under the project's license.

---

Thank you for contributing to Computer Agent Arena! 🎉 
