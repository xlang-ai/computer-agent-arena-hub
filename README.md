<!-- <p align="center">
    <img src="assets/banner.png" alt="Computer Agent Arena">
</p>

<p align="center">
  <a href="https://arena.xlang.ai">Website</a> •
  <a href="https://arena.xlang.ai/blog">Blog (Coming Soon)</a> •
  <a href="#">Paper (Coming Soon)</a> •
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
    </a>
    <a href="https://www.python.org/downloads/">
        <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
    </a>
    <a href="CONTRIBUTING.md">
        <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="Contributions welcome">
    </a>
    <br/>
</p>

<p align="center">
    <img src="assets/example.png" alt="Computer Agent Arena Demo">
</p> -->

<!-- ## 📢 Updates
- 2024-12-02: Initial beta-release of [Computer Agent Arena](https://arena.xlang.ai) platform 🥳, Try it out!
- 2024-11-05: [Claude 3.5 Sonnet for computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) agent is available on the platform!
- 2024-06-07: Basic Prompt Agent is supported on the platform. -->

## 📖 Overview
The [Computer Agent Arena](https://arena.xlang.ai) is an open-ended evaluation platform designed for benchmarking LLMs/VLMs-based AI agents in real-world computer tasks across diverse domains, ranging from general desktop operations to specialized workflows, such as programming, data analysis, and multimedia editing.

This repository hosts **the source code implementations for all supported agents on the platform**, serving as a foundation to integrate and extend support for additional agents within the Computer Agent Arena ecosystem.

## 💾 Getting Started

### Installation

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:xlang-ai/Computer-Agent-Arena.git
   cd Computer-Agent-Arena
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Environment Variables**:
   Copy the example environment configuration file and fill in the necessary credentials:
   ```bash
   cp .env.example .env
   ```
   Example `.env` file:
   ```plaintext
    export API_KEYS="{\
    \"OPENAI_API_KEY\":\"YOUR_KEY_HERE\",\
    \"GENAI_API_KEY\":\"YOUR_KEY_HERE\",\
    \"ANTHROPIC_API_KEY\":\"YOUR_KEY_HERE\"\
    }"
   ```

4. **Run Tests**:
   Test the default agents to ensure the setup is working:
   ```bash
   # Activate the virtual environment
   source .env

   # Run the tests
   python test/test_agents.py
   ```

If the tests pass, your environment is ready!

### Implement Customized Agents

> 🤝 **Want to contribute?**  
> Check out our [Contributing Guide](CONTRIBUTING.md) to learn how you can plugin your agent to improve Computer Agent Arena!

### Test Customized Agents
Once you have implemented your agent, you can add it to the test suite by adding a new test function in `test/test_agents.py`.
```bash
# Activate the virtual environment
source .env

# Run the tests and pass
python test/test_agents.py
```

After testing, submit a pull request (PR) with your implementation. Refer to the [Contributing Guide](CONTRIBUTING.md) for detailed instructions.

Once your PR is submitted, email us at [here](mailto:bryanwang.nlp@gmail.com) for further details.

We really appreciate any contributions to improve Computer Agent Arena! If there are any questions, feel free to open an issue or contact us via [email](mailto:bryanwang.nlp@gmail.com).

## 📄 Citation

If you find this project useful, please consider citing our project:

```bibtex
@misc{ComputerAgentArena2025,
  title={Computer Agent Arena: Compare & Test AI Agents on Crowdsourced Real-World Computer Use Tasks},
  url={https://arena.xlang.ai},
  year={2025}
}
```
    