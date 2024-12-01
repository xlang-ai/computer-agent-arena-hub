from setuptools import setup, find_packages

setup(
    name="computer-agent-arena",
    version="0.1.0",
    description="Computer Agent Arena - Agent implementations for desktop automation",
    author="XLANG Lab",
    author_email="contact@xlang.ai",
    packages=find_packages(),
    install_requires=[
        "pyautogui",
        "pillow",
        "numpy",
    ],
    python_requires=">=3.8",
)