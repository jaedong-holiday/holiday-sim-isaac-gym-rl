from setuptools import setup, find_packages

setup(
    name="holiday_sim_isaac_gym_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["hydra-core"],
    author="jaedong lee",
    author_email="jaedong.lee@holiday-robotics.com",
    description="",
    url="https://https://github.com/jaedong-holiday/holiday-sim-isaac-gym-rl",
    python_requires=">=3.8, <3.9",
    entry_points={
        "console_scripts": [
            "hello-world = my_awesome_package.main:main",  # 커맨드라인 실행 가능
        ]
    },
)
