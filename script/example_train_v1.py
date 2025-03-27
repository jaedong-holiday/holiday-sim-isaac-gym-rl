import argparse


import holiday_sim_isaac_gym


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="insert_holly_v1", help="env name")
    parser.add_argument(
        "--config", type=str, default="default", help="config file (yaml)"
    )    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sim = holiday_sim_isaac_gym.make_simulation(args.env, args.config, device="cuda")

    sim.reset()
    # simulation without action
    for _ in range(100):
        sim.step()
        sim.render()

    # simulation with action
    sim.reset()
    for _ in range(100):
        action = np.random.random((1, 9)).astype(np.float32)
        sim.step(
            action_dataclasses.InsertHollyActionDataClass(
                torque=torch.from_numpy(action).to("cuda")
            )
        )

        sim.render()