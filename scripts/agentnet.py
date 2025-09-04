import json


def count_lines_in_file(path: str) -> int:
    with open(path, "r") as f:
        return sum(1 for _ in f)


def print_schema_of_jsonl(path: str) -> None:
    with open(path, "r") as f:
        for line in f:
            print(json.loads(line).keys())


if __name__ == "__main__":
    print(
        "Total number of trajectories: ",
        sum(
            count_lines_in_file(path)
            for path in [
                "./AgentNet/agentnet_ubuntu_5k.jsonl",
                "./AgentNet/agentnet_win_mac_18k.jsonl",
            ]
        ),
    )

    print("Schema of the jsonl files: ")
    print_schema_of_jsonl("./AgentNet/agentnet_ubuntu_5k.jsonl")
