import concurrent.futures
import json
import time

import requests
import tqdm


def try_solution(candidate_string: str) -> str:
    sleep_counter = 1.
    while True:
        try:
            r = requests.post(
                "http://flags.fly.dev:5000/score",
                json=json.dumps({"challenge_id": "sloth", "submission": candidate_string})
            )
            return r.text
        except requests.exceptions.ConnectionError:
            print("Connection problems. Contact the CTF organizers.")
            time.sleep(sleep_counter)
            sleep_counter += 1.


def main() -> None:
    base_dir = r"C:\Users\isaia\Documents\GitHub\competitions\ai_village_ctf\ai-village-ctf\secret.sloth"
    with open(f"{base_dir}/corncob_caps_word_list.txt", "r") as f:
        all_words = [word.upper().strip() for word in f.readlines()]

    solution = None
    solution_response = None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        candidate_strings = [
            "FLAG{" + word + "}"
            for word in all_words
        ]
        future_to_candidate_string = {
            executor.submit(try_solution, candidate_string): candidate_string
            for candidate_string in candidate_strings
        }
        # Can't use concurrent.futures.as_completed without hanging or waiting for all requests to complete
        # https://tiewkh.github.io/blog/python-thread-pool-executor/
        for future, candidate_string in tqdm.tqdm(future_to_candidate_string.items()):
            if future.cancelled():
                continue

            response = future.result()
            if len(response) > 100:
                solution = candidate_string
                solution_response = response
                executor.shutdown(wait=False, cancel_futures=True)

    if solution:
        time.sleep(0.5)
        print("Success!")
        print(f"Solution: {solution}")
        print(f"Response: {solution_response}")


if __name__ == '__main__':
    main()
