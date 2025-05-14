
log_files = [
    "log/9539362.out",
    "log/9539363.out",
    "log/9539364.out",
]

# error_type = "OSError"
error_type = "Processing"


def get_video_id(line):
    if error_type == "Processing":
        video_id = line.split(" ")[-1]
    elif error_type == "OSError":
        data_path = line.split(" ")[5]
        video_id = data_path.split("/")[-2]
    else:
        raise NotImplementedError
    return video_id


if __name__ == "__main__":
    video_ids = []
    for log_file in log_files:
        with open(log_file, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.startswith(error_type)]
        video_ids += [get_video_id(line) for line in lines]
    print(video_ids)
