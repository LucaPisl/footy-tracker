from SoccerNet.Downloader import SoccerNetDownloader


def download_datasets():
    # SoccerNet-Tracking
    downloader = SoccerNetDownloader(LocalDirectory="data/soccernet_tracking")
    for split in ["train", "test", "valid", "challenge"]:
        print(f"Downloading SoccerNet-Tracking {split}...")
        downloader.downloadDataTask(task="tracking", split=[split])

    # sn-jersey
    # Trying jersey-2023 which seems to be the latest jersey dataset in SoccerNet
    downloader_jersey = SoccerNetDownloader(LocalDirectory="data/sn_jersey")
    for split in ["train", "test", "challenge"]:
        print(f"Downloading sn-jersey {split}...")
        downloader_jersey.downloadDataTask(task="jersey-2023", split=[split])


if __name__ == "__main__":
    download_datasets()
