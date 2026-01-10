from SoccerNet.Downloader import SoccerNetDownloader


def download_tracking():
    # SoccerNet-Tracking
    downloader = SoccerNetDownloader(LocalDirectory="data/soccernet_tracking")
    for split in ["train", "test", "valid", "challenge"]:
        print(f"Downloading SoccerNet-Tracking {split}...")
        try:
            downloader.downloadDataTask(task="tracking", split=[split], password="s0cc3rn3t")
        except Exception as e:
            print(f"Error downloading {split}: {e}")


if __name__ == "__main__":
    download_tracking()
