from SoccerNet.Downloader import SoccerNetDownloader


def test_jersey():
    downloader = SoccerNetDownloader(LocalDirectory="data/sn_jersey")
    print("Trying to download jersey-number train...")
    success = downloader.downloadDataTask(task="jersey-number", split=["train"])
    print(f"Success: {success}")


if __name__ == "__main__":
    test_jersey()
