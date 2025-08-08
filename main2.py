from utils.config import load_config
cfg = load_config()

def main():
    # a = cfg['OPTIMIZER']['warmup_epochs']
    print(type(cfg['OPTIMIZER']['warmup_epochs']))

if __name__ == "__main__":
    main()