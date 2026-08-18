import vame

config_path="/its/home/ap2037/ssl_vame/vame_pipeling_aug17-Aug18-2026/config.yaml"

print("starting vame pytorch training loop")

vame.train_model(config_path)