import vame

config_path="/its/home/ap2037/dissertation/ssl_vame/vame_pipeline-Aug19-2026/config.yaml"

print("starting vame pytorch training loop")

vame.train_model(config_path)