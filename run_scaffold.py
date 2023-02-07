import wandb
from models.config.util import get_args
from models.servers.scaffold_server import SCAFFOLDServer

if __name__ == "__main__":
    id = wandb.util.generate_id()
    run = wandb.init(
            id = id,
            # Set entity to specify your username or team name
            entity="milad-be",
            # Set the project where this run will be logged
            project='aml-project-1',
            group='scaffold',
            # Track hyperparameters and run metadata
            config=get_args())
    server = SCAFFOLDServer()
    server.run()