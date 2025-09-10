from models.hybridNN import HybridNN
from utils.main_utils import CentroidConfigs, CentroidContainer, print_confusion_matrix
from utils.dataset_utils import  get_classic_loaders
from my_custom_ai.custom_train.train import CustomTraining
import torch






def main(configs):

    train_loader, val_loader = get_classic_loaders(
        list_of_digits=configs.digits,
        batch_size=configs.batch_size
    )


    container = CentroidContainer(configs=configs)
    model = HybridNN(configs.qb_rep, configs.qb_pe, configs.iterations, len(configs.digits)).to(configs.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

    print_confusion_matrix(model, val_loader, config.device, configs.digits)

    custom_train = CustomTraining(
        configs=configs,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        eval_on_validation=True,
        function_container=container
    )


    custom_train.train()

    model.load_state_dict(torch.load(configs.checkpoint_dir + configs.exp_name + '.pth'))
    print_confusion_matrix(model, val_loader, config.device, configs.digits)





if __name__ == '__main__':
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)


    config = CentroidConfigs(**cfg_dict)
    main(config)

