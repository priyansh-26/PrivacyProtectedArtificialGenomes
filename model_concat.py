import os
import sys
import torch


def concat_sample_models(model: str, work_dir: str):

    # define path to the model
    save_path = os.path.join(work_dir, f'models/samples/{model}')

    # load each information
    netG_state_dict = torch.load(os.path.join(save_path, f'{model}_generator.pt'), map_location=torch.device('cpu'),weights_only=True)
    netD_state_dict = torch.load(os.path.join(save_path, f'{model}_discriminator.pt'), map_location=torch.device('cpu'),weights_only=True)
    g_opt_state_dict = torch.load(os.path.join(save_path, f'{model}_g_optimizer.pt'), map_location=torch.device('cpu'),weights_only=True)
    d_opt_state_dict = torch.load(os.path.join(save_path, f'{model}_d_optimizer.pt'), map_location=torch.device('cpu'),weights_only=True)

    # save in bulk
    torch.save(
        {
            'Generator': netG_state_dict,
            'Discriminator': netD_state_dict,
            'G_optimizer': g_opt_state_dict,
            'D_optimizer': d_opt_state_dict,
        },
        os.path.join(save_path, f'{model}.pt')
    )


if __name__ == '__main__':
    args = sys.argv
    concat_sample_models(model=args[1], work_dir=args[2])
