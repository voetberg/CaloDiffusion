from calodiffusion.models.layerdiffusion import LayerDiffusion
from calodiffusion.train.train_diffusion import TrainDiffusion

class TrainLayerModel(TrainDiffusion):
    def __init__(self, flags, config, load_data = True, save_model=True, inference=False):
        super().__init__(flags, config, save_model=save_model, load_data=load_data)
        self.init_model()
        if inference: 
            self.model.set_layer_state(False)
        else: 
            self.model.set_layer_state(True)

    def init_model(self):
        self.config['checkpoint'] = self.checkpoint_folder
        self.model = LayerDiffusion(
            self.config, n_steps = self.config["NSTEPS"], loss_type = self.config['LOSS_TYPE']
        )