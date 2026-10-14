from abc import ABC, abstractmethod
import traceback
from typing import Any, Iterable, Literal, Sequence
import pprint

import ray.tune
import numpy as np
import torch
import json
import os

from datetime import datetime
from calodiffusion.utils import utils
from calodiffusion.train import evaluate
from calodiffusion.train.train import Train

class Objective(ABC): 
    @staticmethod
    @abstractmethod
    def direction() -> Literal['minimize', "maximize"]: 
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def failure() -> float: 
        "What is returned if the model has failed to train"
        raise NotImplementedError
    
    @staticmethod
    def __call__(trained_model, eval_data, kwargs) -> float:
        raise NotImplementedError

class EvalCount(Objective): 
    @staticmethod
    def direction() -> Literal['minimize', 'maximize']:
        return "minimize"
    
    @staticmethod
    def failure():
        return 10e8

    @staticmethod
    def get_forward(): 
        class ModelForward(torch.nn.Module): 
            def __init__(self, model, eval_data, sample_steps, sample_offset) -> None:
                super().__init__()
                self.model = model
                self.E, self.layers, _ = next(iter(eval_data))
                self.sample_steps = sample_steps
                self.sample_offset = sample_offset

            def __call__(self, x=None) -> Any:
                self.model.sample(
                    self.E.to(device=self.model.device),
                    layers=self.layers.to(device=self.model.device),
                    num_steps=self.sample_steps,
                    debug=False,
                    sample_offset=self.sample_offset,
                )
                
        return ModelForward

    @staticmethod
    def __call__(trained_model, eval_data, config, *args, **kwargs) -> float:
        random = np.random.default_rng()
        weight_matrix = random.random((24, 24))
        weight_matrix_compare = random.random((24, 24))

        forward = EvalCount.get_forward()(
            model=trained_model, 
            eval_data=eval_data, 
            sample_steps=config['NSTEPS'], 
            sample_offset=0) # Only doing a single sample
        
        start = datetime.now()
        forward()
        inference_time = (start - datetime.now()).total_seconds()

        start = datetime.now()
        weight_matrix*weight_matrix_compare
        reference_time = (start - datetime.now()).total_seconds()
        return inference_time/reference_time


class EvalFPD(Objective): 
    @staticmethod
    def direction() -> Literal['minimize', 'maximize']:
        return "minimize"
    
    @staticmethod
    def failure():
        return 10e8

    @staticmethod
    def __call__(trained_model, generated, energies, eval_data, config, *args, **kwargs) -> float:

        binning_dataset = trained_model.config.get("BIN_FILE", "binning_dataset.xml")
        particle = trained_model.config.get("PART_TYPE", "photon")

        fpd_calc = evaluate.FPD(
            binning_dataset, 
            particle, 
            hgcal=config.get("HGCAL", False))
        try: 
            return fpd_calc(generated=generated, energies=energies, eval_data=eval_data)
        except evaluate.FPDCalculationError:
            return EvalFPD.failure()
        
class EvalCNNMetric(Objective):
    @staticmethod
    def failure(): 
        return 1 
    
    @staticmethod
    def direction() -> Literal['minimize', 'maximize']:
        return "maximize"
    
    @staticmethod
    def __call__(trained_model, eval_data, config, *args, **kwargs):
        cnn_method = evaluate.CNNCompare(
            trained_model=trained_model, 
            config= config, 
            flags = config['flags']
        )
        return cnn_method(eval_data)

class EvalLoss(Objective): 
    @staticmethod
    def failure(): 
        return 10**5
    
    @staticmethod
    def direction():
        return "minimize"
    
    def __call__(self, trained_model, eval_data, config, *args, **kwds):
        energy, layers, data = next(iter(eval_data))

        noise = torch.randn_like(data).to(device=trained_model.device)
        data = data.to(device=trained_model.device)
        energy = energy.to(device=trained_model.device)
        layers = layers.to(device=trained_model.device)

        return trained_model.compute_loss(data=data, energy=energy, noise=noise, layers=layers).detach().cpu().numpy()


OBJECTIVES: dict[str, type[Objective]] = {
    "COUNT": EvalCount(), 
    "FPD": EvalFPD(), 
    "CNN": EvalCNNMetric(), 
    "LOSS": EvalLoss()
}

class InferenceOptimize(ray.tune.Trainable):
    def setup(self, config:dict, base_config:dict, flags:utils.dotdict, objectives: Sequence[type[Objective]], trainer) -> None:
        
        # Get data, load the model
        base_config.update(config)
        print("Using the following config:")
        pprint.pp(base_config)
        self.config = base_config
        self.n_steps = config.get("NSTEPS", 50)
        self.eval_data, _ = utils.load_data(flags, self.config, eval=True)
        self.model_instance = trainer(flags=flags, config=self.config, save_model=False, load_data=False)
        print("Creating Model with the following flags:")
        pprint.pp(flags)
        self.model_instance.init_model()

        self.objectives = [EvalFPD(), EvalCount()]
        self.objective_names = ['FDP', 'COUNT']

        self.current_step = 0

        try: 
            self.model, _, _, _, _, _  = self.model_instance.pickup_checkpoint(
                model=self.model_instance.model,
                optimizer=None,
                scheduler=None,
                early_stopper=None,
                n_epochs=0,
                restart_training=False,
            )

        except RuntimeError as err:
            print(f"Error in loading checkpoint: {err}")
            print(traceback.print_exception(err))
            self.model = None

    def evaluate(self, model, generated, energies): 
        return {
            name: obj(
                trained_model=model,
                generated=generated,
                energies=energies,
                eval_data=self.eval_data,
                config=self.config
            ) 
                for name, obj in zip(self.objective_names, self.objectives)
        }

    def step(self):
        self.current_step += 1
        try: 
            generated_samples, generated_energies = self.model.generate(
                data_loader=self.eval_data, sample_steps=self.n_steps, debug=False, sample_offset=0,
            )
            objectives = self.evaluate(
                self.model, generated_samples, generated_energies
            )
        except Exception as err:
            print("ERROR runnning generation")
            print(traceback.print_exception(err))
            objectives = {name:obj.failure() for name, obj in zip(self.objective_names, self.objectives)}
        
        objectives["step"] = self.current_step
        ray.tune.report(objectives)

        return objectives

class TrainOptimize(ray.tune.Trainable):
    def setup(self, config:dict, base_config:dict, flags:utils.dotdict, objectives: Sequence[type[Objective]], trainer) -> None:
        # Get data, load the model
        base_config.update(config)
        self.config = base_config
        self.n_steps = config.get("NSTEPS", 50)

        if "n_unet_layers" in config.keys():
            init_size = config.get("init_unet")
            n_layers = config.get("n_unet_layers")
            final_layer = int(config.get("layer_ratio") * init_size)  
            unet_layers = [init_size for _ in range(n_layers)]
            unet_layers.append(final_layer)
            
            self.config["LAYER_SIZE_UNET"] = unet_layers

        self.objectives = [EvalFPD(), EvalCount()]
        self.objective_names = ['FDP', 'COUNT']

        self.eval_data, _ = utils.load_data(flags, self.config, eval=True)

        self.flags = flags
        try: 
            self.trainer = trainer(flags=self.flags, config=self.config, save_model=False, load_data=True, inference=False)
            self.trainer.init_model()

            self.training_losses = dict()
            self.val_losses = dict()

            self.early_stopper = utils.EarlyStopper(
                patience=self.config["EARLYSTOP"], mode="val_loss", min_delta=1e-5
            )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config["LR"]))
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, factor=0.1, patience=15
            )
        except RuntimeError as err: 
            print(traceback.print_exception(err))

        self.current_epoch = 0


    def evaluate(self, model, generated, energies):
        return {
            name: obj(
                trained_model=model,
                generated=generated,
                energies=energies,
                eval_data=self.eval_data,
                config=self.config
            )
                for name, obj in zip(self.objective_names, self.objectives)
        }

    def _train(self): 
        try: 
            model, _, self.training_losses, self.val_losses, self.optimizer, self.scheduler, self.early_stopper = self.trainer.training_loop(
                self.optimizer, 
                self.scheduler, 
                self.early_stopper, 
                self.current_epoch, 
                1, 
                self.training_losses, 
                self.val_losses
            )

        except RuntimeError as err:
            print(f"Error in training model: {err}")
            model = None

        self.current_epoch += 1
        return model

    def step(self):
        self.current_step +=1
        model = self._train()

        if model is None:
            objectives = {name: obj.failure() for name, obj in zip(self.objective_names, self.objectives)}
        
        else: 
            samples, energies = model.generate(
                data_loader=self.eval_data, sample_steps=self.n_steps, debug=False, sample_offset=0,
            )
            objectives = self.evaluate(
                model, samples, energies
            )
        objectives['step'] = self.current_step
        ray.tune.report(objectives)

        return objectives

class Optimize: 
    """
    
    Optimize the training results and sampling quality. 
    Assume a config the form of: 
    {
        <Generic Unchanged Settings, datasets, etc>, 
        "OPTIMIZE": {
            "setting_1": [float_min, float_max], 
            "setting_2": [int_min, int_max], 
            "setting_3": [str_choice_1, str_choice_2, str_choice_3 ...]
            "setting_4: [True, False], 
            "SAMPLER_SETTINGS": {
                <Settings for each type of sampler. Few each sampler docs for details>
            }
        }
    }

    Exceptions - 
    * LAYER_SIZE_UNET (u-net architecture settings) - becomes a dictionary with the integer list fields: 
        {
            "init_unet": The range of the initial layer size, 
            "n_unet_layers": Range for up/down operations in the unet, 
            "layer_ratio": Range for how small the inner layer is compared to init_unet
        }
    * SAMPLER_SETTINGS (where SAMPLER="Restart") - has the following custom fields to create "RESTART_LIST": 
        {
            "RESTART_I": Integer range for number of possible restart configurations, 
            "N_RESTART": Number of steps each restart configuration can take, 
            "RESTART_K": Integer range of K parameter, 
            "RESTART_T": Range allowed for T_MIN_{i}. T_MAX_{i} is decided by taking t_min_i as the bottom of the range and t_min_i + {top of the restart_t range} as the top
        }
    """
    def __init__(self, flags, trainer, objectives: str, inference: bool = False) -> None:

        self.flags = utils.dotdict(flags)
        self.config = utils.LoadJson(self.flags.config) if isinstance(self.flags.config, str) else self.flags.config
        
        self.checkpoint_folder = os.path.join(self.flags.results_folder, self.flags.study_name, "checkpoints/")
        os.makedirs(self.checkpoint_folder, exist_ok=True)

        self.trainer = trainer

        self.objectives = []
        if isinstance(objectives, str):
            objectives = [objectives]
        for objective in objectives: 
            try: 
                self.objectives.append(OBJECTIVES[objective])
            except KeyError:
                raise ValueError(f"Objective {objective} not in {OBJECTIVES.keys()}")

        # Automatically make the results folder if it does not exist
        if not os.path.exists(self.flags.results_folder):
            os.makedirs(self.flags.results_folder)

        self.experiment_name = f"{'inference' if inference else 'train'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.inference = inference

    def _generic_param_space(self, settings: dict) -> dict:
        config_space = {}
        for setting, values in settings.items():
            if not isinstance(values, Iterable):
                continue # Ignore non-iterables

            if np.all([isinstance(i, str) for i in values]) or (True in values):
                config_space[setting] = ray.tune.choice(values)
            elif np.all([isinstance(i, int) for i in values]):
                config_space[setting] = ray.tune.randint(*values)
            else:
                config_space[setting] = ray.tune.uniform(*values)
        return config_space
    
    def make_param_space_inference(self): 
        config_space = {}

        config_space.update(self._generic_param_space(self.config.get('OPTIMIZE', {})))

        # Go through the rest of the sampler settings
        sampler_settings = self.config.get("OPTIMIZE", {}).get("SAMPLER_OPTIONS", {})
        config_space['SAMPLER_OPTIONS'] = self._generic_param_space(sampler_settings)

        print("Running with selected options")
        pprint.pp(config_space)

        return config_space
    
    def make_param_space_training(self):
        config_space = {}
        # Generic hyperparameters
        optimized_section = self.config.get("OPTIMIZE", {})
        config_space.update(self._generic_param_space(optimized_section))

        # Sub parameters for UNet architecture
        if "LAYER_SIZE_UNET" in optimized_section.keys(): 
            config_space['init_unet']  = ray.tune.qrandint(*optimized_section["LAYER_SIZE_UNET"]["init_unet"], 2)
            config_space['n_unet_layers'] = ray.tune.randint(*optimized_section["LAYER_SIZE_UNET"]["n_unet_layers"])
            config_space['layer_ratio'] = ray.tune.uniform(*optimized_section["LAYER_SIZE_UNET"]["layer_ratio"])
            
        return config_space
    
    def _run_config(self): 
        max_epochs = 1 if self.inference else self.config.get("MAXEPOCH", 100)
        return ray.tune.RunConfig(
            name=self.experiment_name,
            storage_path=self.checkpoint_folder,
            stop={
                "step": max_epochs
            },
        )

    def _optimize(self, resources: dict, tuner_instance: callable, param_space:callable):
        
        def make_trainable(base_config, flags, objectives, trainer):
            class CustomTrainable(tuner_instance):
                def setup(self, config):
                    super().setup(config, base_config, flags, objectives, trainer)
                
                def save_checkpoint(self, checkpoint_dir): 
                    pass
            return CustomTrainable


        if os.path.exists(os.path.join(self.checkpoint_folder, self.experiment_name, "tuner.pkl")):
            return ray.tune.Tuner.restore(
                os.path.join(self.checkpoint_folder, self.experiment_name),
                make_trainable(self.config, self.flags, self.objectives, self.trainer)
            )
        else: 
            return ray.tune.Tuner(
            ray.tune.with_resources(make_trainable(self.config, self.flags, self.objectives, self.trainer), resources=resources),
            tune_config=ray.tune.TuneConfig(
                metric="FDP",
                mode="min",
                num_samples=self.flags.n_trials, 
            ),
            run_config=self._run_config(),
            param_space=param_space()
        )

    def inference_optimize(self, resources:dict):
        return self._optimize(resources, InferenceOptimize, self.make_param_space_inference)

    def train_optimize(self, resources:dict):
        return self._optimize(resources, TrainOptimize, self.make_param_space_training)

    def __call__(self) -> None:

        ray.init()
        try: 
            n_accessable_gpu = len(os.environ['SLURM_JOB_GPUS'].split(','))
            cpu_ratio = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])/n_accessable_gpu
            n_gpu = 1
        except KeyError: 
            n_gpu = 0 
            cpu_ratio = 1

        if cpu_ratio >= 64:  # Estimated number of max threads 
            n_cpu_per_job =  os.environ['NUMEXPR_MAX_THREADS'] - 2
        else: 
            n_cpu_per_job = int(cpu_ratio)
            
        resources = {"cpu": n_cpu_per_job, "gpu": n_gpu}
        if self.inference: 
            print("Running inference job...")
            tuner = self.inference_optimize(resources=resources)
        else: 
            tuner = self.train_optimize(resources=resources)

        tuner.fit()
        
        final_df = tuner.get_results().get_dataframe().to_json()
        fp = os.path.join(self.flags.results_folder, self.flags.study_name, "results.json")
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, 'w') as f: 
            json.dump(final_df, f)
        
