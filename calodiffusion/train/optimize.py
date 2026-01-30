from abc import ABC, abstractmethod
import traceback
from typing import Any, Iterable, Literal, Sequence

import ray.tune
import numpy as np
import torch
import json
import os

from datetime import datetime
from calodiffusion.utils import utils
from calodiffusion.train import evaluate

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
            hgcal=utils.LoadJson(os.environ.get("CONFIG", {})).get("HGCAL", False))
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
        self.config = base_config
        self.n_steps = config.get("NSTEPS", 50)
        self.eval_data, _ = utils.load_data(flags, self.config, eval=True)
        self.model_instance = trainer(flags=flags, config=self.config, save_model=False, load_data=False)
        self.model_instance.init_model()

        self.objectives = [EvalFPD(), EvalCount()]
        self.objective_names = ['FDP', 'COUNT']

        self.current_step = 0


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
            model, _, _, _, _, _  = self.model_instance.pickup_checkpoint(
                model=self.model_instance.model,
                optimizer=None,
                scheduler=None,
                early_stopper=None,
                n_epochs=0,
                restart_training=False,
            )

            generated_samples, generated_energies = model.generate(
                data_loader=self.eval_data, sample_steps=self.n_steps, debug=False, sample_offset=0,
            )
            objectives = self.evaluate(
                model, generated_samples, generated_energies
            )
        except RuntimeError as err:
            print(f"Error in loading checkpoint: {err}")
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

        if "BASE_CHANNELS" in config.keys():
            depth = config.get('UNET_DEPTH', 4)
            base_channels = config['BASE_CHANNELS']
            channel_mult = config.get("CHANNEL_MULT", 8)
        
            base_size = int(base_channels/channel_mult)*channel_mult
            layer_sizes = [base_size] + [base_size + base_size*i for i in range(depth)]

            self.config["LAYER_SIZE_UNET"] = layer_sizes
            self.config['COND_SIZE_UNET'] = base_channels*(depth+1)
            self.config['BLOCK_GROUPS'] = channel_mult

        self.objectives = [EvalFPD(), EvalCount()]
        self.objective_names = ['FDP', 'COUNT']

        self.current_step = 0

        self.eval_data, _ = utils.load_data(flags, self.config, eval=True)
        self.objective_names = ["FDP", "LOSS"]
        self.objectives = [EvalFPD(), EvalLoss()]

        self.flags = flags
        self.trainer_module = trainer


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
            trainer_instance = self.trainer_module(flags=self.flags, config=self.config, save_model=False, load_data=True)
            trainer_instance.init_model()
            trainer_instance.train()
            model = trainer_instance.model

        except RuntimeError as err:
            print(f"Error in training model: {err}")
            print(traceback.print_exception(err))
            model = None
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
            "BASE_CHANNELS":  Size of the first block of the unet
            "CHANNEL_MULT": How quickly blocks layers scale 
            "UNET_DEPTH": The number of total layers blocks
        
        }
        Such that: 
            "BLOCK_GROUPS" = "CHANNEL_MULT" (insure the geometric works out)
            "LAYER_SIZE_UNET" = [int(BASE_CHANNELS/CHANNEL_MULT)*CHANNEL_MULT * i for i range(DEPTH)]

    * COND_SIZE_UNET (u-net archective setting) - Dependent on unet settings such that: 
        "COND_SIZE_UNET":  int(BASE_CHANNELS/CHANNEL_MULT)*CHANNEL_MULT * (depth+1)
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

        self.experiment_name = f"{self.flags.study_name}_{'inference' if inference else 'train'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

        sampler_options = self.config.get("OPTIMIZE", {}).get("SAMPLER", [])
        if sampler_options != []: 
            config_space["SAMPLER"] = ray.tune.choice(sampler_options)

        noise_schedules = self.config.get("OPTIMIZE", {}).get("NOISE_SCHED", ["log", "linear"])
        config_space["NOISE_SCHED"] = ray.tune.choice(noise_schedules)
        
        steps = self.config.get("OPTIMIZE", {}).get("NSTEPS", [50, 500])
        config_space["NSTEPS"] = ray.tune.randint(*steps)

        time_embeds = self.config.get("OPTIMIZE", {}).get("TIME_EMBED", ["sigma", "log", "sin", "id"])
        config_space["TIME_EMBED"] = ray.tune.choice(time_embeds)

        # Go through the rest of the sampler settings
        sampler_settings = self.config.get("OPTIMIZE", {}).get("SAMPLER_SETTINGS", {})
        config_space.update(self._generic_param_space(sampler_settings))

        return config_space
    
    def make_param_space_training(self):
        config_space = {}
        # Generic hyperparameters
        optimized_section = self.config.get("OPTIMIZE", {})
        config_space.update(self._generic_param_space(optimized_section))

        # Sub parameters for UNet architecture
        if "LAYER_SIZE_UNET" in optimized_section.keys(): 

            config_space['BASE_CHANNELS']  = ray.tune.qrandint(*optimized_section["LAYER_SIZE_UNET"]["BASE_CHANNELS"], 8)
            config_space['UNET_DEPTH'] = ray.tune.randint(*optimized_section["LAYER_SIZE_UNET"]["UNET_DEPTH"])
            config_space['CHANNEL_MULT'] = ray.tune.uniform(*optimized_section["LAYER_SIZE_UNET"]["CHANNEL_MULT"])
            
        return config_space
    
    def _run_config(self): 
        return ray.tune.RunConfig(
            name=self.experiment_name,
            storage_path=self.checkpoint_folder,
            stop={
                "step": self.config.get("MAXEPOCH", 100),
            },
            checkpoint_config=ray.tune.CheckpointConfig(
                checkpoint_frequency=5, checkpoint_at_end=True, 
            ),
        )

    def _optimize(self, resources: dict, tuner_instance: callable, param_space:callable):

        def make_trainable(base_config, flags, objectives, trainer):
            class CustomTrainable(tuner_instance):
                def setup(self, config):
                    super().setup(config, base_config, flags, objectives, trainer)
                
                def save_checkpoint(self, checkpoint_dir): 
                    pass
            return CustomTrainable


        if os.path.exists(os.path.join(self.checkpoint_folder, "tuner.pkl")):
            return ray.tune.Tuner.restore(
                self.checkpoint_folder,
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
            tuner = self.inference_optimize(resources=resources)
        else: 
            tuner = self.train_optimize(resources=resources)

        tuner.fit()
        
        final_df = tuner.get_results().get_dataframe().to_json()
        fp = os.path.join(self.flags.results_folder, self.flags.study_name, "results.json")
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, 'w') as f: 
            json.dump(final_df, f)
        
