import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Diffu', help='Model for plotting')
parser.add_argument('-d', '--model_dir', default='../models/TEST', help='Directory containing saved model')
parser.add_argument('-n', '--name', default='test', help='job name')
parser.add_argument('-v', '--model_version', default='checkpoint.pth', help='Which model to plot (best_val.pth, checkpoint.pth, final.pth)')
parser.add_argument('--sample_algo', default='ddpm', help='Sampling algo')
parser.add_argument('--sample_offset', default=0, type = int, help='Offset for sampling')
parser.add_argument('--sample_steps', default=-1, type = int, help='Number of sampling steps')
parser.add_argument('--nevts', default=1000, type = int, help='Offset for sampling')
parser.add_argument('--eval', default=False, action = 'store_true', help='Run CaloChallenge eval metrics')
parser.add_argument("--constraint", default = "a100|v100|p100" , help='gpu resources')
parser.add_argument("--memory", default = 16000 , help='RAM')
parser.add_argument("--num_jobs", type = int, default = 1 , help='How many jobs to split among')
parser.add_argument("--batch_size", type = int, default = 100 , help='Batch size for sampling')
parser.add_argument("--chip_type",  default = 'gpu' , help='gpu or cpu')
parser.add_argument("--layer_model", default = False, action = 'store_true', help='Sep model for layer energies')
parser.add_argument("--extra_args", default = "" , help='RAM')
flags = parser.parse_args()


if(flags.num_jobs == 1):
    job_idxs = [-1]
else:
    job_idxs = list(range(flags.num_jobs))
    flags.nevts /= flags.num_jobs


base_dir = r"\/work1\/cms_mlsim\/oamram\/CaloDiffusion\/models\/"
if(flags.model_dir[-1] != "/"): flags.model_dir += "/"
if(flags.name[-1] == "/"): flags.name = flags.name[:-1]
model_dir_tail = flags.model_dir.split("/")[-2]
if(flags.model == 'Diffu'):
    if(not os.path.exists(flags.name)): os.system("mkdir %s" % flags.name)

    for job_idx in job_idxs:
        if(job_idx < 0): 
            script_loc = flags.name + "/plot.sh"
        else:
            script_loc = flags.name + "/plot_j%i.sh" % job_idx

        os.system("cp plot.sh %s" % (script_loc))

        #some options only for gpu
        if(flags.chip_type == 'gpu'):
            os.system("sed -i 's/DOGPU/SBATCH/g' %s" % (script_loc))

        if(flags.layer_model):
            os.system("sed -i 's/LAYMODEL/%s/g' %s" % (base_dir + model_dir_tail +"\/" + "layer_checkpoint.pth", script_loc) )
        else:
            os.system("""sed -i "s/LAYMODEL/dummy/g" %s""" % (script_loc) )

        os.system("sed -i 's/JOB_NAME/%s/g' %s" % (flags.name, script_loc))
        os.system("sed -i 's/JOB_OUT/%s/g' %s" % (flags.name, script_loc))
        os.system("sed -i 's/MODEL/%s/g' %s" % (flags.model, script_loc) )
        os.system("sed -i 's/MDIR/%s/g' %s" % (base_dir +  model_dir_tail, script_loc) )
        os.system("sed -i 's/MNAME/%s/g' %s" % (flags.model_version, script_loc) )


        os.system("sed -i 's/SAMPLE_ALGO/%s/g' %s" % (flags.sample_algo, script_loc) )
        os.system("sed -i 's/SAMPLE_OFFSET/%s/g' %s" % (flags.sample_offset, script_loc) )
        os.system("sed -i 's/SAMPLE_STEPS/%s/g' %s" % (flags.sample_steps, script_loc) )
        os.system("sed -i 's/NEVTS/%s/g' %s" % (str(flags.nevts), script_loc) )
        os.system("sed -i 's/BATCH_SIZE/%s/g' %s" % (str(flags.batch_size), script_loc) )

        os.system("sed -i 's/JOBIDX/%s/g' %s" % (str(job_idx), script_loc) )

        os.system("sed -i 's/EVAL_VAR/%s/g' %s" % (str(flags.eval).lower(), script_loc) )
        os.system("sed -i 's/MTAG/%s/g' %s" % (model_dir_tail, script_loc) )

        os.system("sed -i 's/TYPE/%s/' %s" % (flags.chip_type, script_loc))
        os.system("sed -i 's/CONSTRAINT/%s/' %s" % (flags.constraint, script_loc))
        os.system("sed -i 's/MEMORY/%s/' %s" % (flags.memory, script_loc))
        os.system("sed -i 's/EXTRAARGS/%s/' %s" % (flags.extra_args, script_loc) )

        #submit
        os.system("sbatch %s" % script_loc)

