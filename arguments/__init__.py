

import os
import sys
from argparse import ArgumentParser, Namespace


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.img_skip_step = 1  # training image skip step
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")


class OptimizationParamsBlock(ParamGroup):
    def __init__(self, parser):
       
        self.ratio_block_scene = 0.25  # initial block ratio
        self.num_splats = [100]*8  # initial number of splats, each block has 100 points
        self.scale_block_min = 0.2  # block min scale

        self.iterations = 30000
        self.alpha_lr = 0.0005
        self.feature_lr = 0.0025
        self.opacity_lr = 0.01
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.mesh_until_iter = 15_000
        self.coarse_learning_until_iter = 25000
        self.add_primitive_until_iter = 10000
        self.primitive_coverage_until_iter = 3000
        self.add_primitive_interval = 5000
        self.mesh_from_iter = 500
        self.mesh_remove_interval = 1000
        self.reset_params = 3000
        self.random_background = False
        self.use_mesh = True
        self.use_ray_constrain = True
        self.lambda_dssim = 0.2
        self.lambda_dist = 100.0
        self.lambda_normal = 0.05
        self.lambda_mask_entropy = 0.1
        self.rgb_weight = 1.
        self.parsimony_weight = 0.002 # 0.1 
        self.overlap_weight = 2.
        self.occupancy_weight = 0.02
        self.ray_coverage_weigth = 10.
        self.ray_uncoverage_weigth = 100.
        self.primtive_scale_weight = 0.1
        self.guassian_scale_weight = 1.
        self.percent_dense = 0.01
        super().__init__(parser, "Optimization Parameters")

class OptimizationParamsPart(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01

        self.opacity_cull = 0.05
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")



def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
