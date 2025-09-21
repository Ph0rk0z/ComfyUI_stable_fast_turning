import torch
from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig

from sfast.compilers.stable_diffusion_pipeline_compiler import compile_vae

from .module.sfast_pipeline_compiler import build_lazy_trace_module

import copy

from comfy import model_management as mm

def _cast_tree(obj, device, dtype):
    # Tensor: cast with Comfy's helper (ints keep dtype)
    if torch.is_tensor(obj):
        want_dtype = None if obj.dtype in (torch.int, torch.long) else dtype
        return mm.cast_to_device(obj, device, want_dtype)
    # Dict-like
    if isinstance(obj, dict):
        return {k: _cast_tree(v, device, dtype) for k, v in obj.items()}
    # Sequence: list or tuple
    if isinstance(obj, (list, tuple)):
        seq = [_cast_tree(v, device, dtype) for v in obj]
        return tuple(seq) if isinstance(obj, tuple) else seq
    # Anything else: leave as-is
    return obj

def is_cuda_malloc_async():
    return "cudaMallocAsync" in torch.cuda.get_allocator_backend()


def gen_stable_fast_config():
    config = CompilationConfig.Default()
    # xformers and triton are suggested for achieving best performance.
    # It might be slow for triton to generate, compile and fine-tune kernels.
    try:
        import xformers

        config.enable_xformers = True
    except ImportError:
        print("xformers not installed, skip")
    try:
        import triton

        config.enable_triton = True
    except ImportError:
        print("triton not installed, skip")

    if config.enable_triton and is_cuda_malloc_async():
        print("disable stable fast triton because of cudaMallocAsync")
        config.enable_triton = False

    # CUDA Graph is suggested for small batch sizes.
    # After capturing, the model only accepts one fixed image size.
    # If you want the model to be dynamic, don't enable it.
    config.enable_cuda_graph = True
    #config.enable_fused_linear_geglu = False
    # config.enable_jit_freeze = False
    #config.trace_scheduler=True #Slower
    #print("\n--- Final Configuration ---")
    #print(config)

    return config


class StableFastPatch:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.stable_fast_model = None

    def __deepcopy__(self, memo=None):
        return self

    def release_original_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            # If you're on CUDA, you might want to empty the cache too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("Original model has been released from memory.")

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        # disable with accelerate for now
        if hasattr(model_function.__self__, "hf_device_map"):
            return model_function(input_x, timestep_, **c)

        if self.stable_fast_model is None:
            self.stable_fast_model = build_lazy_trace_module(
                self.config,
                input_x.device,
                id(self),
            )
        # Pre-cast to avoid device moves during CUDA Graph capture
        device, dtype = input_x.device, input_x.dtype
        c = _cast_tree(c, device, dtype)

        return self.stable_fast_model(
            model_function, input_x=input_x, timestep=timestep_, **c
        )

    def to(self, device):
       if type(device) == torch.device:
            if self.config.enable_cuda_graph or self.config.enable_jit_freeze:
                if device.type == "cpu":
                    self.to("cuda") #Don't let it!
                    #if self.model is not None:
                    #    del self.model
                    #    self.model = None
                    #if torch.cuda.is_available(): #GC
                    #    torch.cuda.empty_cache()
                    # comfyui tell we should move to cpu. but we cannt do it with cuda graph and freeze now.
#                    del self.stable_fast_model
#                    self.stable_fast_model = None
#                    print(
#                        "\33[93mWarning: Your graphics card doesn't have enough video memory to keep the model. If you experience a noticeable delay every time you start sampling, please consider disable enable_cuda_graph.\33[0m"
#                    )
            else:
                if self.stable_fast_model != None and device.type == "cpu":
                    self.stable_fast_model.to_empty()
       return self


class ApplyStableFastUnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_cuda_graph": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_stable_fast"

    CATEGORY = "loaders"

    def apply_stable_fast(self, model, enable_cuda_graph):
        config = gen_stable_fast_config()

        if not enable_cuda_graph:
            config.enable_cuda_graph = False
            config.enable_jit_freeze = False

        if config.memory_format is not None:
            model.model.to(memory_format=config.memory_format)

        patch = StableFastPatch(model, config)
        model_stable_fast = model.clone()
        model_stable_fast.set_model_unet_function_wrapper(patch)
        # Free old weights
        patch.release_original_model()
        return (model_stable_fast,)



class ApplyStableFastVae:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "apply_stable_fast_vae"

    CATEGORY = "loaders"

    def apply_stable_fast_vae(self, vae):
        vae_clone = copy.copy(
            vae
        )  # Use deepcopy to avoid modifying the original VAE

        #print(type(vae))
        #print(dir(vae))
        #print(isinstance(vae, torch.nn.Module))
        #print(vae.__dict__)

        config = gen_stable_fast_config() #grab the config used by unet, same way
        #config.memory_format=None
        #config.enable_jit=True
        config.enable_cuda_graph = True

        vae_clone.first_stage_model = compile_vae(vae_clone.first_stage_model, config)
        #vae_clone = compile_vae(vae_clone, config) #compile the clone

        print(
            "Stable fast VAE mode enabled. This will persist until you restart your runtime."
        )

        return (vae_clone,) #return the clone
