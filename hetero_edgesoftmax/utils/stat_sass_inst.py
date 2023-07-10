# TODO: follow how https://github.com/cloudcores/CuAssembler/blob/master/bin/dsass.py uses CuInsFeeder that iterates the .sass file and figure out the instruction in each line. The regex patterns from that class are especially useful.
# TODO: dyn_inst count could use cu_sass_stats inhttps://github.com/VerticalResearchGroup/casio/blob/main/scripts/utils/ncu_sass.py#L44
# TODO: figure out global/shared/local memory acccesses
import subprocess


def demangle_cuda_function_name(func_name: str) -> str:
    """Demangle CUDA function name."""
    return subprocess.check_output(["cu++filt", func_name]).decode("utf-8").strip()


if __name__ == "__main__":
    print(
        demangle_cuda_function_name(
            "_ZN3cub18DeviceReduceKernelINS_18Device"
            "ReducePolicyIN6thrust5tupleIblNS2_9null"
            "_typeES4_S4_S4_S4_S4_S4_S4_EElNS2_8cuda"
            "_cub9__find_if7functorIS5"
            "_EEE9Policy600ENS2_12zip_iteratorINS3"
            "_INS6_26transform_input_iterator_tIbNS6"
            "_35transform_pair_of_input_iterators"
            "_tIbNS2_6detail15normal_iteratorINS2"
            "_10device_ptrIKfEEEESK_NS2_8equal"
            "_toIfEEEENSF_12unary_negateINS6"
            "_8identityEEEEENS6_19counting_iterator"
            "_tIlEES4_S4_S4_S4_S4_S4_S4_S4_EEEElS9"
            "_S5_EEvT0_PT3_T1_NS_13GridEvenShareISZ"
            "_EET2_"
        )
    )
