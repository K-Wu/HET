import json

gemm_1_schedule = {
    "C": '(EDGEWISE, "zi")',
    "A": '(SRCNODE, "feature")',
    "B": "(W, EDGETYPE)",
    "schedule": {
        "param": {"tile_sz": 16},
        "A": "[GATHER (row_idx), NO_TRANSPOSE]",
        "B": "[WEIGHTS]",
        "C": "[NO_SCATTER]",
    },
}

if __name__ == "__main__":
    print(gemm_1_schedule)
    print(json.dumps(gemm_1_schedule))
