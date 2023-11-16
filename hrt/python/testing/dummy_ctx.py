#!/usr/bin/env python3


class AlwaysTrueList:
    def __init__(self):
        pass

    def __len__(self):
        return -1

    def __getitem__(self, idx):
        return True


class DummyCtx:
    def __init__(self):
        self.saved_tensors = []
        self.needs_input_grad = AlwaysTrueList()

    def save_for_backward(self, *args):
        self.saved_tensors = args
