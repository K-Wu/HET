#!/usr/bin/env python3


class DummyCtx:
    def __init__(self):
        self.saved_tensors = []

    def save_for_backward(self, *args):
        self.saved_tensors = args
