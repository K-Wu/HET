import json


def is_json_nvtx_event(json_data):
    # NB: NVTX event types are defined in https://docs.nvidia.com/nsight-systems/UserGuide/index.html
    # """
    # NVTX Event Type Values
    # 33 - NvtxCategory
    # 34 - NvtxMark
    # 39 - NvtxThread
    # 59 - NvtxPushPopRange
    # 60 - NvtxStartEndRange
    # 75 - NvtxDomainCreate
    # 76 - NvtxDomainDestroy
    # """
    result = "NvtxEvent" in line
    if result:
        if json_data["Type"] != 59:
            print("Unexpected NVTX event type: {}".format(json_data["Type"]))
            result = False
    return result


class HectorNVTXEvents:
    def __init__(self, json_data):
        self.json_data = json_data
        self.start_timestamp = int(json_data["NvtxEvent"]["Timestamp"])
        self.end_timestamp = int(json_data["NvtxEvent"]["EndTimestamp"])
        self.text = json_data["NvtxEvent"]["Text"]
        # extract seq and op_id if there are any
        self.seq = -1
        self.op_id = -1
        self.hector_op_category = ""
        text_parts = self.text.split(",")
        for text_part in text_parts:
            if "seq" in text_part:
                self.seq = int(text_part.split(" = ")[1])
            if "op_id" in text_part:
                self.op_id = int(text_part.split(" = ")[1])
            if "hector_op_category" in text_part:
                self.hector_op_category = text_part.split(" = ")[1].strip()
        self.children = []


if __name__ == "__main__":
    nvtx_events = []
    with open("hgt_acm") as fd:
        result = 0
        for line in fd:
            if "NvtxEvent" in line:
                nvtx_events.append(json.loads(line))
                result += is_json_nvtx_event(nvtx_events[-1])
        print(result)

# A multi-level NVTX range looks like this:
# {"Type":59,"NvtxEvent":{"Type":59,"Timestamp":"4739751477","Text":"autograd::engine::evaluate_function: AddmmBackward0, op_id = 1718","GlobalTid":"281863302175408","EndTimestamp":"4740091747","DomainId":"0","NsTime":true}}
# {"Type":59,"NvtxEvent":{"Type":59,"Timestamp":"4739753014","Text":"AddmmBackward0, seq = 596, op_id = 1719","GlobalTid":"281863302175408","EndTimestamp":"4740063789","DomainId":"0","NsTime":true}}
# {"Type":59,"NvtxEvent":{"Type":59,"Timestamp":"4739759015","Text":"aten::t, op_id = 1720","GlobalTid":"281863302175408","EndTimestamp":"4739762864","DomainId":"0","NsTime":true}}
# {"Type":59,"NvtxEvent":{"Type":59,"Timestamp":"4739760324","Text":"aten::transpose, op_id = 1721","GlobalTid":"281863302175408","EndTimestamp":"4739762002","DomainId":"0","NsTime":true}}
# {"Type":59,"NvtxEvent":{"Type":59,"Timestamp":"4739761055","Text":"aten::as_strided, op_id = 1722","GlobalTid":"281863302175408","EndTimestamp":"4739761828","DomainId":"0","NsTime":true}}
# {"Type":59,"NvtxEvent":{"Type":59,"Timestamp":"4739764299","Text":"aten::mm, op_id = 1723","GlobalTid":"281863302175408","EndTimestamp":"4739988492","DomainId":"0","NsTime":true}}
