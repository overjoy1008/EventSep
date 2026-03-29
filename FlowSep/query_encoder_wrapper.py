# FlowSep/query_encoder_wrapper.py

class FlowSepQueryEncoder:
    """
    FlowSep는 generate_sample() 내부에서 텍스트를 처리하므로,
    evaluator 호환을 위해 껍데기만 제공하는 dummy encoder.
    """

    def get_query_embed(self, *args, **kwargs):
        return None
