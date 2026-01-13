from .ar_decoder import LLM2TTSCodecAR

def build_ar_ori_v2_new_speech_generator(config):
    generator_type = getattr(config, 'speech_generator_type', 'ar_ori_v2_new')
    if generator_type == 'ar_ori_v2_new':
        return LLM2TTSCodecAR(config)

    raise ValueError(f'Unknown generator type: {generator_type}')
