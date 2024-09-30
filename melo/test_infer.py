import os
from api import TTS

ckpt_path = 'logs/mix_en/G_10000.pth'
text = """With fusion energy at its core, Virginia would become the epicenter of the next tech revolution. Companies would flood the state, drawn by the unlimited, cheap power. Research labs would push the boundaries of AI, robotics, and space exploration, all powered by this clean energy. Job creation would skyrocket, attracting talent from across the country. But could this explosion in growth come at a price? Stay tuned to see the unexpected consequences!"""
language = 'EN'
output_dir = 'output'
speaker = 'en_001'

def main(ckpt_path, text, language, output_dir):
    if ckpt_path is None:
        raise ValueError("The model_path must be specified")
    
    config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
    model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)
    
    for spk_name, spk_id in model.hps.data.spk2id.items():
        print('spk_name:', spk_name, 'spk_id:', spk_id)
        if spk_name == speaker:
            save_path = f'{output_dir}/{spk_name}/output.wav'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.tts_to_file(text, spk_id, save_path)

if __name__ == "__main__":
    main(ckpt_path, text, language, output_dir)
