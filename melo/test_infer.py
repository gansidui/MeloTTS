import os
from api import TTS
import time

ckpt_path = 'models/mix_en/G_14.pth'
text = """With fusion energy at its core, Virginia would become the epicenter of the next tech revolution. Companies would flood the state, drawn by the unlimited, cheap power. Research labs would push the boundaries of AI, robotics, and space exploration, all powered by this clean energy. Job creation would skyrocket, attracting talent from across the country. But could this explosion in growth come at a price? Stay tuned to see the unexpected consequences!"""
language = 'EN'
output_dir = 'output'
speaker = 'en_148'

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

            start_time = time.time()
            model.tts_to_file(text, spk_id, save_path)
            end_time = time.time()
            used_time = (end_time - start_time) * 1000
            print(f"tts_to_file used_time: {used_time:.2f} ms")

if __name__ == "__main__":
    main(ckpt_path, text, language, output_dir)
