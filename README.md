# Investigating Compositional Challenges in Vision-Language Models for Visual Grounding - CVPR 24

### Get Started
- datasets format are followed by [link](https://github.com/hassanhub/MultiGrounding/tree/master/data)

##### Evaluation on standard visual grounding datasets
CLIP
```bash
python inference_grounding.py -dataset refit -val_path /path_to/RefIt -Isize 224 -clip_eval -nW 1
python inference_grounding.py -dataset flicker -val_path /path_to/flicker -Isize 224 -clip_eval -nW 1
python inference_grounding.py -dataset vg -val_path /path_to/VG -Isize 224 -clip_eval -nW 1
```

ALBEF
```bash
python inference_grounding.py -dataset refit -val_path /path_to/RefIt -Isize 384 -albef_eval -nW 1
python inference_grounding.py -dataset flicker -val_path /path_to/flicker -Isize 384 -albef_eval -nW 1
python inference_grounding.py -dataset vg -val_path /path_to/VG -Isize 384 -albef_eval -nW 1
```

METER
```bash
# please download the METER checkpoint first
python inference_grounding.py -dataset refit -val_path /path_to/RefIt -Isize 384 -meter_eval -nW 1
python inference_grounding.py -dataset flicker -val_path /path_to/flicker -Isize 384 -meter_eval -nW 1
python inference_grounding.py -dataset vg -val_path /path_to/VG -Isize 384 -meter_eval -nW 1
```

BLIP2
```bash
python inference_grounding.py -dataset refit -val_path /path_to/RefIt -Isize 384 -albef_eval -nW 1
python inference_grounding.py -dataset flicker -val_path /path_to/flicker -Isize 384 -albef_eval -nW 1
python inference_grounding.py -dataset vg -val_path /path_to/VG -Isize 384 -albef_eval -nW 1
```

##### Evaluation on ARPGrounding
CLIP
```bash
python inference_arpgrounding.py  -val_path /path_to/VG -clip_eval -Isize 224 -split attribute
python inference_arpgrounding.py  -val_path /path_to/VG -clip_eval -Isize 224 -split relation
python inference_arpgrounding.py  -val_path /path_to/VG -clip_eval -Isize 224 -split priority
```

ALBEF
```bash
python inference_arpgrounding.py  -val_path /path_to/VG -albef_eval -Isize 384 -split attribute
python inference_arpgrounding.py  -val_path /path_to/VG -albef_eval -Isize 384 -split relation
python inference_arpgrounding.py  -val_path /path_to/VG -albef_eval -Isize 384 -split priority
```

METER
```bash
# please download the METER checkpoint first
python inference_arpgrounding.py  -val_path /path_to/VG -meter_eval -Isize 384 -split attribute
python inference_arpgrounding.py  -val_path /path_to/VG -meter_eval -Isize 384 -split relation
python inference_arpgrounding.py  -val_path /path_to/VG -meter_eval -Isize 384 -split priority
```

BLIP2
```bash
python inference_arpgrounding.py  -val_path /path_to/VG -blip2_eval -Isize 364 -split attribute
python inference_arpgrounding.py  -val_path /path_to/VG -blip2_eval -Isize 364 -split relation
python inference_arpgrounding.py  -val_path /path_to/VG -blip2_eval -Isize 364 -split priority
```
