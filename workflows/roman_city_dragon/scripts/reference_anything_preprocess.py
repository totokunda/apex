import sys
sys.path.append('/workspace/apex')

from src.preprocess.composition import ReferenceAnythingPreprocessor

preprocessor = ReferenceAnythingPreprocessor()

out = preprocessor(
    images=['assets/ominous_mountain.png'],
    mode='label',
    return_mask=True,
    label='mountain',
)

for idx, image in enumerate(out.images):
    image.save(f'assets/reference_anything/ominous_mountain_reference_anything_label.png')