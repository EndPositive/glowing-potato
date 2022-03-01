from multiprocessing import Pool
from split_image import split_image
import os
from tqdm import tqdm
from watermarker import Watermarker

DATA_DIR = 'resources/edge'
OUTPUT_DIR = 'resources/dataset'


def process_image(file_path):
    # todo: add a single function that does all this
    watermarker = Watermarker(file_path)
    watermarker.add_text()
    watermarker.add_simple_grid()

    inputs = split_image(watermarker.image)
    outputs = split_image(file_path)

    # get new file names for each chunk
    file_names = [
        os.path.basename(file_path)[:-4] + '_' + str(i) + '.jpg' for i in range(len(inputs))
    ]

    # save files
    for fname, input, output in zip(file_names, inputs, outputs):
        input.save(os.path.join(OUTPUT_DIR, 'input', fname))
        output.save(os.path.join(OUTPUT_DIR, 'output', fname))


def process(threads=16):
    # create output if it doesn't exist
    os.makedirs(os.path.join(OUTPUT_DIR, 'input'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'output'), exist_ok=True)

    # get name of all files
    all_files = list(map(lambda x: os.path.join(DATA_DIR, x), os.listdir(DATA_DIR)))

    with Pool(threads) as p:
        with tqdm(total=len(all_files)) as pbar:
            for _ in p.imap_unordered(process_image, all_files):
                pbar.update()
        p.close()
        p.join()


if __name__ == '__main__':
    process()
