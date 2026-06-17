import time
from cp_measure import examples
from cp_measure.core import (
    measureobjectintensity,
    measureobjectsizeshape,
    measuretexture,
    measuregranularity,
)


def benchmark_intensity(pixels, masks):
    print("Benchmarking Intensity...")
    start_time = time.time()
    measureobjectintensity.get_intensity(masks, pixels)
    end_time = time.time()
    print(f"Intensity time: {end_time - start_time:.4f} seconds")


def benchmark_sizeshape(pixels, masks):
    print("Benchmarking SizeShape...")
    start_time = time.time()
    measureobjectsizeshape.get_sizeshape(masks, pixels)
    end_time = time.time()
    print(f"SizeShape time: {end_time - start_time:.4f} seconds")


def benchmark_zernike(pixels, masks):
    print("Benchmarking Zernike...")
    start_time = time.time()
    measureobjectsizeshape.get_zernike(masks, pixels)
    end_time = time.time()
    print(f"Zernike time: {end_time - start_time:.4f} seconds")


def benchmark_feret(pixels, masks):
    print("Benchmarking Feret...")
    start_time = time.time()
    measureobjectsizeshape.get_feret(masks, pixels)
    end_time = time.time()
    print(f"Feret time: {end_time - start_time:.4f} seconds")


def benchmark_texture(pixels, masks):
    print("Benchmarking Texture...")
    start_time = time.time()
    measuretexture.get_texture(masks, pixels)
    end_time = time.time()
    print(f"Texture time: {end_time - start_time:.4f} seconds")


def benchmark_granularity(pixels, masks):
    print("Benchmarking Granularity...")
    start_time = time.time()
    measuregranularity.get_granularity(masks, pixels)
    end_time = time.time()
    print(f"Granularity time: {end_time - start_time:.4f} seconds")


def run_benchmarks():
    # Setup data
    size = 500
    print(f"Generating data (size={size})...")
    pixels = examples.get_pixels(size=size)
    masks_dict = examples.get_masks(size=size)

    # Use the 'full' mask for maximum coverage, or 'full_2' for multiple objects if needed.
    # 'full' is a single large object. 'full_2' has 2 large objects.
    # Let's use 'full_2' to have at least 2 objects to measure.
    masks = masks_dict["full_2"]

    # Ensure pixels and masks are compatible types/shapes if needed
    # examples.get_pixels returns random integers 0-10.

    benchmark_intensity(pixels, masks)
    benchmark_sizeshape(pixels, masks)
    benchmark_zernike(pixels, masks)
    benchmark_feret(pixels, masks)
    benchmark_texture(pixels, masks)
    benchmark_granularity(pixels, masks)


if __name__ == "__main__":
    run_benchmarks()
