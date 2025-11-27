#!/usr/bin/env python3
# robust_stego_with_metadata_phash_adaptive_color.py
import os
import math
import zlib
import cv2
import numpy as np
from scipy.fftpack import dct, idct
from reedsolo import RSCodec
from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt

# ---------------- 默认参数 ----------------
BLOCK_SIZE = 8
BITS_PER_POSITION_BLOCK = 9
DEFAULT_REDUNDANCY = 4
DEFAULT_ECC_NSYM = 20
PHASH_BITS = 64
METADATA_RESERVED_MIN = 16

# ---------- 工具函数 ----------
def bytes_to_bits(b: bytes) -> str:
    return ''.join(bin(x)[2:].zfill(8) for x in b)

def bits_to_bytes(bits: str) -> bytes:
    if len(bits) % 8 != 0:
        bits = bits.ljust(len(bits) + (8 - len(bits)%8), '0')
    return bytes(int(bits[i:i+8],2) for i in range(0,len(bits),8))

def text_to_bytes(s: str) -> bytes:
    return s.encode('utf-8')

def bytes_to_text(b: bytes) -> str:
    try:
        return b.decode('utf-8', errors='replace')
    except:
        return b.decode('latin1', errors='replace')

# ---------- pHash ----------
def phash_bits_from_block_bits(bits: str) -> str:
    L = 32
    arr = np.zeros((L, L), dtype=float)
    flat = [int(b) for b in bits.ljust(L*L, '0')[:L*L]]
    arr[:] = np.array(flat).reshape((L, L))
    d = dct(dct(arr.T, norm='ortho').T, norm='ortho')
    sub = d[1:9,1:9].flatten()
    med = np.median(sub)
    return ''.join('1' if v > med else '0' for v in sub)

def phash_bits_from_block(block: np.ndarray) -> str:
    small = cv2.resize(block, (32,32), interpolation=cv2.INTER_AREA).astype(float)
    d = dct(dct(small.T, norm='ortho').T, norm='ortho')
    sub = d[1:9,1:9].flatten()
    med = np.median(sub)
    return ''.join('1' if v > med else '0' for v in sub)

def phash_hamming_sim(b1: str, b2: str) -> float:
    assert len(b1) == len(b2)
    dist = sum(a!=b for a,b in zip(b1,b2))
    return 1.0 - dist / len(b1)

# ---------- DCT 嵌入/提取 ----------
def embed_bits_in_block(block: np.ndarray, bits: str) -> np.ndarray:
    coeff = dct(dct(block.T, norm='ortho').T, norm='ortho')
    idx = 0
    for i in range(1,4):
        for j in range(1,4):
            if idx >= len(bits):
                break
            coeff_ij = int(round(coeff[i,j]))
            coeff[i,j] = (coeff_ij & ~1) | int(bits[idx])
            idx += 1
    rec = idct(idct(coeff.T, norm='ortho').T, norm='ortho')
    return rec

def extract_bits_from_block(block: np.ndarray, length: int) -> str:
    coeff = dct(dct(block.T, norm='ortho').T, norm='ortho')
    bits = ''
    for i in range(1,4):
        for j in range(1,4):
            if len(bits) >= length:
                break
            bits += str(int(round(coeff[i,j])) & 1)
    return bits

# ---------- ECC ----------
def ecc_encode_bytes(data: bytes, ecc_nsym: int) -> bytes:
    rs = RSCodec(ecc_nsym)
    return rs.encode(data)

def ecc_decode_bytes(data: bytes, ecc_nsym: int):
    rs = RSCodec(ecc_nsym)
    try:
        dec = rs.decode(data)
        if isinstance(dec, tuple):
            return dec[0]
        return dec
    except:
        return None

# ---------- Hilbert 曲线 ----------
def generate_positions_hilbert(img_h: int, img_w: int, block_size: int, num_blocks: int):
    n_x = img_w // block_size
    n_y = img_h // block_size
    N = max(n_x, n_y)
    p = max(1, int(np.ceil(np.log2(N))))
    hilbert = HilbertCurve(p, 2)
    positions = []
    max_d = 2 ** (p*2)
    for d in range(max_d):
        if hasattr(hilbert, 'point_from_distance'):
            pt = hilbert.point_from_distance(d)
        else:
            pt = hilbert.coordinates_from_distance(d)
        i, j = pt
        if i < n_y and j < n_x:
            positions.append((i, j))
        if len(positions) >= num_blocks:
            break
    return positions

# ---------- 容量 ----------
def capacity_in_blocks(img_h, img_w):
    n_x = img_w // BLOCK_SIZE
    n_y = img_h // BLOCK_SIZE
    return n_x * n_y

# ---------- 自动调参 ----------
def auto_tune_parameters(img_h, img_w, secret_bytes,
                         start_redundancy=DEFAULT_REDUNDANCY,
                         start_ecc=DEFAULT_ECC_NSYM):
    comp = zlib.compress(secret_bytes)
    if len(comp) < len(secret_bytes):
        payload = comp
        used_compressed = True
    else:
        payload = secret_bytes
        used_compressed = False
    blocks_total = capacity_in_blocks(img_h, img_w)
    bits_per_block = BITS_PER_POSITION_BLOCK
    for red in range(start_redundancy, 0, -1):
        for ecc_try in range(start_ecc, 2, -4):
            enc_len_bytes = len(payload) + ecc_try
            enc_bits = enc_len_bytes * 8
            num_data_blocks = math.ceil(enc_bits / bits_per_block)
            metadata_bits = 72 + PHASH_BITS * num_data_blocks
            metadata_blocks_needed = math.ceil(metadata_bits / bits_per_block)
            total_blocks_needed = metadata_blocks_needed + num_data_blocks * red
            if total_blocks_needed <= blocks_total:
                return {
                    'payload': payload,
                    'used_compressed': used_compressed,
                    'ecc_nsym': ecc_try,
                    'redundancy': red,
                    'num_data_blocks': num_data_blocks,
                    'metadata_blocks': metadata_blocks_needed,
                    'total_blocks_needed': total_blocks_needed
                }
    payload2 = zlib.compress(payload, level=9)
    if len(payload2) < len(payload):
        payload = payload2
        used_compressed = True
        ecc_try = 4
        red = 1
        enc_len_bytes = len(payload) + ecc_try
        enc_bits = enc_len_bytes * 8
        num_data_blocks = math.ceil(enc_bits / bits_per_block)
        metadata_bits = 72 + PHASH_BITS * num_data_blocks
        metadata_blocks_needed = math.ceil(metadata_bits / bits_per_block)
        total_blocks_needed = metadata_blocks_needed + num_data_blocks * red
        if total_blocks_needed <= blocks_total:
            return {
                'payload': payload,
                'used_compressed': used_compressed,
                'ecc_nsym': ecc_try,
                'redundancy': red,
                'num_data_blocks': num_data_blocks,
                'metadata_blocks': metadata_blocks_needed,
                'total_blocks_needed': total_blocks_needed
            }
    return None

# ---------- 灰度提取函数 ----------
def extract_text_from_image(img_gray: np.ndarray):
    h, w = img_gray.shape
    blocks_total = capacity_in_blocks(h, w)
    positions = generate_positions_hilbert(h, w, BLOCK_SIZE, blocks_total)

    read_blocks = max(METADATA_RESERVED_MIN, int(blocks_total * 0.01))
    read_blocks = min(read_blocks, blocks_total)
    metadata_bits = ''
    for m_idx in range(read_blocks):
        pos = positions[m_idx]
        y = pos[0] * BLOCK_SIZE
        x = pos[1] * BLOCK_SIZE
        if y + BLOCK_SIZE > h or x + BLOCK_SIZE > w:
            metadata_bits += '0' * BITS_PER_POSITION_BLOCK
            continue
        block = img_gray[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
        metadata_bits += extract_bits_from_block(block, BITS_PER_POSITION_BLOCK)

    if len(metadata_bits) < 72:
        metadata_bits = metadata_bits.ljust(72, '0')
    try:
        total_bits = int(metadata_bits[0:32],2)
        num_data_blocks = int(metadata_bits[32:48],2)
        bpb = int(metadata_bits[48:56],2)
        ecc = int(metadata_bits[56:64],2)
        red = int(metadata_bits[64:72],2)
    except Exception as e:
        raise RuntimeError("Failed to parse metadata header.") from e

    phash_bits_needed = num_data_blocks * PHASH_BITS
    metadata_bits_needed = 72 + phash_bits_needed
    metadata_blocks_needed = math.ceil(metadata_bits_needed / BITS_PER_POSITION_BLOCK)
    if metadata_blocks_needed > read_blocks:
        for m_idx in range(read_blocks, min(metadata_blocks_needed, blocks_total)):
            pos = positions[m_idx]
            y = pos[0] * BLOCK_SIZE
            x = pos[1] * BLOCK_SIZE
            if y + BLOCK_SIZE > h or x + BLOCK_SIZE > w:
                metadata_bits += '0'*BITS_PER_POSITION_BLOCK
            else:
                block = img_gray[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
                metadata_bits += extract_bits_from_block(block, BITS_PER_POSITION_BLOCK)
    if len(metadata_bits) < metadata_bits_needed:
        metadata_bits = metadata_bits.ljust(metadata_bits_needed, '0')

    phashes = []
    ptr = 72
    for i in range(num_data_blocks):
        ph = metadata_bits[ptr:ptr+PHASH_BITS]
        ph = ph.ljust(PHASH_BITS, '0')
        phashes.append(ph)
        ptr += PHASH_BITS

    recovered_chunks = [''] * num_data_blocks
    confidences = [0.0] * num_data_blocks
    for data_idx in range(num_data_blocks):
        best_chunk = None
        best_conf = -1.0
        for r in range(red):
            pos_index = metadata_blocks_needed + data_idx + r*num_data_blocks
            if pos_index >= len(positions):
                continue
            pos = positions[pos_index]
            y = pos[0] * BLOCK_SIZE
            x = pos[1] * BLOCK_SIZE
            if y + BLOCK_SIZE > h or x + BLOCK_SIZE > w:
                continue
            block = img_gray[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
            chunk_bits = extract_bits_from_block(block, BITS_PER_POSITION_BLOCK)
            ph = phash_bits_from_block_bits(chunk_bits)
            sim = phash_hamming_sim(ph, phashes[data_idx])
            if sim > best_conf:
                best_conf = sim
                best_chunk = chunk_bits
        if best_chunk is None:
            recovered_chunks[data_idx] = '0' * BITS_PER_POSITION_BLOCK
            confidences[data_idx] = 0.0
        else:
            recovered_chunks[data_idx] = best_chunk
            confidences[data_idx] = best_conf

    combined_bits = ''.join(recovered_chunks)[:total_bits]
    combined_bytes = bits_to_bytes(combined_bits)
    dec = ecc_decode_bytes(combined_bytes, ecc)
    if dec is not None:
        try:
            text_bytes = dec
            try:
                text_bytes2 = zlib.decompress(text_bytes)
                text_bytes = text_bytes2
            except:
                pass
            text = bytes_to_text(text_bytes)
        except Exception:
            text = bytes_to_text(dec if isinstance(dec, bytes) else combined_bytes)
    else:
        text = bytes_to_text(combined_bytes)
    return text, confidences

# ---------- 彩色嵌入 ----------
def embed_text_in_image_color(img_color: np.ndarray, text: str):
    h, w, c = img_color.shape
    secret_bytes = text_to_bytes(text)
    meta_overall = None
    stego_channels = []
    for ch in range(c):
        tune = auto_tune_parameters(h, w, secret_bytes)
        if tune is None:
            raise ValueError("Image cannot contain the secret.")
        payload = tune['payload']
        used_compressed = tune['used_compressed']
        ecc_nsym = tune['ecc_nsym']
        redundancy = tune['redundancy']
        num_data_blocks = tune['num_data_blocks']
        metadata_blocks = tune['metadata_blocks']
        total_blocks_needed = tune['total_blocks_needed']

        enc = ecc_encode_bytes(payload, ecc_nsym)
        bitstream = bytes_to_bits(enc)
        total_bits = len(bitstream)
        usable_bits_per_block = BITS_PER_POSITION_BLOCK
        positions = generate_positions_hilbert(h, w, BLOCK_SIZE, total_blocks_needed)

        chunks = []
        chunk_phashes = []
        idx = 0
        for i in range(num_data_blocks):
            chunk = bitstream[idx:idx+usable_bits_per_block].ljust(usable_bits_per_block,'0')
            idx += usable_bits_per_block
            chunks.append(chunk)
            ph = phash_bits_from_block_bits(chunk)
            chunk_phashes.append(ph)
        header = ''.join([
            format(total_bits, '032b'),
            format(num_data_blocks, '016b'),
            format(usable_bits_per_block, '08b'),
            format(ecc_nsym, '08b'),
            format(redundancy, '08b')
        ])
        metadata_bits = header + ''.join(chunk_phashes)

        img_float = img_color[:,:,ch].astype(float).copy()
        bitptr = 0
        for m_idx in range(metadata_blocks):
            pos = positions[m_idx]
            y = pos[0] * BLOCK_SIZE
            x = pos[1] * BLOCK_SIZE
            if y + BLOCK_SIZE > h or x + BLOCK_SIZE > w:
                to_write = '0'*usable_bits_per_block
            else:
                block = img_float[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
                to_write = metadata_bits[bitptr:bitptr+usable_bits_per_block].ljust(usable_bits_per_block,'0')
                img_float[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = embed_bits_in_block(block, to_write[:usable_bits_per_block])
            bitptr += usable_bits_per_block
        for data_idx, chunk in enumerate(chunks):
            for r in range(redundancy):
                pos_index = metadata_blocks + data_idx + r*num_data_blocks
                if pos_index >= len(positions):
                    continue
                pos = positions[pos_index]
                y = pos[0]*BLOCK_SIZE
                x = pos[1]*BLOCK_SIZE
                if y + BLOCK_SIZE > h or x + BLOCK_SIZE > w:
                    continue
                block = img_float[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
                img_float[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = embed_bits_in_block(block, chunk)
        stego_channels.append(np.clip(img_float,0,255).astype(np.uint8))
        if meta_overall is None:
            meta_overall = {
                'used_compressed': used_compressed,
                'ecc_nsym': ecc_nsym,
                'redundancy': redundancy,
                'num_data_blocks': num_data_blocks,
                'metadata_blocks': metadata_blocks,
                'total_bits': total_bits,
                'total_blocks_needed': total_blocks_needed
            }
    stego = np.stack(stego_channels, axis=2)
    return stego, meta_overall

def extract_text_from_image_color(img_color: np.ndarray):
    return extract_text_from_image(img_color[:,:,0])

# ---------- 热图 ----------
def save_confidence_heatmap(confidences, img_shape, outpath):
    h, w = img_shape
    n_y = h // BLOCK_SIZE
    n_x = w // BLOCK_SIZE
    grid = np.zeros((n_y, n_x))
    positions = generate_positions_hilbert(h, w, BLOCK_SIZE, n_y*n_x)
    for idx, conf in enumerate(confidences):
        pos_index = METADATA_RESERVED_MIN + idx
        if pos_index >= len(positions):
            break
        pos = positions[pos_index]
        i,j = pos
        if i < n_y and j < n_x:
            grid[i,j] = conf
    up = np.kron(grid, np.ones((BLOCK_SIZE, BLOCK_SIZE)))
    plt.figure(figsize=(6,6))
    plt.imshow(up, vmin=0, vmax=1)
    plt.colorbar(label='confidence (0..1)')
    plt.title('Per-block confidence heatmap')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ---------- 示例运行 ----------
def example_run():
    inp = 'input.jpg'
    if not os.path.exists(inp):
        raise FileNotFoundError("请把一张图片命名为 input.jpg 放在当前目录后再运行脚本。")
    img = cv2.imread(inp, cv2.IMREAD_COLOR)
    secret = "Competition demo message: robust stego with adaptive parameters and pHash confidence." * 2

    stego, meta = embed_text_in_image_color(img, secret)
    print("Embed meta:", meta)
    cv2.imwrite('stego_with_meta_color.jpg', stego, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    cropped = stego[:, :stego.shape[1]//2]
    cv2.imwrite('stego_cropped_color.jpg', cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    recovered, confs = extract_text_from_image_color(cropped)
    print("Recovered text (from cropped):")
    print(recovered[:500])
    save_confidence_heatmap(confs, cropped.shape[:2], 'confidence_heatmap_color.png')
    print("Saved: stego_with_meta_color.jpg, stego_cropped_color.jpg, confidence_heatmap_color.png")

if __name__ == "__main__":
    example_run()
